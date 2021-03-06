/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXConstProp.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to constprop an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the constpropd operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

#include "onnx-mlir/Runtime/OMTensor.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Runtime/OMTensorHelper.h"

#include "llvm/ADT/SmallPtrSet.h"
#include <math.h>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Instructions to add a constant operation.
//===----------------------------------------------------------------------===//
// There is currently support for adding constant propagation for unary and
// binary athythmetic ops (binary ops support broadcast). To add an operation,
// you simply have to add a templated method on how to compute the result in
// terms of one or two inputs.
//
// The methods are:
//
// ElementWiseBinaryOpImpl and ElementWiseUnaryOpImpl
// and they need to be templated with an ONNX Operation (presuably).
//
// Then you need to add rules on how to transform the patterns; look into
// ConstProp.td for example.
//

const StringRef CONSTANT_ID_ATTR_NAME = "constantID";
const StringRef CONSTANT_USERS_ATTR_NAME = "users";

/// A helper function to get a value of a given type from an attribute.
template <typename T>
T getAttrValue(Attribute attr) {
  llvm_unreachable("unknown operation");
}

template <>
double getAttrValue(Attribute attr) {
  return attr.cast<FloatAttr>().getValueAsDouble();
}

template <>
float getAttrValue(Attribute attr) {
  return (float)attr.cast<FloatAttr>().getValueAsDouble();
}

template <>
int64_t getAttrValue(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

template <>
int32_t getAttrValue(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

//===----------------------------------------------------------------------===//
// Code for a constant pool to store intermediate constants.
//===----------------------------------------------------------------------===//
struct ConstantPool {
  static unsigned header;
  static DenseMap<unsigned, OMTensor *> storage;
  static OMTensor *get(unsigned id);
  static unsigned put(OMTensor *omt);
  static unsigned createOrGet(PatternRewriter &rewriter, Operation *op);
  static bool erase(unsigned id);
  static bool update(PatternRewriter &rewriter, Operation *op);
  static void reset();
};
unsigned ConstantPool::header = 0;
DenseMap<unsigned, OMTensor *> ConstantPool::storage;

/// Get a constant by index from the constant pool.
OMTensor *ConstantPool::get(unsigned id) {
  auto it = ConstantPool::storage.find(id);
  if (it != ConstantPool::storage.end())
    return it->second;
  else
    return nullptr;
}

/// Put a constant to the constant pool.
/// Return the constant index in the pool.
unsigned ConstantPool::put(OMTensor *omt) {
  unsigned constantID = ConstantPool::header;
  ConstantPool::storage.insert({constantID, omt});
  // Move header forward.
  ConstantPool::header += 1;
  return constantID;
}

void ConstantPool::reset() { ConstantPool::header = 0; }

/// Erase a constant by index.
bool ConstantPool::erase(unsigned id) {
  OMTensor *omt = ConstantPool::get(id);
  if (omt) {
    ConstantPool::storage.erase(id);
    omTensorDestroy(omt);
    return true;
  }
  return false;
}

/// Update an ONNXConstantOp.
/// Erase its constant from the constant pool if the op has no other uses.
/// Otherwise, update the ops's ref-count.
bool ConstantPool::update(PatternRewriter &rewriter, Operation *op) {
  bool updated = false;
  Attribute constantIDAttr =
      op->getAttrOfType<::mlir::Attribute>(CONSTANT_ID_ATTR_NAME);
  if (constantIDAttr) {
    unsigned constantID = constantIDAttr.cast<IntegerAttr>().getUInt();
    Attribute refCountAttr =
        op->getAttrOfType<::mlir::Attribute>(CONSTANT_USERS_ATTR_NAME);
    if (refCountAttr) {
      int refCount = refCountAttr.cast<IntegerAttr>().getUInt();
      if (refCount == 1)
        ConstantPool::erase(constantID);
      else
        op->setAttr(CONSTANT_USERS_ATTR_NAME,
            IntegerAttr::get(
                rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false),
                refCount - 1));
      updated = true;
    }
  }
  return updated;
}

/// Create or get a constant in the constant pool for a given ONNXConstantOp.
/// Return the constant index in the pool.
unsigned ConstantPool::createOrGet(PatternRewriter &rewriter, Operation *op) {
  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  assert(constOp && "Not a constant operation");

  Attribute constantIDAttr =
      op->getAttrOfType<::mlir::Attribute>(CONSTANT_ID_ATTR_NAME);
  unsigned constantID;
  if (constantIDAttr) {
    // The OMTensor for this constant op existed in the constant pool.
    // Just get its index in the constant pool.
    constantID = constantIDAttr.cast<IntegerAttr>().getUInt();
  } else {
    // Create a new OMTensor, and add it to the constant pool.
    DenseElementsAttr dataAttr =
        op->getAttrOfType<::mlir::Attribute>("value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>();
    Type elementType = dataAttr.getType().cast<ShapedType>().getElementType();
    std::vector<int64_t> shape =
        dataAttr.getType().cast<ShapedType>().getShape();

    OMTensor *resOmt;
    // copy data.
    if (elementType.isa<FloatType>()) {
      auto it = dataAttr.getValues<FloatAttr>().begin();
      FloatType floatTy = elementType.cast<FloatType>();
      if (floatTy.getWidth() == 32) {
        resOmt = omTensorCreateWithShape<float>(shape);
        for (int64_t i = 0; i < omTensorGetNumElems(resOmt); ++i) {
          omTensorGetElemByOffset<float>(resOmt, i) =
              getAttrValue<float>(*it++);
        }
      } else if (floatTy.getWidth() == 64) {
        resOmt = omTensorCreateWithShape<double>(shape);
        for (int64_t i = 0; i < omTensorGetNumElems(resOmt); ++i) {
          omTensorGetElemByOffset<double>(resOmt, i) =
              getAttrValue<double>(*it++);
        }
      } else
        llvm_unreachable("Unsupported data type");
    } else if (elementType.isa<IntegerType>()) {
      auto it = dataAttr.getValues<IntegerAttr>().begin();
      IntegerType intTy = elementType.cast<IntegerType>();
      if (intTy.getWidth() == 32) {
        resOmt = omTensorCreateWithShape<int32_t>(shape);
        for (int64_t i = 0; i < omTensorGetNumElems(resOmt); ++i) {
          omTensorGetElemByOffset<int32_t>(resOmt, i) =
              getAttrValue<int32_t>(*it++);
        }
      } else if (intTy.getWidth() == 64) {
        resOmt = omTensorCreateWithShape<int64_t>(shape);
        for (int64_t i = 0; i < omTensorGetNumElems(resOmt); ++i) {
          omTensorGetElemByOffset<int64_t>(resOmt, i) =
              getAttrValue<int64_t>(*it++);
        }
      } else
        llvm_unreachable("Unsupported data type");
    } else
      llvm_unreachable("Unsupported data type");

    constantID = ConstantPool::put(resOmt);
    // Insert an Attribute to the constant op for keeping the constant index
    // of the newly created OMTensor.
    op->setAttr(CONSTANT_ID_ATTR_NAME,
        IntegerAttr::get(
            rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false),
            constantID));
    // Set the number of users.
    int userCount = 0;
    for (auto u : constOp.getResult().getUsers())
      userCount++;
    op->setAttr(CONSTANT_USERS_ATTR_NAME,
        IntegerAttr::get(
            rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false),
            userCount));
  }
  return constantID;
}

/// A helper function to contruct a RankedTensorType from a ShapedType.
RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

///  A helper function to construct a DenseElementsAttr from an OMTensor.
static DenseElementsAttr createDenseElementsAttr(
    OMTensor *omt, ShapedType outputType) {
  RankedTensorType resType = constructRankedTensorType(outputType);
  OM_DATA_TYPE dtype = omTensorGetDataType(omt);
  int64_t numElements = omTensorGetNumElems(omt);
  int rank = omTensorGetRank(omt);
  // FloatType
  if (resType.getElementType().isa<FloatType>()) {
    FloatType floatTy = resType.getElementType().cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      float *res = (float *)omTensorGetDataPtr(omt);
      std::vector<float> resVector(res, res + numElements);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
    if (floatTy.getWidth() == 64) {
      double *res = (double *)omTensorGetDataPtr(omt);
      std::vector<double> resVector(res, res + numElements);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
  }

  // IntegerType
  if (resType.getElementType().isa<IntegerType>()) {
    IntegerType intTy = resType.getElementType().cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      int32_t *res = (int32_t *)omTensorGetDataPtr(omt);
      std::vector<int32_t> resVector(res, res + numElements);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
    if (intTy.getWidth() == 64) {
      int64_t *res = (int64_t *)omTensorGetDataPtr(omt);
      std::vector<int64_t> resVector(res, res + numElements);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
  }

  llvm_unreachable("Unknown data type");
  return DenseElementsAttr();
}

ONNXConstantOp CreateDenseONNXConstantOp(
    PatternRewriter &rewriter, Value replacingValue, unsigned constantID) {
  Location loc = replacingValue.getLoc();
  Type outputType = replacingValue.getType();
  ArrayRef<int64_t> shape = outputType.cast<ShapedType>().getShape();
  Type elementType = outputType.cast<ShapedType>().getElementType();

  int64_t elementCount = 1;
  for (int i = 0; i < shape.size(); ++i)
    elementCount *= shape[i];

  // A DenseElementsAttr is just to make ONNXConstantOp legal. We don't use
  // its value for computation. Real value will be obtained from the constant
  // pool. This DenseElementsAttr is created so that it consumes memory as
  // little as possbile.
  DenseElementsAttr denseAttr;
  RankedTensorType denseType =
      constructRankedTensorType(outputType.cast<ShapedType>());
  if (elementType.isa<FloatType>()) {
    // FloatType
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      std::vector<float> data(elementCount, 0.0);
      denseAttr = mlir::SplatElementsAttr::get<float>(
          denseType, llvm::makeArrayRef(data));
    } else if (floatTy.getWidth() == 64) {
      std::vector<double> data(elementCount, 0.0);
      denseAttr = mlir::SplatElementsAttr::get<double>(
          denseType, llvm::makeArrayRef(data));
    } else
      llvm_unreachable("Upsupported data type");
  } else if (elementType.isa<IntegerType>()) {
    // IntegerType
    IntegerType intTy = elementType.cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      std::vector<int32_t> data(elementCount, 0);
      denseAttr = mlir::SplatElementsAttr::get<int32_t>(
          denseType, llvm::makeArrayRef(data));
    } else if (intTy.getWidth() == 64) {
      std::vector<int64_t> data(elementCount, 0);
      denseAttr = mlir::SplatElementsAttr::get<int64_t>(
          denseType, llvm::makeArrayRef(data));
    } else
      llvm_unreachable("Upsupported data type");
  } else
    llvm_unreachable("Upsupported data type");

  ONNXConstantOp constOp =
      rewriter.create<ONNXConstantOp>(loc, outputType, Attribute(), denseAttr);

  // Set constant index in the constant pool.
  constOp.getOperation()->setAttr(CONSTANT_ID_ATTR_NAME,
      IntegerAttr::get(
          rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false),
          constantID));
  // Set the number of users.
  int userCount = 0;
  for (auto u : replacingValue.getUsers())
    userCount++;
  constOp.getOperation()->setAttr(CONSTANT_USERS_ATTR_NAME,
      IntegerAttr::get(
          rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false),
          userCount));
  return constOp;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for binary in presence of broadcast.
//===----------------------------------------------------------------------===//

// Template to generate binary operation results. It takes as inupt
// the element type as well as the two element attributes for the
// operation, and return the result of the operation.

template <typename OP, typename T>
struct ElementWiseBinaryOpImpl {
  static T impl(T lhs, T rhs) { llvm_unreachable("unknown operation"); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXAddOp, T> {
  static T impl(T lhs, T rhs) { return (lhs + rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXSubOp, T> {
  static T impl(T lhs, T rhs) { return (lhs - rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXMulOp, T> {
  static T impl(T lhs, T rhs) { return (lhs * rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXDivOp, T> {
  static T impl(T lhs, T rhs) { return (lhs / rhs); }
};

template <typename OP, typename T>
T ComputeConstPropElementwiseBinary(T lhs, T rhs) {
  return ElementWiseBinaryOpImpl<OP, T>::impl(lhs, rhs);
}

template <typename ElementwiseBinaryOp, typename T>
OMTensor *IterateConstPropElementwiseBinary(
    OMTensor *lhsOmt, OMTensor *rhsOmt, Type outputType) {
  auto outputShape = outputType.cast<ShapedType>().getShape();
  int64_t *lhsShape = omTensorGetShape(lhsOmt);
  int64_t *rhsShape = omTensorGetShape(rhsOmt);
  int lhsRank = omTensorGetRank(lhsOmt);
  int rhsRank = omTensorGetRank(rhsOmt);
  int outputRank = outputShape.size();

  // Check broadcasting.
  bool broadcasting = false;
  if (lhsRank != rhsRank)
    broadcasting = true;
  else
    for (int i = 0; i < outputRank; ++i)
      if (*(lhsShape + i) != *(rhsShape + i)) {
        broadcasting = true;
        break;
      }

  // Initialize a result OMTensor.
  auto resOmt = omTensorCreateWithShape<T>(outputShape);
  for (const auto &outputIndices : omTensorComputeIndexSet(resOmt)) {
    // Compute indices to access inputs.
    std::vector<int64_t> lhsIndices, rhsIndices;
    if (!broadcasting)
      for (int k = 0; k < outputRank; ++k) {
        lhsIndices.emplace_back(outputIndices[k]);
        rhsIndices.emplace_back(outputIndices[k]);
      }
    else
      for (int k = 0; k < outputRank; ++k) {
        // in the lhs index range.
        if (k >= outputRank - lhsRank) {
          int lhsIndex = k - outputRank + lhsRank;
          if (*(lhsShape + lhsIndex) == 1)
            // broadcast
            lhsIndices.emplace_back(0);
          else
            lhsIndices.emplace_back(outputIndices[k]);
        }
        // in the rhs index range.
        if (k >= outputRank - rhsRank) {
          int rhsIndex = k - outputRank + rhsRank;
          if (*(rhsShape + rhsIndex) == 1)
            // broadcast
            rhsIndices.emplace_back(0);
          else
            rhsIndices.emplace_back(outputIndices[k]);
        }
      }

    // Calculate element-wise binary result.
    T lhsValue = omTensorGetElem<T>(lhsOmt, lhsIndices);
    T rhsValue = omTensorGetElem<T>(rhsOmt, rhsIndices);
    omTensorGetElem<T>(resOmt, outputIndices) =
        ComputeConstPropElementwiseBinary<ElementwiseBinaryOp, T>(
            lhsValue, rhsValue);
  }
  return resOmt;
}

/// Do element-wise binary calculation of 'lhs' and 'rhs' values and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseBinaryOp>
ONNXConstantOp ConstPropElementwiseBinary(
    PatternRewriter &rewriter, Value replacingValue, Value lhs, Value rhs) {
  Type outputType = replacingValue.getType();
  Type elementType = outputType.cast<ShapedType>().getElementType();
  Operation *lhsOp = lhs.getDefiningOp();
  Operation *rhsOp = rhs.getDefiningOp();

  // Get lhs and rhs values as OMTensors.
  unsigned lhsId = ConstantPool::createOrGet(rewriter, lhsOp);
  OMTensor *lhsOmt = ConstantPool::get(lhsId);
  assert(lhsOmt && "LHS null pointer");
  unsigned rhsId = ConstantPool::createOrGet(rewriter, rhsOp);
  OMTensor *rhsOmt = ConstantPool::get(rhsId);
  assert(rhsOmt && "RHS null pointer");

  // Do calculation.
  OMTensor *resOmt;
  if (elementType.isa<FloatType>()) {
    // FloatType
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      resOmt = IterateConstPropElementwiseBinary<ElementwiseBinaryOp, float>(
          lhsOmt, rhsOmt, outputType);
    }
    if (floatTy.getWidth() == 64) {
      resOmt = IterateConstPropElementwiseBinary<ElementwiseBinaryOp, double>(
          lhsOmt, rhsOmt, outputType);
    }
  } else if (elementType.isa<IntegerType>()) {
    // IntegerType
    IntegerType intTy = elementType.cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      resOmt = IterateConstPropElementwiseBinary<ElementwiseBinaryOp, int32_t>(
          lhsOmt, rhsOmt, outputType);
    }
    if (intTy.getWidth() == 64) {
      resOmt = IterateConstPropElementwiseBinary<ElementwiseBinaryOp, int64_t>(
          lhsOmt, rhsOmt, outputType);
    }
  } else
    llvm_unreachable("Unknown data type");

  // Add the result to the constant pool.
  unsigned constantID = ConstantPool::put(resOmt);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      CreateDenseONNXConstantOp(rewriter, replacingValue, constantID);

  // Clean up memory for lhs and rhs if they are only used here.
  ConstantPool::update(rewriter, lhsOp);
  ConstantPool::update(rewriter, rhsOp);

  return res;
}

//===----------------------------------------------------------------------===//
//// Code to perform constant propagation for unary operation.
//===----------------------------------------------------------------------===//

template <typename OP, typename T>
struct ElementWiseUnaryOpImpl {
  static T impl(T val) { llvm_unreachable("unknown operation"); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXNegOp, T> {
  static T impl(T val) { return (-val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXSqrtOp, T> {
  static T impl(T val) { return sqrt(val); }
};
template <typename OP, typename T>
T ComputeConstPropElementwiseUnary(T val) {
  return ElementWiseUnaryOpImpl<OP, T>::impl(val);
}

template <typename ElementwiseUnaryOp, typename T>
OMTensor *IterateConstPropElementwiseUnary(OMTensor *omt, Type outputType) {
  // Initialize a result OMTensor.
  auto outputShape = outputType.cast<ShapedType>().getShape();
  OMTensor *resOmt = omTensorCreateWithShape<T>(outputShape);
  // Calculate element-wise unary result.
  for (const auto &idx : omTensorComputeIndexSet(resOmt)) {
    T value = omTensorGetElem<T>(omt, idx);
    omTensorGetElem<T>(resOmt, idx) =
        ComputeConstPropElementwiseUnary<ElementwiseUnaryOp, T>(value);
  }
  return resOmt;
}

/// Do element-wise unary calculation of 'input' value and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseUnaryOp>
ONNXConstantOp ConstPropElementwiseUnary(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  Type outputType = replacingValue.getType();
  Type elementType = outputType.cast<ShapedType>().getElementType();
  Operation *inputOp = input.getDefiningOp();

  // Get input value as OMTensor.
  unsigned inputId = ConstantPool::createOrGet(rewriter, inputOp);
  OMTensor *inputOmt = ConstantPool::get(inputId);

  // Do calculation.
  OMTensor *resOmt;
  if (elementType.isa<FloatType>()) {
    // FloatType
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      resOmt = IterateConstPropElementwiseUnary<ElementwiseUnaryOp, float>(
          inputOmt, outputType);
    }
    if (floatTy.getWidth() == 64) {
      resOmt = IterateConstPropElementwiseUnary<ElementwiseUnaryOp, double>(
          inputOmt, outputType);
    }
  } else if (elementType.isa<IntegerType>()) {
    // IntegerType
    IntegerType intTy = elementType.cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      resOmt = IterateConstPropElementwiseUnary<ElementwiseUnaryOp, int32_t>(
          inputOmt, outputType);
    }
    if (intTy.getWidth() == 64) {
      resOmt = IterateConstPropElementwiseUnary<ElementwiseUnaryOp, int64_t>(
          inputOmt, outputType);
    }
  } else
    llvm_unreachable("Unknown data type");

  // Add the result to the constant pool.
  unsigned constantID = ConstantPool::put(resOmt);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      CreateDenseONNXConstantOp(rewriter, replacingValue, constantID);

  // Clean up memory for input if it is only used here.
  ConstantPool::update(rewriter, inputOp);

  return res;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for transpose.
//===----------------------------------------------------------------------===//

void RecurseConstPropTranspose(PatternRewriter &rewriter,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, SmallVector<uint64_t, 4> &perm,
    int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    auto res = attr.getValue(ArrayRef<uint64_t>(indices));
    resVector.emplace_back(res);
  } else {
    // Recurse.
    auto shape = attr.getType().getShape();
    int rank = shape.size();
    int index = perm[rank - freeRank];
    int size = attr.getType().getShape()[index];
    for (int i = 0; i < size; ++i) {
      indices[index] = i;
      RecurseConstPropTranspose(
          rewriter, resVector, attr, indices, perm, freeRank - 1);
    }
  }
}

DenseElementsAttr ConstPropTranspose(PatternRewriter &rewriter,
    Value resOperand, Attribute attr, ArrayAttr permAttr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());
  auto rank = denseAttr.getType().getShape().size();
  // Read permute vector.
  SmallVector<uint64_t, 4> perm;
  assert(permAttr && "permute attribute expected to be defined here");
  for (auto permVal : permAttr.getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());
  // Init indice vector.
  SmallVector<uint64_t, 4> indices(rank, 0);
  std::vector<Attribute> resVector;
  // Copy using permute order.
  RecurseConstPropTranspose(
      rewriter, resVector, denseAttr, indices, perm, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unsqueeze.
//===----------------------------------------------------------------------===//

template <typename T>
OMTensor *IterateConstPropUnsqueeze(OMTensor *omt, Type outputType) {
  // Initialize a result OMTensor.
  auto outputShape = outputType.cast<ShapedType>().getShape();
  OMTensor *resOmt = omTensorCreateWithShape<T>(outputShape);
  for (int64_t i = 0; i < omTensorGetNumElems(resOmt); ++i) {
    T value = omTensorGetElemByOffset<T>(omt, i);
    omTensorGetElemByOffset<T>(resOmt, i) = value;
  }
  return resOmt;
}

ONNXConstantOp ConstPropUnsqueeze(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  Type outputType = replacingValue.getType();
  Type elementType = outputType.cast<ShapedType>().getElementType();
  Operation *inputOp = input.getDefiningOp();

  // Get input value as OMTensor.
  unsigned inputId = ConstantPool::createOrGet(rewriter, inputOp);
  OMTensor *inputOmt = ConstantPool::get(inputId);

  // Do calculation.
  OMTensor *resOmt;
  if (elementType.isa<FloatType>()) {
    // FloatType
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      resOmt = IterateConstPropUnsqueeze<float>(inputOmt, outputType);
    }
    if (floatTy.getWidth() == 64) {
      resOmt = IterateConstPropUnsqueeze<double>(inputOmt, outputType);
    }
  } else if (elementType.isa<IntegerType>()) {
    // IntegerType
    IntegerType intTy = elementType.cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      resOmt = IterateConstPropUnsqueeze<int32_t>(inputOmt, outputType);
    }
    if (intTy.getWidth() == 64) {
      resOmt = IterateConstPropUnsqueeze<int64_t>(inputOmt, outputType);
    }
  } else
    llvm_unreachable("Unknown data type");

  // Add the result to the constant pool.
  unsigned constantID = ConstantPool::put(resOmt);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      CreateDenseONNXConstantOp(rewriter, replacingValue, constantID);

  // Clean up memory for input if it is only used here.
  ConstantPool::update(rewriter, inputOp);

  return res;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//

void RecurseConstPropSplit(PatternRewriter &rewriter,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, uint64_t splitAxis, uint64_t axisOffset,
    uint64_t axisSize, int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    Attribute res = attr.getValue(ArrayRef<uint64_t>(indices));
    resVector.emplace_back(res);
  } else {
    // Recurse.
    ArrayRef<int64_t> shape = attr.getType().getShape();
    int rank = shape.size();
    int index = rank - freeRank;
    int start, size;
    if (index == splitAxis) {
      start = axisOffset;
      size = axisSize;
    } else {
      start = 0;
      size = attr.getType().getShape()[index];
    }
    for (int i = start; i < start + size; ++i) {
      indices[index] = i;
      RecurseConstPropSplit(rewriter, resVector, attr, indices, splitAxis,
          axisOffset, axisSize, freeRank - 1);
    }
  }
}

DenseElementsAttr ConstPropSplit(PatternRewriter &rewriter, Value resOperand,
    Attribute attr, IntegerAttr axisAttr, ArrayAttr splitAttr,
    unsigned resIndex) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());
  unsigned rank = denseAttr.getType().getShape().size();
  // Read split axis.
  uint64_t splitAxis = axisAttr.getValue().getSExtValue();
  // Read split vector.
  SmallVector<uint64_t, 4> splits;
  assert(splitAttr && "split attribute expected to be defined here");
  for (Attribute splitVal : splitAttr.getValue())
    splits.emplace_back(splitVal.cast<IntegerAttr>().getInt());
  // Compute the range of elements of interest in the given axis.
  uint64_t axisOffset = 0, axisSize = splits[resIndex];
  for (int i = 0; i < resIndex; ++i)
    axisOffset += splits[i];
  // Init indice vector.
  SmallVector<uint64_t, 4> indices(rank, -1);
  std::vector<Attribute> resVector;
  // Copy.
  RecurseConstPropSplit(rewriter, resVector, denseAttr, indices, splitAxis,
      axisOffset, axisSize, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

class ConstPropSplitPattern : public OpRewritePattern<ONNXSplitOp> {
public:
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // A dense attribute that contains constant values of the split op's
    // input.
    Attribute denseAttr;

    // Match
    ONNXSplitOp *splitOp = ::llvm::dyn_cast_or_null<::mlir::ONNXSplitOp>(&op);
    {
      Operation *producerOp = splitOp->input().getDefiningOp();
      ONNXConstantOp castedProducerOp =
          ::llvm::dyn_cast_or_null<::mlir::ONNXConstantOp>(producerOp);
      if (!castedProducerOp)
        return failure();
      // Check whether the constant op is using a dense value or not.
      Attribute sparseAttr =
          producerOp->getAttrOfType<::mlir::Attribute>("sparse_value");
      if (sparseAttr)
        return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
          diag << "entities '' failed to satisfy constraint: Attribute "
                  "is null";
        });
      Attribute dataAttr =
          producerOp->getAttrOfType<::mlir::Attribute>("value");
      denseAttr = dataAttr;
    }

    // Rewrite
    unsigned outputNum = splitOp->getNumResults();
    Value splitInput = splitOp->input();
    int64_t rank = splitInput.getType().cast<ShapedType>().getRank();
    IntegerAttr axisAttr = splitOp->axisAttr();
    ArrayAttr splitAttr = splitOp->splitAttr();
    if (!splitAttr) {
      // If split attribute is not specified, it is constructed from input.
      ArrayRef<int64_t> shape =
          splitInput.getType().cast<ShapedType>().getShape();
      uint64_t splitAxis = axisAttr.getValue().getSExtValue();
      assert(shape[splitAxis] % outputNum == 0 &&
             "The dimension at the split axis is expected to be divisible by "
             "the number of results");
      Attribute splitSize = rewriter.getIntegerAttr(
          rewriter.getIntegerType(64), shape[splitAxis] / outputNum);
      SmallVector<Attribute, 4> splits(outputNum, splitSize);
      splitAttr = rewriter.getArrayAttr(splits);
    }

    SmallVector<::mlir::Value, 4> returnValues;
    for (int i = 0; i < outputNum; ++i) {
      Value splitOutput = splitOp->getResults()[i];
      Value constOp =
          rewriter.create<ONNXConstantOp>(loc, splitOutput.getType(),
              /*sparse_value=*/Attribute(),
              /*dense_value=*/
              ConstPropSplit(
                  rewriter, splitOutput, denseAttr, axisAttr, splitAttr, i));
      returnValues.emplace_back(constOp);
    }

    rewriter.replaceOp(op, returnValues);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern definition.
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/ONNXConstProp.inc"

//===----------------------------------------------------------------------===//
// Code to manage the pass.
//===----------------------------------------------------------------------===//

struct ConstPropONNXToONNXPass
    : public PassWrapper<ConstPropONNXToONNXPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void ConstPropONNXToONNXPass::runOnFunction() {
  auto function = getFunction();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect>();

  OwningRewritePatternList patterns;
  populateWithGenerated(context, patterns);
  patterns.insert<ConstPropSplitPattern>(&getContext());

  applyPatternsAndFoldGreedily(function, std::move(patterns));

  // Create DenseElementsAttr and clean up helper attributes.
  function.walk([&](ONNXConstantOp constOp) {
    Operation *op = constOp.getOperation();
    Attribute constantIDAttr =
        op->getAttrOfType<::mlir::Attribute>(CONSTANT_ID_ATTR_NAME);
    if (constantIDAttr) {
      unsigned constantID = constantIDAttr.cast<IntegerAttr>().getUInt();
      OMTensor *omt = ConstantPool::get(constantID);
      ShapedType outputType = constOp.getResult().getType().cast<ShapedType>();
      DenseElementsAttr denseAttr = createDenseElementsAttr(omt, outputType);
      op->setAttr("value", denseAttr);
      op->removeAttr(CONSTANT_ID_ATTR_NAME);
      op->removeAttr(CONSTANT_USERS_ATTR_NAME);
      ConstantPool::erase(constantID);
    }
  });
  // TODO: clean up the constant pool.
  // ConstantPool::reset();
} // end anonymous namespace

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
