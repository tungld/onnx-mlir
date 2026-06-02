/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Expand.cpp - Lowering Expand Op----------------------=== //
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Expand Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXExpandOpLowering : public OpConversionPattern<ONNXExpandOp> {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, MathBuilder>;
  DimAnalysis *dimAnalysis;
  bool enableParallel = false;

  ONNXExpandOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      DimAnalysis *dimAnalysis, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx), dimAnalysis(dimAnalysis) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXExpandOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ONNXExpandOp expandOp,
      ONNXExpandOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = expandOp.getOperation();
    Location loc = ONNXLoc<ONNXExpandOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value input = adaptor.getInput();

    MDBuilder create(rewriter, loc);

    // Get shape.
    ONNXExpandOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    MemRefType inputMemRefType = mlir::cast<MemRefType>(input.getType());
    int64_t outputRank = outputMemRefType.getRank();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Check if we can use block expansion optimization using DimAnalysis.
    bool useSingleDimExpand = false;
    int expandedDim = -1;

    if (dimAnalysis) {
      expandedDim = singleDimensionExpansion(
          inputMemRefType, outputMemRefType, input, expandOp.getOutput());
      useSingleDimExpand = (expandedDim >= 0);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Expand: useSingleDimExpand = " << useSingleDimExpand
                   << ", expandedDim = " << expandedDim << "\n";
      llvm::dbgs() << "Input shape: ";
      for (int64_t i = 0; i < inputMemRefType.getRank(); ++i)
        llvm::dbgs() << inputMemRefType.getShape()[i] << " ";
      llvm::dbgs() << "\nOutput shape: ";
      for (int64_t i = 0; i < outputMemRefType.getRank(); ++i)
        llvm::dbgs() << outputMemRefType.getShape()[i] << " ";
      llvm::dbgs() << "\n";
    });

    if (useSingleDimExpand && expandedDim >= 0) {
      // Use optimized block expansion with memcpy.
      blockExpand(op, input, alloc, expandedDim, &create, enableParallel);
    } else {
      // Fall back to element-wise expansion.
      // Iterate over the output values.
      ValueRange outputLoopDef = create.krnl.defineLoops(outputRank);
      LiteralIndexExpr zeroIE(0);
      SmallVector<IndexExpr, 4> lbs(outputRank, zeroIE);
      DimsExpr ubs = shapeHelper.getOutputDims();

      // Enable parallelism if required.
      if (enableParallel)
        tryCreateKrnlParallel(
            create.krnl, op, "expand", outputLoopDef, lbs, ubs, 0, 2, {}, 4);

      create.krnl.iterateIE(outputLoopDef, outputLoopDef, lbs, ubs,
          [&](const KrnlBuilder &createKrnl, ValueRange outputLoopInd) {
            IndexExprScope outputScope(createKrnl, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputLoopIndices, lhsAccessExprs;
            getIndexExprList<DimIndexExpr>(outputLoopInd, outputLoopIndices);
            LogicalResult res = shapeHelper.getAccessExprs(
                input, 0, outputLoopIndices, lhsAccessExprs);
            assert(succeeded(res) && "Could not compute access indices");
            Value val = createKrnl.loadIE(input, lhsAccessExprs);
            createKrnl.store(val, alloc, outputLoopInd);
          });
    }

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  // Do expand by copying blocks of consecutive elements using memcpy.
  // This is used when expansion is along a single dimension.
  void blockExpand(Operation *op, Value inputMemRef, Value outputMemRef,
      int expandedDim, MDBuilder *create, bool enableParallel) const {
    llvm::outs() << "tung, block expand\n";
    Type i64Ty = create->math.getBuilder().getI64Type();
    MemRefType inMemRefType = mlir::cast<MemRefType>(inputMemRef.getType());
    int64_t rank = inMemRefType.getRank();

    // Input and output upper bounds.
    SmallVector<IndexExpr, 4> inUBs;
    create->krnlIE.getShapeAsDims(inputMemRef, inUBs);
    SmallVector<IndexExpr, 4> outUBs;
    create->krnlIE.getShapeAsDims(outputMemRef, outUBs);

    // Compute the number of elements in the inner block to copy.
    // This is the product of all dimensions after the expanded dimension.
    IndexExpr elemsToCopy = LitIE(1);
    for (int64_t i = expandedDim + 1; i < rank; ++i)
      elemsToCopy = elemsToCopy * inUBs[i];

    // Convert to i64 for memcpy - this handles both static and dynamic
    // dimensions.
    Value elemsToCopyI64 = create->math.cast(i64Ty, elemsToCopy.getValue());

    // Compute strides for input and output.
    SmallVector<IndexExpr, 4> inStrides, outStrides;
    inStrides.resize_for_overwrite(rank);
    outStrides.resize_for_overwrite(rank);
    inStrides[rank - 1] = LitIE(1);
    outStrides[rank - 1] = LitIE(1);
    IndexExpr strideIE = LitIE(1);
    for (int i = rank - 2; i >= 0; --i) {
      strideIE = strideIE * inUBs[i + 1];
      inStrides[i] = strideIE;
    }
    strideIE = LitIE(1);
    for (int i = rank - 2; i >= 0; --i) {
      strideIE = strideIE * outUBs[i + 1];
      outStrides[i] = strideIE;
    }

    // Flatten ALL dimensions up to and including the expanded dimension for maximum parallelism.
    int64_t outerRank = expandedDim;
    
    // Compute total flattened size (outer dims * expanded dim).
    IndexExpr totalFlatSize = LitIE(1);
    for (int64_t i = 0; i < outerRank; ++i) {
      totalFlatSize = totalFlatSize * inUBs[i];
    }
    totalFlatSize = totalFlatSize * outUBs[expandedDim];

    // Create a single flattened loop for ALL dimensions.
    ValueRange flatLoopDef = create->krnl.defineLoops(1);
    SmallVector<IndexExpr, 4> flatLbs(1, LitIE(0));
    SmallVector<IndexExpr, 4> flatUbs(1, totalFlatSize);

    if (enableParallel) {
      tryCreateKrnlParallel(create->krnl, op, "block expand", flatLoopDef,
          flatLbs, flatUbs, 0, 2, {}, 4);
    }

    if (outerRank > 0) {
      create->krnl.iterateIE(flatLoopDef, flatLoopDef, flatLbs, flatUbs,
          [&](const KrnlBuilder &createKrnl, ValueRange flatIndices) {
            MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createKrnl);
            IndexExprScope flatScope(createKrnl);

            // Decompose flat index into all dimension indices (outer + expanded).
            DimIndexExpr flatIdx(flatIndices[0]);
            SmallVector<IndexExpr, 4> allIndices;
            allIndices.resize(outerRank + 1);
            IndexExpr remaining = flatIdx;
            
            // Decompose from rightmost to leftmost: expanded dim, then outer dims.
            IndexExpr expandDimSize = SymIE(outUBs[expandedDim]);
            allIndices[outerRank] = remaining % expandDimSize;
            remaining = remaining.floorDiv(expandDimSize);
            
            for (int64_t i = outerRank - 1; i >= 0; --i) {
              IndexExpr dimSize = SymIE(inUBs[i]);
              allIndices[i] = remaining % dimSize;
              remaining = remaining.floorDiv(dimSize);
            }

            // Compute source offset (only uses outer dimensions).
            IndexExpr srcOffsetIE = LitIE(0);
            for (int64_t i = 0; i < outerRank; ++i) {
              srcOffsetIE = srcOffsetIE + allIndices[i] * SymIE(inStrides[i]);
            }

            // Compute destination offset (uses outer + expanded dimension).
            IndexExpr destOffsetIE = LitIE(0);
            for (int64_t i = 0; i < outerRank; ++i) {
              destOffsetIE = destOffsetIE + allIndices[i] * SymIE(outStrides[i]);
            }
            destOffsetIE = destOffsetIE + allIndices[outerRank] * SymIE(outStrides[expandedDim]);

            // Call memcpy.
            create.krnl.memcpy(outputMemRef, inputMemRef, elemsToCopyI64,
                destOffsetIE.getValue(), srcOffsetIE.getValue());
          });
    } else {
      // No outer dimensions, just loop over the expanded dimension.
      IndexExprScope scope(create->krnl);
      IndexExpr srcOffsetIE = LitIE(0);

      create->krnl.iterateIE(flatLoopDef, flatLoopDef, flatLbs, flatUbs,
          [&](const KrnlBuilder &createKrnl, ValueRange expandIndices) {
            MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createKrnl);
            IndexExprScope expandScope(createKrnl);

            // Compute destination offset.
            DimIndexExpr expandIndex(expandIndices[0]);
            IndexExpr destOffsetIE =
                expandIndex * SymIE(outStrides[expandedDim]);

            // Call memcpy.
            create.krnl.memcpy(outputMemRef, inputMemRef, elemsToCopyI64,
                destOffsetIE.getValue(), srcOffsetIE.getValue());
          });
    }
  }

  // Find which dimension is being expanded (when single_dim_expand attribute is
  // set). Returns the dimension index, or -1 if not found.
  int findExpandedDimension(
      MemRefType inputMemRefType, MemRefType outputMemRefType) const {
    int64_t rank = inputMemRefType.getRank();
    ArrayRef<int64_t> inputShape = inputMemRefType.getShape();
    ArrayRef<int64_t> outputShape = outputMemRefType.getShape();

    for (int64_t i = 0; i < rank; ++i) {
      int64_t inputSize = inputShape[i];
      int64_t outputSize = outputShape[i];

      // Check if this dimension is being expanded (static case).
      if (inputSize >= 0 && outputSize >= 0 && inputSize != outputSize) {
        return i;
      }
    }

    return -1;
  }

  // Determine if expansion is along a single dimension only using DimAnalysis.
  // Returns the dimension index if true, -1 otherwise.
  int singleDimensionExpansion(MemRefType inputMemRefType,
      MemRefType outputMemRefType, Value input, Value output) const {
    int64_t rank = inputMemRefType.getRank();
    ArrayRef<int64_t> inputShape = inputMemRefType.getShape();
    ArrayRef<int64_t> outputShape = outputMemRefType.getShape();

    // Count how many dimensions are being expanded.
    int expandedDim = -1;
    int numExpandedDims = 0;
    bool hasUnknownExpansion = false;

    for (int64_t i = 0; i < rank; ++i) {
      int64_t inputSize = inputShape[i];
      int64_t outputSize = outputShape[i];

      // Check if this dimension is being expanded.
      if (inputSize >= 0 && outputSize >= 0) {
        // Both dimensions are static.
        if (inputSize != outputSize) {
          expandedDim = i;
          numExpandedDims++;
        }
      } else if (inputSize == ShapedType::kDynamic && outputSize == ShapedType::kDynamic) {
        // Both are dynamic - use DimAnalysis to check if they're the same.
        if (dimAnalysis && !dimAnalysis->sameDynDim(input, i, output, i)) {
          // Dynamic dimensions are different - this is an expansion.
          expandedDim = i;
          numExpandedDims++;
        }
      } else {
        // One is static, one is dynamic - this is unusual and we can't optimize.
        hasUnknownExpansion = true;
      }
    }

    // Only optimize if we have exactly one expanded dimension and no unknown expansions.
    if (numExpandedDims == 1 && !hasUnknownExpansion) {
      return expandedDim;
    }

    return -1;
  }
};

void populateLoweringONNXExpandOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis,
    bool enableParallel) {
  patterns.insert<ONNXExpandOpLowering>(
      typeConverter, ctx, dimAnalysis, enableParallel);
}

} // namespace onnx_mlir
