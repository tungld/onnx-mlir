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
  bool enableParallel = false;

  ONNXExpandOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
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

    // Check if we can use block expansion optimization.
    // This is determined by the annotateONNXOps function using DimAnalysis.
    bool useSingleDimExpand = false;
    int expandedDim = -1;

    if (auto attr = expandOp->getAttrOfType<BoolAttr>("single_dim_expand")) {
      useSingleDimExpand = attr.getValue();
      if (useSingleDimExpand) {
        // Find which dimension is being expanded.
        expandedDim = findExpandedDimension(inputMemRefType, outputMemRefType);
      }
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

    // Create loops for dimensions before the expanded dimension.
    int64_t outerRank = expandedDim;
    ValueRange outerLoopDef = create->krnl.defineLoops(outerRank);
    SmallVector<IndexExpr, 4> outerLbs(outerRank, LitIE(0));
    SmallVector<IndexExpr, 4> outerUbs;
    for (int64_t i = 0; i < outerRank; ++i)
      outerUbs.emplace_back(inUBs[i]);

    if (enableParallel && outerRank > 0) {
      tryCreateKrnlParallel(create->krnl, op, "block expand", outerLoopDef,
          outerLbs, outerUbs, 0, 2, {}, 4);
    }

    // Create a loop for the expanded dimension.
    ValueRange expandLoopDef = create->krnl.defineLoops(1);
    SmallVector<IndexExpr, 4> expandLbs(1, LitIE(0));
    SmallVector<IndexExpr, 4> expandUbs(1, outUBs[expandedDim]);

    if (outerRank > 0) {
      create->krnl.iterateIE(outerLoopDef, outerLoopDef, outerLbs, outerUbs,
          [&](const KrnlBuilder &createKrnl, ValueRange outerIndices) {
            MultiDialectBuilder<MathBuilder, KrnlBuilder> create(createKrnl);
            IndexExprScope outerScope(createKrnl);

            // Compute source offset for the outer dimensions.
            IndexExpr srcOffsetIE = LitIE(0);
            for (int64_t i = 0; i < outerRank; ++i) {
              DimIndexExpr srcIndex(outerIndices[i]);
              srcOffsetIE = srcOffsetIE + srcIndex * SymIE(inStrides[i]);
            }

            // Compute base destination offset for outer dimensions (computed once).
            IndexExpr destBaseOffsetIE = LitIE(0);
            for (int64_t i = 0; i < outerRank; ++i) {
              DimIndexExpr outerIndex(outerIndices[i]);
              destBaseOffsetIE = destBaseOffsetIE + outerIndex * SymIE(outStrides[i]);
            }

            // Loop over the expanded dimension.
            create.krnl.iterateIE(expandLoopDef, expandLoopDef, expandLbs,
                expandUbs,
                [&](const KrnlBuilder &createKrnl, ValueRange expandIndices) {
                  MultiDialectBuilder<MathBuilder, KrnlBuilder> create(
                      createKrnl);
                  IndexExprScope expandScope(createKrnl);

                  // Compute destination offset by adding expand dimension contribution.
                  DimIndexExpr expandIndex(expandIndices[0]);
                  IndexExpr destOffsetIE =
                      SymIE(destBaseOffsetIE) +
                      expandIndex * SymIE(outStrides[expandedDim]);

                  // Call memcpy.
                  create.krnl.memcpy(outputMemRef, inputMemRef, elemsToCopyI64,
                      destOffsetIE.getValue(), srcOffsetIE.getValue());
                });
          });
    } else {
      // No outer dimensions, just loop over the expanded dimension.
      IndexExprScope scope(create->krnl);
      IndexExpr srcOffsetIE = LitIE(0);

      create->krnl.iterateIE(expandLoopDef, expandLoopDef, expandLbs, expandUbs,
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

  // Determine if expansion is along a single dimension only.
  // Returns the dimension index if true, -1 otherwise.
  int singleDimensionExpansion(MemRefType inputMemRefType,
      MemRefType outputMemRefType, ONNXExpandOpShapeHelper &shapeHelper) const {
    int64_t rank = inputMemRefType.getRank();
    ArrayRef<int64_t> inputShape = inputMemRefType.getShape();
    ArrayRef<int64_t> outputShape = outputMemRefType.getShape();

    // Count how many dimensions are being expanded.
    int expandedDim = -1;
    int numExpandedDims = 0;
    bool hasUnknownExpansion = false;

    // Use the static shape information from the MemRefType.
    // This is more reliable than the shape helper for detecting expansions.
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
      } else if (inputSize == -1 && outputSize == -1) {
        // Both are dynamic - assume they represent the same dimension (no
        // expansion). This is the common case where a dynamic dimension is
        // preserved.
        continue;
      } else {
        // One is static, one is dynamic - this is unusual and we can't
        // optimize.
        hasUnknownExpansion = true;
      }
    }

    // Only optimize if we have exactly one expanded dimension and no unknown
    // expansions. Dynamic dimensions that are the same in input and output are
    // allowed.
    if (numExpandedDims == 1 && !hasUnknownExpansion) {
      return expandedDim;
    }

    return -1;
  }
};

void populateLoweringONNXExpandOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXExpandOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
