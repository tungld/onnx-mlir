/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Concat Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConcatOpLowering : public OpConversionPattern<ONNXConcatOp> {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, MathBuilder>;
  ONNXConcatOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXConcatOp::getOperationName());
  }

  bool enableParallel = false;

  LogicalResult matchAndRewrite(ONNXConcatOp concatOp,
      ONNXConcatOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = concatOp.getOperation();
    Location loc = ONNXLoc<ONNXConcatOp>(op);
    ValueRange operands = adaptor.getOperands();

    // Gather info.
    MDBuilder create(rewriter, loc);

    // Get shape.
    ONNXConcatOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type outputTensorType = *op->result_type_begin();
    Type convertedType = typeConverter->convertType(outputTensorType);
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Alloc and dealloc.
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Determine if using block copy is benifical.
    bool useBlockCopy = true;

    if (useBlockCopy) {
      genBlockConcat(
          create, concatOp, operands, shapeHelper, alloc, enableParallel);
    } else {
      genScalarConcat(
          create, concatOp, operands, shapeHelper, alloc, enableParallel);
    }

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  void genScalarConcat(MDBuilder &create, ONNXConcatOp concatOp,
      ValueRange operands, ONNXConcatOpShapeHelper &shapeHelper, Value alloc,
      bool enableParallel) const {
    Operation *op = concatOp.getOperation();
    unsigned int inputNum = operands.size();
    int64_t axis = concatOp.getAxis();
    assert(axis >= 0 && "negative axis is supposed to have been normalized");

    // Creates loops, one for each input.
    // Since the each input should have same size for each dimension(except
    // axis), we will try to make the loop upper bound the same for further
    // optimization. Difference may come from constant vs. dynamic, or dynamic
    // dim of different inputs.
    SmallVector<IndexExpr, 4> commonUB(shapeHelper.getOutputDims());
    unsigned int rank = commonUB.size();
    // IndexExprScope IEScope(&rewriter, loc);
    IndexExpr accumulatedOffset = LitIE(0);
    for (unsigned int i = 0; i < inputNum; ++i) {
      // Since the accumulatedOffsetValue will be used in a nested
      // IndexExprScope, we get the Value of this IndexExpr and pass it as a
      // symbol
      Value accumulatedOffsetValue = accumulatedOffset.getValue();
      OpBuilder::InsertionGuard insertGuard(create.getBuilder());
      // Create loop.
      ValueRange loopDef = create.krnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(operands[i], ubs);
      // For each input, only the dimension 'axis' is different
      // Explore parallelism at the first two outermost dimensions and give up
      // if the found dimension is 'axis'.
      commonUB[axis] = ubs[axis];

      // Enable parallelism if required. Do not parallel on the axis dimension.
      if (enableParallel)
        tryCreateKrnlParallel(
            create.krnl, op, "concat", loopDef, lbs, ubs, 0, 2, {axis});

      create.krnl.iterateIE(loopDef, loopDef, lbs, commonUB,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Indices for the read and write.
            SmallVector<Value, 4> readIndices, writeIndices;
            for (unsigned int r = 0; r < rank; ++r) {
              if (r != axis || i == 0)
                writeIndices.emplace_back(loopInd[r]);
              else {
                IndexExprScope IEScope(createKrnl, shapeHelper.getScope());
                IndexExpr writeOffset = DimIE(loopInd[r]);
                IndexExpr accumulatedOffsetIE = SymIE(accumulatedOffsetValue);
                writeOffset = writeOffset + accumulatedOffsetIE;
                writeIndices.emplace_back(writeOffset.getValue());
              }
            }
            // Insert copy.
            Value loadData = createKrnl.load(operands[i], loopInd);
            createKrnl.store(loadData, alloc, writeIndices);
          });
      accumulatedOffset =
          accumulatedOffset + create.krnlIE.getShapeAsDim(operands[i], axis);
    }
  }

  // Loop over outer dimensions before axis, and copy data of a block starting
  // from axis to the innermost dimension from each input to the output.
  void genBlockConcat(MDBuilder &create, ONNXConcatOp concatOp,
      ValueRange operands, ONNXConcatOpShapeHelper &shapeHelper, Value alloc,
      bool enableParallel) const {
    Operation *op = concatOp.getOperation();

    int64_t axis = concatOp.getAxis();
    assert(axis >= 0 && "negative axis is supposed to have been normalized");

    SmallVector<IndexExpr, 4> outputDims(shapeHelper.getOutputDims());
    unsigned int inputNum = operands.size();
    unsigned int rank = outputDims.size();

    // Compute block size to copy for each input.
    SmallVector<Value, 4> blockSizes;
    for (unsigned int i = 0; i < inputNum; ++i) {
      SmallVector<IndexExpr, 4> dims;
      create.krnlIE.getShapeAsDims(operands[i], dims);
      IndexExpr bs = LitIE(1);
      for (unsigned int r = axis; r < rank; ++r) {
        bs = bs * dims[r];
      }
      blockSizes.emplace_back(bs.getValue());
    }

    // Compute block size for the output.
    IndexExpr bsOutIE = LitIE(1);
    for (unsigned int r = axis; r < rank; ++r) {
      bsOutIE = bsOutIE * outputDims[r];
    }
    Value bsOut = bsOutIE.getValue();

    // Creates loops for the outer dimensions before axis.
    SmallVector<IndexExpr, 4> ubs;
    for (unsigned int r = 0; r < axis; ++r) {
      ubs.emplace_back(outputDims[r]);
    }
    SmallVector<IndexExpr, 4> lbs(axis, LitIE(0));
    ValueRange loopDef = create.krnl.defineLoops(axis);

    // Enable parallelism if required. Do not parallel on the axis dimension.
    if (enableParallel)
      tryCreateKrnlParallel(
          create.krnl, op, "concat", loopDef, lbs, ubs, 0, axis, {axis});

    // Loop body.
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MathBuilder createMath(createKrnl);
          // Common factor.
          Value factor = loopInd[0];
          for (unsigned int i = 1; i < loopInd.size(); ++i) {
            factor = createMath.mul(factor, loopInd[i]);
          }

          // Compute write offset.
          Value outOffset = createMath.mul(factor, bsOut);

          // Copy data from each input to the output.
          SmallVector<Value, 4> readIndices, writeIndices;
          for (unsigned int i = 0; i < inputNum; ++i) {
            // Compute read offset.
            Value inOffset = createMath.mul(factor, blockSizes[i]);
            // Copy data.
            Value blockSizeI64 = createMath.cast(
                createKrnl.getBuilder().getI64Type(), blockSizes[i]);
            createKrnl.memcpy(
                alloc, operands[i], blockSizeI64, outOffset, inOffset);
            // Update output offset.
            outOffset = createMath.add(outOffset, blockSizes[i]);
          }
        });
  }
};

void populateLoweringONNXConcatOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXConcatOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
