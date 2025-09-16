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
#include "src/Support/SmallVectorHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXExpandOpLowering : public OpConversionPattern<ONNXExpandOp> {
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

    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXExpandOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getRank();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Iterate over the output values.
    ValueRange outputLoopDef = create.krnl.defineLoops(outputRank);
    LiteralIndexExpr zeroIE(0);
    SmallVector<IndexExpr, 4> lbs(outputRank, zeroIE);
    DimsExpr ubs = shapeHelper.getOutputDims();

    // Enable parallelism if required.
    int64_t parId = -1;
    if (enableParallel)
      parId = tryCreateKrnlParallel(
          create.krnl, op, "expand", outputLoopDef, lbs, ubs);

    SmallVector<Value, 4> optimizedLoopDef = outputLoopDef;
    // Test for easy unrolling for the innermost store dimension.
    // const int64_t unroll = 8;
    // int64_t uDim = outputRank - 1;
    // if (parId < uDim) {
    //   int64_t tripCount;
    //   int64_t u =
    //       getNoLeftoverUnrollFactor(lbs[uDim], ubs[uDim], unroll, tripCount);
    //   if (u > 1) {
    //     ValueRange blockedLoopDef = create.krnl.block(outputLoopDef[uDim], u);
    //     create.krnl.unroll(blockedLoopDef[1]);
    //     optimizedLoopDef = firstFew<Value, 4>(outputLoopDef, -2);
    //     optimizedLoopDef.emplace_back(blockedLoopDef[0]);
    //     optimizedLoopDef.emplace_back(blockedLoopDef[1]);
    //   }
    // }

    create.krnl.iterateIEWithOrigLoop(outputLoopDef, optimizedLoopDef, lbs, ubs,
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

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  bool enableParallel = false;
};

void populateLoweringONNXExpandOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXExpandOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
