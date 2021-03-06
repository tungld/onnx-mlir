//===----- normalization.cpp - Lowering Normalization Ops -----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX Normalization Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

struct ONNXBatchNormalizationTestModeOpLowering : public ConversionPattern {
  ONNXBatchNormalizationTestModeOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXBatchNormalizationTestModeOp::getOperationName(), 1,
            ctx) {}
  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter & rewriter) const final {
    // batchnorm{epsilon}(x, scale, bias, mean, variance) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    auto loc = op->getLoc();

    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto epsilonAttr =
        FloatAttr::get(memRefType.getElementType(),
                       llvm::dyn_cast<ONNXBatchNormalizationTestModeOp>(op)
                           .epsilon()
                           .convertToFloat());
    auto epsilon = rewriter.create<ConstantOp>(loc, epsilonAttr);

    auto operand = operands[0];
    auto scale = operands[1];
    auto bias = operands[2];
    auto mean = operands[3];
    auto variance = operands[4];

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    {operand});

    // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
    // In case of N, C is assumed to be 1.
    // Shapes of scale, bias, mean and variance must be C.
    // Computation of BatchNormalization is done as if scale, bias, mean, and
    // variance are reshaped to Cx1x1x...x1.

    // rank
    int64_t rank = memRefType.getRank();

    std::vector<Value> originalLoops;
    std::vector<Value> optimizedLoops;
    Block *optimizationBlock =
        defineLoops(rewriter, loc, originalLoops, optimizedLoops, rank);

    // Create a KrnlIterateOp along C dimension.
    // This will be the outer-most loop in order to re-use scale, bias,
    // mean and variance.

    SmallVector<Value, 1> loopCIVs;
    if (rank > 1) {
      KrnlIterateOperandPack cPack(rewriter, originalLoops[1],
                                   optimizedLoops[1]);
      addDimensionToPack(rewriter, loc, cPack, operand, 1);
      auto cIterateOp = rewriter.create<KrnlIterateOp>(loc, cPack);
      Block &cIterationBlock = cIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&cIterationBlock);
      for (auto arg : cIterationBlock.getArguments())
        loopCIVs.emplace_back(arg);
    } else {
       loopCIVs.emplace_back(rewriter.create<ConstantIndexOp>(loc, 0));
    }

    auto scaleVal = rewriter.create<LoadOp>(loc, scale, loopCIVs);
    auto biasVal = rewriter.create<LoadOp>(loc, bias, loopCIVs);
    auto meanVal = rewriter.create<LoadOp>(loc, mean, loopCIVs);
    auto varianceVal = rewriter.create<LoadOp>(loc, variance, loopCIVs);

    // Create a KrnlIterateOp along the other dimensions.
    SmallVector<int64_t, 4> axes;
    axes.emplace_back(0);
    for (int64_t i = 2; i < rank; ++i)
      axes.emplace_back(i);
    std::vector<Value> packLoops, packOptimizedLoops;
    for (int i = 0; i < axes.size(); ++i) {
      packLoops.emplace_back(originalLoops[axes[i]]);
      packOptimizedLoops.emplace_back(optimizedLoops[axes[i]]);
    }
    KrnlIterateOperandPack pack(rewriter, packLoops, packOptimizedLoops);
    for (int i = 0; i < axes.size(); ++i) {
      addDimensionToPack(rewriter, loc, pack, operand, axes[i]);
    }
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);

    // No optimization
    rewriter.setInsertionPointToEnd(optimizationBlock);
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    Block &iterationBlock = iterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&iterationBlock);

    SmallVector<Value, 4> loopIVs;
    auto args = iterationBlock.getArguments();
    if (args.size() > 1) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
      for (int i = 1; i < args.size(); ++i)
        loopIVs.emplace_back(args[i]);
    } else {
      loopIVs.emplace_back(args[0]);
    }

    auto xVal = rewriter.create<LoadOp>(loc, operand, loopIVs);
    // normalize
    auto dividend = rewriter.create<SubFOp>(loc, xVal, meanVal);
    auto adjustedVarianceVal =
        rewriter.create<AddFOp>(loc, varianceVal, epsilon);
    auto divisor = rewriter.create<SqrtOp>(loc, adjustedVarianceVal);
    auto normVal = rewriter.create<DivFOp>(loc, dividend, divisor);
    // scale and shift
    auto scaleNormVal = rewriter.create<MulFOp>(loc, scaleVal, normVal);
    auto shiftScaleNormVal =
        rewriter.create<AddFOp>(loc, scaleNormVal, biasVal);
    rewriter.create<StoreOp>(loc, shiftScaleNormVal, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXNormalizationOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXBatchNormalizationTestModeOpLowering>(ctx);
}
