/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZHighClipToDLFloat.cpp - ZHigh High Level Optimizer -------===//
//
// Copyright 2023- The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewritten rules to clip CPU numerical values
// before passing to ZHighStick, which avoids data range violation error due to
// the dlfloat range.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace onnx_mlir {
namespace zhigh {

namespace {

/// Check if a value is from or transitively from a zTensor without value
/// modification.
bool valueFromZTensor(Value tensor) {
  if (!tensor.dyn_cast<BlockArgument>()) {
    Operation *op = tensor.getDefiningOp();
    // From a zTensor.
    if (isa<ZHighUnstickOp>(op))
      return true;
    // Concat/Transpose does not change the input precision. So we can consider
    // that the input is already in the dlfloat range if it comes from zTensor.
    if (isa<ONNXConcatOp, ONNXTransposeOp>(op)) {
      return llvm::all_of(
          op->getOperands(), [&](Value v) { return valueFromZTensor(v); });
    }
  }
  return false;
}

class ZHighClipToDLFloatPattern : public OpRewritePattern<ZHighStickOp> {
public:
  using OpRewritePattern<ZHighStickOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZHighStickOp stickOp, PatternRewriter &rewriter) const override {
    Operation *genericOp = stickOp.getOperation();
    Location loc = genericOp->getLoc();

    Value input = stickOp.getIn();
    Type inputElementType = getElementType(input.getType());

    // Do not clip if the input tensor is already in the dlfloat range.
    // For example, the input was unstickified from a zTensor.
    if (valueFromZTensor(input))
      return failure();

    // Clip the input values if required since the values are potentially
    // out-of-bound of dlfloat.
    // dlfloat value =  (-1)^s * 2^(e-31) * (1 + m/512), e=[0, 63], m=[0, 511],
    // according to the paper: "DLFloat: A 16-b Floating Point Format Designed
    // for Deep Learning Training and Inference", Ankur Agrawal, et al.,
    // (e=63, m=511) is preserved for NaN and Infinity, so use (e=63p, m=510)
    // as the minimum value for clipping.
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    float dlfMin = -1 * (1L << 32) * (1.0 + (510.0 / 512.0));
    DenseElementsAttr minAttr = DenseElementsAttr::get<float>(
        RankedTensorType::get({1}, inputElementType), dlfMin);
    Value minVal = create.onnx.constant(minAttr);
    Value clippedVal = create.onnx.max({input, minVal});
    Value replacedVal = rewriter.create<ZHighStickOp>(
        loc, stickOp.getOut().getType(), clippedVal, stickOp.getLayoutAttr());

    rewriter.replaceOp(genericOp, replacedVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ZHighClipToDLFloatPass
//===----------------------------------------------------------------------===//

struct ZHighClipToDLFloatPass
    : public PassWrapper<ZHighClipToDLFloatPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighClipToDLFloatPass)

  StringRef getArgument() const override { return "zhigh-clip-to-dlfloat"; }

  StringRef getDescription() const override {
    return "Clip stickification inputs at ZHighIR.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<ZHighClipToDLFloatPattern>(&getContext());

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    /// Only pre-existing ops (that were were on the worklist at the very
    /// beginning) enqueued. All other ops are excluded.
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsAndFoldGreedily(
            function, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // anonymous namespace

std::unique_ptr<Pass> createZHighClipToDLFloatPass() {
  return std::make_unique<ZHighClipToDLFloatPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
