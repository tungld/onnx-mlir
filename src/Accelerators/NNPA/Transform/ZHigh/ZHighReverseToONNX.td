// SPDX-License-Identifier: Apache-2.0

//===------- ZHighReverseToONNX.td - Pattern Match for ZHighDialect -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ZHIGH_REVERSE_TO_ONNX
#define ZHIGH_REVERSE_TO_ONNX

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Layout propagation for binary element-wise operations
//
// Binary operations, such as Add and Div, do not depend on a specific layout.
// Thus, it is possible to use the layout of the previous zTensor. In other
// words, propagate the existing layout down to the unary operation, so that
// other rules can be applied to remove unstick-stick pairs.For example,
// This sequence of operations:
//   Unstick (NHWC) -> Stick (4D) -> zhigh.Add
//                                       ^
//   Unstick (NHWC) -> Stick (4D) -------|
//              
// will become:
//   Unstick -> zhigh.Add -> Unstick (NHWC) -> Stick (4D)
//                ^
//   Unstick -----|
//
// then, canonicalization of unstick/stick will remove the unstick-stick pairs.
//
//===----------------------------------------------------------------------===//

// Add
def AddLayoutPropagatePattern : Pat<
  (ZHighUnstickOp (ZHighAddOp (ZHighStickOp $x, $_), (ZHighStickOp $y, $_))),
  (ONNXAddOp $x, $y),
  []
>;

// Mul
def MulLayoutPropagatePattern : Pat<
  (ZHighUnstickOp (ZHighMulOp (ZHighStickOp $x, $_), (ZHighStickOp $y, $_))),
  (ONNXMulOp $x, $y),
  []
>;

// Sub
def SubLayoutPropagatePattern : Pat<
  (ZHighUnstickOp (ZHighSubOp (ZHighStickOp $x, $_), (ZHighStickOp $y, $_))),
  (ONNXSubOp $x, $y),
  []
>;

#endif // ZHIGH_REVERSE_TO_ONNX


