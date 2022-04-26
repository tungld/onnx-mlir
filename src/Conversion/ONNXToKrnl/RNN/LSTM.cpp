/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- LSTM.cpp - Lowering LSTM Op --------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LSTM Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

struct LstmState {
  // returned states.
  Value allH;
  Value ht;
  Value ct;
  // intermediate states.
  Value forwardHt;
  Value reverseHt;
  Value forwardCt;
  Value reverseCt;
};

struct LstmActivationPack {
  RNNActivation f;
  RNNActivation g;
  RNNActivation h;
};

struct LstmWeightPack {
  Value WT;
  Value RT;
};

struct LstmBiasPack {
  bool hasBias = false;
  Value Wb;
  Value Rb;
  // Put peephole here.
  bool hasPeephole = false;
  Value Pi;
  Value Po;
  Value Pf;
};

template <>
bool hasAllNoneOutput<ONNXLSTMOp>(ONNXLSTMOp *op) {
  return (
      isNoneType(op->Y()) && isNoneType(op->Y_h()) && isNoneType(op->Y_c()));
}

template <>
std::tuple<LstmActivationPack, LstmActivationPack>
getActivationPack<ONNXLSTMOp, LstmActivationPack>(ONNXLSTMOp *op) {
  auto direction = op->direction();
  auto activations = op->activations();
  auto activationAlpha = op->activation_alpha();
  auto activationBeta = op->activation_beta();

  LstmActivationPack activationForward, activationReverse;

  // Get activation function name.
  // Default forward functions
  activationForward.f.name = "sigmoid";
  activationForward.g.name = "tanh";
  activationForward.h.name = "tanh";
  // Default backward functions
  activationReverse.f.name = "sigmoid";
  activationReverse.g.name = "tanh";
  activationReverse.h.name = "tanh";
  if (activations) {
    ArrayAttr activationArrAttr = activations.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.name =
            activationArrAttr[0].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.name =
            activationArrAttr[1].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > 2) {
        activationForward.h.name =
            activationArrAttr[2].cast<StringAttr>().getValue();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.name =
            activationArrAttr[startIndex].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.name =
            activationArrAttr[startIndex + 1].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.name =
            activationArrAttr[startIndex + 2].cast<StringAttr>().getValue();
      }
    }
  }

  // Get alpha attributes.
  if (activationAlpha) {
    ArrayAttr activationArrAttr = activationAlpha.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.alpha = activationArrAttr[0].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.alpha = activationArrAttr[1].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > 2) {
        activationForward.h.alpha = activationArrAttr[2].cast<FloatAttr>();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.alpha =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.alpha =
            activationArrAttr[startIndex + 1].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.alpha =
            activationArrAttr[startIndex + 2].cast<FloatAttr>();
      }
    }
  }

  // Get beta attributes.
  if (activationBeta) {
    ArrayAttr activationArrAttr = activationBeta.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.beta = activationArrAttr[0].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.beta = activationArrAttr[1].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > 2) {
        activationForward.h.beta = activationArrAttr[2].cast<FloatAttr>();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.beta =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.beta =
            activationArrAttr[startIndex + 1].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.beta =
            activationArrAttr[startIndex + 2].cast<FloatAttr>();
      }
    }
  }

  return std::make_tuple(activationForward, activationReverse);
}

template <>
std::tuple<LstmWeightPack, LstmWeightPack>
getWeightPack<ONNXLSTMOp, LstmWeightPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op) {
  // Return values.
  LstmWeightPack weightForward, weightReverse;

  // parameter weight: [direction, 4*hiddenSize, inputSize]
  Value W = op->W();
  // recurrence weight: [direction, 4*hiddenSize, hiddenSize]
  Value R = op->R();
  // direction
  StringRef direction = op->direction();

  ArrayRef<int64_t> wShape = W.getType().cast<ShapedType>().getShape();
  Type elementType = W.getType().cast<ShapedType>().getElementType();
  int64_t hiddenSize = wShape[1] / 4;
  int64_t inputSize = wShape[2];

  // MemRef types for parameter weights.
  auto w3DTy = MemRefType::get({1, 4 * hiddenSize, inputSize}, elementType);
  auto w2DTy = MemRefType::get({4 * hiddenSize, inputSize}, elementType);
  auto wTranspose2DTy =
      MemRefType::get({inputSize, 4 * hiddenSize}, elementType);
  SmallVector<Type, 4> w3D2Ty(2, w3DTy);

  // MemRef types for recurrence weights.
  auto r3DTy = MemRefType::get({1, 4 * hiddenSize, hiddenSize}, elementType);
  auto r2DTy = MemRefType::get({4 * hiddenSize, hiddenSize}, elementType);
  auto rTranspose2DTy =
      MemRefType::get({hiddenSize, 4 * hiddenSize}, elementType);
  SmallVector<Type, 4> r3D2Ty(2, r3DTy);

  // Squeeze the direction axis from W and R.
  Value fW, bW, fR, bR;
  if (direction == FORWARD) {
    fW = foldOrEmitONNXSqueezeV11Op(rewriter, loc, w2DTy, W, /*axis=*/0);
    fR = foldOrEmitONNXSqueezeV11Op(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else if (direction == REVERSE) {
    bW = foldOrEmitONNXSqueezeV11Op(rewriter, loc, w2DTy, W, /*axis=*/0);
    bR = foldOrEmitONNXSqueezeV11Op(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else { // BIDIRECTIONAL
    // W
    std::vector<Value> vals =
        foldOrEmitONNXSplitOp(rewriter, loc, w3D2Ty, W, 0);
    fW = foldOrEmitONNXSqueezeV11Op(rewriter, loc, w2DTy, vals[0], /*axis=*/0);
    bW = foldOrEmitONNXSqueezeV11Op(rewriter, loc, w2DTy, vals[1], /*axis=*/0);
    // R
    vals.clear();
    vals = foldOrEmitONNXSplitOp(rewriter, loc, r3D2Ty, R, 0);
    fR = foldOrEmitONNXSqueezeV11Op(rewriter, loc, r2DTy, vals[0], /*axis=*/0);
    bR = foldOrEmitONNXSqueezeV11Op(rewriter, loc, r2DTy, vals[1], /*axis=*/0);
  }

  // Transpose W and R.
  ArrayAttr permAttr = rewriter.getI64ArrayAttr({1, 0});
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    // W
    weightForward.WT =
        foldOrEmitONNXTransposeOp(rewriter, loc, wTranspose2DTy, fW, permAttr);
    // R
    weightForward.RT =
        foldOrEmitONNXTransposeOp(rewriter, loc, rTranspose2DTy, fR, permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    // W
    weightReverse.WT =
        foldOrEmitONNXTransposeOp(rewriter, loc, wTranspose2DTy, bW, permAttr);
    // R
    weightReverse.RT =
        foldOrEmitONNXTransposeOp(rewriter, loc, rTranspose2DTy, bR, permAttr);
  }
  return std::make_tuple(weightForward, weightReverse);
}

template <>
std::tuple<LstmBiasPack, LstmBiasPack> getBiasPack<ONNXLSTMOp, LstmBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op) {
  // Return values.
  LstmBiasPack biasForward, biasReverse;

  // bias: [direction, 8*hiddenSize] for both parameter and recurrence weights.
  Value B = op->B();
  // peephold: [direction, 3*hiddenSize] for input, output and forget gates.
  Value P = op->P();

  // direction
  StringRef direction = op->direction();

  // Split B.
  if (!isNoneType(B)) {
    ArrayRef<int64_t> bShape = B.getType().cast<ShapedType>().getShape();
    Type elementType = B.getType().cast<ShapedType>().getElementType();
    int64_t hiddenSize = bShape[1] / 8;

    // MemRef types.
    auto bType2D = MemRefType::get({1, 8 * hiddenSize}, elementType);
    auto bType1D = MemRefType::get({8 * hiddenSize}, elementType);
    auto bSplitType1D = MemRefType::get({4 * hiddenSize}, elementType);
    SmallVector<Type, 2> split1D2Ty(2, bSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, bType2D);

    // Squeeze the direction axis from B.
    Value fB, bB;
    if (direction == FORWARD) {
      fB = foldOrEmitONNXSqueezeV11Op(rewriter, loc, bType1D, B, /*axis=*/0);
    } else if (direction == REVERSE) {
      bB = foldOrEmitONNXSqueezeV11Op(rewriter, loc, bType1D, B, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals;
      vals = foldOrEmitONNXSplitOp(rewriter, loc, split2D2Ty, B, 0);
      fB = foldOrEmitONNXSqueezeV11Op(
          rewriter, loc, bType1D, vals[0], /*axis=*/0);
      bB = foldOrEmitONNXSqueezeV11Op(
          rewriter, loc, bType1D, vals[1], /*axis=*/0);
    }

    // Split B into individual bias tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOp(rewriter, loc, split1D2Ty, fB, 0);
      biasForward.Wb = vals[0];
      biasForward.Rb = vals[1];
      biasForward.hasBias = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOp(rewriter, loc, split1D2Ty, bB, 0);
      biasReverse.Wb = vals[0];
      biasReverse.Rb = vals[1];
      biasReverse.hasBias = true;
    }
  }

  // Split P.
  if (!isNoneType(P)) {
    ArrayRef<int64_t> pShape = P.getType().cast<ShapedType>().getShape();
    Type elementType = P.getType().cast<ShapedType>().getElementType();
    int64_t hiddenSize = pShape[1] / 3;

    // MemRef types.
    auto pType2D = MemRefType::get({1, 3 * hiddenSize}, elementType);
    auto pType1D = MemRefType::get({3 * hiddenSize}, elementType);
    auto pSplitType1D = MemRefType::get({hiddenSize}, elementType);
    SmallVector<Type, 4> split1D3Ty(3, pSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, pType2D);

    // Squeeze the direction axis from P.
    Value fP, bP;
    if (direction == FORWARD) {
      fP = foldOrEmitONNXSqueezeV11Op(rewriter, loc, pType1D, P, /*axis=*/0);
    } else if (direction == REVERSE) {
      bP = foldOrEmitONNXSqueezeV11Op(rewriter, loc, pType1D, P, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals =
          foldOrEmitONNXSplitOp(rewriter, loc, split2D2Ty, P, 0);
      fP = foldOrEmitONNXSqueezeV11Op(
          rewriter, loc, pType1D, vals[0], /*axis=*/0);
      bP = foldOrEmitONNXSqueezeV11Op(
          rewriter, loc, pType1D, vals[1], /*axis=*/0);
    }

    // Split P into individual tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOp(rewriter, loc, split1D3Ty, fP, 0);
      biasForward.Pi = vals[0];
      biasForward.Po = vals[1];
      biasForward.Pf = vals[2];
      biasForward.hasPeephole = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOp(rewriter, loc, split1D3Ty, bP, 0);
      biasReverse.Pi = vals[0];
      biasReverse.Po = vals[1];
      biasReverse.Pf = vals[2];
      biasReverse.hasPeephole = true;
    }
  }

  return std::make_tuple(biasForward, biasReverse);
}

template <>
LstmState allocAndInitializeStates<ONNXLSTMOp, LstmState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op,
    typename ONNXLSTMOp::Adaptor operandAdaptor) {
  LstmState state;

  // direction
  StringRef direction = op->direction();

  // Insert allocation and deallocation for the results of this operation.
  // If the result is not returned, then no allocation happens.
  // Y :: [seq_length, num_directions, batch_size, hidden_size]
  state.allH = allocAllHidden(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y(),
      checkInsertDealloc(op->getOperation(), 0));
  // Y_h :: [num_directions, batch_size, hidden_size]
  state.ht = allocHiddenOrCell(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y_h(),
      checkInsertDealloc(op->getOperation(), 1));
  // Y_c :: [num_directions, batch_size, hidden_size]
  state.ct = allocHiddenOrCell(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y_c(),
      checkInsertDealloc(op->getOperation(), 2));

  // Insert allocation and deallocation the intermediate Ht and Ct for the
  // forward and reverse directions.
  // Ht :: [batch_size, hidden_size]
  // Ct :: [batch_size, hidden_size]
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    state.forwardHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
    state.forwardCt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    state.reverseHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
    state.reverseCt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
  }

  // Initialize Ht and Ct.
  initializeIntermediateStates(rewriter, loc, state.forwardHt, state.reverseHt,
      state.forwardCt, state.reverseCt, operandAdaptor.initial_h(),
      operandAdaptor.initial_c(),
      operandAdaptor.X().getType().cast<MemRefType>().getElementType(),
      direction, /*onlyHidden=*/false);
  return state;
}

template <>
void calculateState<LstmState, LstmActivationPack, LstmWeightPack,
    LstmBiasPack>(ConversionPatternRewriter &rewriter, Location loc, Value Xt,
    LstmState state, LstmActivationPack activationPack,
    LstmWeightPack weightPack, LstmBiasPack biasPack, Value sequenceIV,
    Value directionIV, bool isForward) {
  // Equations for LSTM.
  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // Ct = ft (.) Ct-1 + it (.) ct
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // Ht = ot (.) h(Ct)

  // TODO remove scope
  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder, OnnxBuilder>
      create(rewriter, loc);

  ArrayRef<int64_t> xtShape = Xt.getType().cast<ShapedType>().getShape();
  int64_t batchSize = xtShape[0];

  // Get Ht, Ct.
  Value Ht = (isForward) ? state.forwardHt : state.reverseHt;
  Value Ct = (isForward) ? state.forwardCt : state.reverseCt;

  ArrayRef<int64_t> htShape = Ht.getType().cast<ShapedType>().getShape();
  int64_t hiddenSize = htShape[1];

  // Frequently used types.
  MemRefType matrixType = Ht.getType().cast<MemRefType>();
  Type elementType = matrixType.getElementType();
  MemRefType matrixAllGatesType =
      MemRefType::get({batchSize, 4 * hiddenSize}, elementType);

  // Do matrix multiplications.
  // Xt * (Wi^T ++ Wo^T ++ Wf^T ++ Wc^T) + (Wbi ++ Wbo ++ Wbf ++ Wbc)
  // Ht * (Ri^T ++ Ro^T ++ Rf^T ++ Rc^T) + (Rbi ++ Rbo ++ Rbf ++ Rbc)
  // where '++' is matrix concatenation.
  Value XtWTWb =
      create.onnx.gemm(matrixAllGatesType, Xt, weightPack.WT, biasPack.Wb);
  Value HtRTRb =
      create.onnx.gemm(matrixAllGatesType, Ht, weightPack.RT, biasPack.Rb);

  // Do element-wise computations. Fuse them into a single nested loop.
  // Lower and upper bounds derived from Ht tensor.
  unsigned HtRank = matrixType.getRank();
  Value iZero = create.math.constantIndex(0);
  SmallVector<Value, 4> HtLbs(HtRank, iZero);
  SmallVector<Value, 4> HtUbs;
  for (unsigned r = 0; r < HtRank; ++r) {
    HtUbs.emplace_back(create.mem.dim(Ht, r));
  }

  ValueRange loops = create.krnl.defineLoops(HtRank);
  create.krnl.iterate(loops, loops, HtLbs, HtUbs,
      [&](KrnlBuilder &createKrnl, ValueRange indices) {
        MathBuilder createMath(createKrnl);
        IndexExprScope ieScope(createKrnl);
        Value bs(indices[0]), hs(indices[1]);
        SymbolIndexExpr bsie(bs), hsie(hs);
        LiteralIndexExpr hsieLit(hiddenSize);

        Value CtVal = createKrnl.load(Ct, indices);
        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        Value XtWTWbiVal = createKrnl.loadIE(XtWTWb, {bsie, hsie});
        Value HtRTRbiVal = createKrnl.loadIE(HtRTRb, {bsie, hsie});
        Value it = createMath.add(XtWTWbiVal, HtRTRbiVal);
        if (biasPack.hasPeephole) {
          Value PiVal = createKrnl.load(biasPack.Pi, {hs});
          Value PiCt = createMath.mul(PiVal, CtVal);
          it = createMath.add(it, PiCt);
        }
        it =
            applyActivation(createKrnl.getBuilder(), loc, activationPack.f, it);

        // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        Value XtWTWbfVal =
            createKrnl.loadIE(XtWTWb, {bsie, hsie + 2 * hsieLit});
        Value HtRTRbfVal =
            createKrnl.loadIE(HtRTRb, {bsie, hsie + 2 * hsieLit});
        Value ft = createMath.add(XtWTWbfVal, HtRTRbfVal);
        if (biasPack.hasPeephole) {
          Value PfVal = createKrnl.load(biasPack.Pf, {hs});
          Value PfCt = createMath.mul(PfVal, CtVal);
          ft = createMath.add(ft, PfCt);
        }
        ft =
            applyActivation(createKrnl.getBuilder(), loc, activationPack.f, ft);

        // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        Value XtWTWbcVal =
            createKrnl.loadIE(XtWTWb, {bsie, hsie + 3 * hsieLit});
        Value HtRTRbcVal =
            createKrnl.loadIE(HtRTRb, {bsie, hsie + 3 * hsieLit});
        Value ct = createMath.add(XtWTWbcVal, HtRTRbcVal);
        ct =
            applyActivation(createKrnl.getBuilder(), loc, activationPack.g, ct);

        // Ct = ft (.) Ct-1 + it (.) ct
        Value ftCt = createMath.mul(ft, CtVal);
        Value itct = createMath.mul(it, ct);
        Value nextCt = createMath.add(ftCt, itct);

        // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        Value XtWTWboVal = createKrnl.loadIE(XtWTWb, {bsie, hsie + hsieLit});
        Value HtRTRboVal = createKrnl.loadIE(HtRTRb, {bsie, hsie + hsieLit});
        Value ot = createMath.add(XtWTWboVal, HtRTRboVal);
        if (biasPack.hasPeephole) {
          Value PoVal = createKrnl.load(biasPack.Po, {hs});
          Value PoCt = createMath.mul(PoVal, nextCt);
          ot = createMath.add(ot, PoCt);
        }
        ot =
            applyActivation(createKrnl.getBuilder(), loc, activationPack.f, ot);

        // Ht = ot (.) h(Ct)
        Value nextHt = applyActivation(
            createKrnl.getBuilder(), loc, activationPack.h, nextCt);
        nextHt = createMath.mul(ot, nextHt);

        // Store the intermediate Ht, Ct.
        createKrnl.store(nextCt, Ct, indices);
        createKrnl.store(nextHt, Ht, indices);
        if (!isNoneType(state.allH))
          createKrnl.store(
              nextHt, state.allH, {sequenceIV, directionIV, bs, hs});
      });
}

template <>
void stateToOutput<ONNXLSTMOp, LstmState>(ConversionPatternRewriter &rewriter,
    Location loc, ONNXLSTMOp *op, LstmState state,
    std::vector<Value> &outputs) {
  Value noneValue;
  auto direction = op->direction();

  // First output: all sequences.
  outputs.emplace_back((isNoneType(op->Y()) ? noneValue : state.allH));
  // Second output: hidden.
  if (isNoneType(op->Y_h()))
    outputs.emplace_back(noneValue);
  else {
    stateToOutputForHiddenOrCell(
        rewriter, loc, state.forwardHt, state.reverseHt, direction, state.ht);
    outputs.emplace_back(state.ht);
  }
  // Third output: cell.
  if (isNoneType(op->Y_c()))
    outputs.emplace_back(noneValue);
  else {
    stateToOutputForHiddenOrCell(
        rewriter, loc, state.forwardCt, state.reverseCt, direction, state.ct);
    outputs.emplace_back(state.ct);
  }
}

void populateLoweringONNXLSTMOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp, LstmState, LstmActivationPack,
      LstmWeightPack, LstmBiasPack>>(typeConverter, ctx);
}

} // namespace onnx_mlir
