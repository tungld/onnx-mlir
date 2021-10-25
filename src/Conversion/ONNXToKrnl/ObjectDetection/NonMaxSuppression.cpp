/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- NonMaxSuppression.cpp - Lowering NonMaxSuppression Op ----------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX NonMaxSuppression Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

using AffineBuilderKrnlMem = GenericAffineBuilder<KrnlLoadOp, KrnlStoreOp>;

/// Compute the intersection-over-union (IOU) score between two boxes.
/// IOU tells us how much two boxes are overlapped.
static Value emitIOU(MathBuilder &createMath, SmallVectorImpl<Value> &box1,
    SmallVectorImpl<Value> &box2, int64_t centerPointBox = 0) {
  Value area1, area2;
  Value y1_min, x1_min, y1_max, x1_max;
  Value y2_min, x2_min, y2_max, x2_max;
  if (centerPointBox == 0) {
    // The box data is supplied as [y1, x1, y2, x2]. (y1, x1) and (y2, x2)
    // are the coordinates of the diagonal pair of bottom-left and top-right
    // corners.
    y1_min = box1[0];
    x1_min = box1[1];
    y1_max = box1[2];
    x1_max = box1[3];

    y2_min = box2[0];
    x2_min = box2[1];
    y2_max = box2[2];
    x2_max = box2[3];

    area1 = createMath.mul(
        createMath.sub(y1_max, y1_min), createMath.sub(x1_max, x1_min));
    area2 = createMath.mul(
        createMath.sub(y2_max, y2_min), createMath.sub(x2_max, x2_min));
  } else {
    // The box data is supplied as [x_center, y_center, width, height].
    Value x1_center = box1[0];
    Value y1_center = box1[1];
    Value w1 = box1[2];
    Value h1 = box1[3];

    Value x2_center = box2[0];
    Value y2_center = box2[1];
    Value w2 = box2[2];
    Value h2 = box2[3];

    Value two = createMath.constant(w1.getType(), 2);
    x1_min = createMath.sub(x1_center, createMath.div(w1, two));
    x1_max = createMath.add(x1_center, createMath.div(w1, two));
    y1_min = createMath.sub(y1_center, createMath.div(h1, two));
    y1_max = createMath.add(y1_center, createMath.div(h1, two));

    y2_min = createMath.sub(y2_center, createMath.div(h2, two));
    y2_max = createMath.add(y2_center, createMath.div(h2, two));
    x2_min = createMath.sub(x2_center, createMath.div(w2, two));
    x2_max = createMath.add(x2_center, createMath.div(w2, two));

    area1 = createMath.mul(h1, w1);
    area2 = createMath.mul(h2, w2);
  }

  Value intersection_x_min = createMath.max(x1_min, x2_min);
  Value intersection_y_min = createMath.max(y1_min, y2_min);
  Value intersection_x_max = createMath.min(x1_max, x2_max);
  Value intersection_y_max = createMath.min(y1_max, y2_max);

  Value zero = createMath.constant(intersection_x_min.getType(), 0);
  Value intersection_w = createMath.sub(intersection_x_max, intersection_x_min);
  Value intersection_h = createMath.sub(intersection_y_max, intersection_y_min);
  Value intersection_area = createMath.mul(createMath.max(intersection_w, zero),
      createMath.max(intersection_h, zero));

  Value union_area = createMath.add(area1, area2);
  union_area = createMath.sub(union_area, intersection_area);
  // Avoid zero division.
  Value epsilon = createMath.constant(zero.getType(), 1e-8);
  union_area = createMath.add(union_area, epsilon);
  return createMath.div(intersection_area, union_area);
}

/// Suppress the number of output bounding boxes per class by scores.
static void suppressByScores(ConversionPatternRewriter &rewriter, Location loc,
    Value scores, Value scoreThreshold, Value maxOutputPerClass) {

  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);
  MemRefBuilder createMemref(createKrnl);
  IndexExprScope scope(createKrnl);
  Type indexType = rewriter.getIndexType();

  MemRefBoundsIndexCapture scoreBounds(scores);
  IndexExpr bsIE = scoreBounds.getDim(0); // batch size.
  IndexExpr csIE = scoreBounds.getDim(1); // class size.
  IndexExpr ssIE = scoreBounds.getDim(2); // spatial size.
  LiteralIndexExpr zeroIE(0), oneIE(1);

  // Compute the effective max output per class.
  Value effectiveMaxPerClass =
      createMemref.alloca(MemRefType::get({}, indexType));
  createKrnl.store(zeroIE.getValue(), effectiveMaxPerClass, {});

  ValueRange bcLoopDef = createKrnl.defineLoops(2);
  createKrnl.iterateIE(bcLoopDef, bcLoopDef, {zeroIE, zeroIE}, {bsIE, csIE},
      [&](KrnlBuilder &createKrnl, ValueRange bcLoopInd) {
        MathBuilder createMath(createKrnl);
        MemRefBuilder createMemref(createKrnl);
        IndexExprScope bcScope(createKrnl);
        Value b(bcLoopInd[0]), c(bcLoopInd[1]);

        // Store the number of scores whose value is greater than the
        // threshold. Counting is done per class.
        Value topk = createMemref.alloca(MemRefType::get({}, indexType));
        createKrnl.store(zeroIE.getValue(), topk, {});

        // Load the score threshold.
        Value threshold = createKrnl.loadIE(scoreThreshold, {zeroIE});

        // Count the number of scores whose value is greater than the
        // threshold. Counting is done per class.
        ValueRange sLoopDef = createKrnl.defineLoops(1);
        createKrnl.iterateIE(sLoopDef, sLoopDef, {zeroIE}, {ssIE},
            [&](KrnlBuilder &createKrnl, ValueRange sLoopInd) {
              Value s(sLoopInd[0]);
              MathBuilder createMath(createKrnl);
              IndexExprScope sScope(createKrnl);

              Value score = createKrnl.load(scores, {b, c, s});
              // Increase the counter if score > threshold.
              Value gt = createMath.sgt(score, threshold);
              Value topkVal = createKrnl.load(topk, {});
              Value topkPlusOneVal = createMath.add(topkVal, oneIE.getValue());
              topkVal = createMath.select(gt, topkPlusOneVal, topkVal);
              createKrnl.store(topkVal, topk, {});
            });

        // Update the effective max output per class.
        Value x = createKrnl.load(topk, {});
        Value y = createKrnl.load(effectiveMaxPerClass, {});
        createKrnl.store(createMath.max(x, y), effectiveMaxPerClass, {});
      });

  // Suppress the number of output bounding boxes per class.
  Value x = createKrnl.load(maxOutputPerClass, {});
  Value y = createKrnl.load(effectiveMaxPerClass, {});
  createKrnl.store(createMath.min(x, y), maxOutputPerClass, {});
}

/// Returns the indices that would sort the score tensor.
/// Scores :: [num_of_batch, num_of_class, spatial_dimension]
/// Sort along `spatial_dimension` axis.
static Value emitArgSort(
    ConversionPatternRewriter &rewriter, Location loc, Value scores) {
  KrnlBuilder createKrnl(rewriter, loc);
  IndexExprScope scope(createKrnl);

  MemRefType scoreMemRefType = scores.getType().cast<MemRefType>();
  Type indexType = rewriter.getIndexType();

  MemRefBoundsIndexCapture scoreBounds(scores);
  SmallVector<IndexExpr, 4> dimsSize;
  scoreBounds.getDimList(dimsSize);
  IndexExpr bsIE = dimsSize[0]; // batch size.
  IndexExpr csIE = dimsSize[1]; // class size.
  IndexExpr ssIE = dimsSize[2]; // spatial size.
  LiteralIndexExpr zeroIE(0), oneIE(1);

  // Create and initialize the result.
  Value order = insertAllocAndDeallocSimple(rewriter, nullptr,
      MemRefType::get(scoreMemRefType.getShape(), indexType), loc, dimsSize,
      /*insertDealloc=*/true);
  ValueRange initLoopDef = createKrnl.defineLoops(3);
  createKrnl.iterateIE(initLoopDef, initLoopDef, {zeroIE, zeroIE, zeroIE},
      {bsIE, csIE, ssIE}, [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
        // order[b, c, s] = s
        createKrnl.store(loopInd[2], order, loopInd);
      });

  // Do sorting in the descending order of scores and return their indices.
  // Using bubble sort.
  ValueRange loopDef = createKrnl.defineLoops(3);
  createKrnl.iterateIE(loopDef, loopDef, {zeroIE, zeroIE, zeroIE},
      {bsIE, csIE, ssIE - oneIE},
      [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
        Value b(loopInd[0]), c(loopInd[1]), i(loopInd[2]);
        IndexExpr i1 = DimIndexExpr(i) + LiteralIndexExpr(1);

        ValueRange swapLoopDef = createKrnl.defineLoops(1);
        createKrnl.iterateIE(swapLoopDef, swapLoopDef, {i1}, {ssIE},
            [&](KrnlBuilder &createKrnl, ValueRange swapLoopInd) {
              MathBuilder createMath(createKrnl);
              Value k(swapLoopInd[0]);
              Value xOrd = createKrnl.load(order, {b, c, i});
              Value yOrd = createKrnl.load(order, {b, c, k});
              Value x = createKrnl.load(scores, {b, c, xOrd});
              Value y = createKrnl.load(scores, {b, c, yOrd});
              Value lt = createMath.slt(x, y);
              auto ifOp =
                  rewriter.create<scf::IfOp>(loc, lt, /*withElseRegion=*/false);
              rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
              createKrnl.store(yOrd, order, {b, c, i});
              createKrnl.store(xOrd, order, {b, c, k});
            });
      });

  return order;
}

/// Bounding boxes may contain a mix of flipped and non-flipped boxes. Try to
/// flip the flipped boxes back.
/// BoundingBoxes: [num_of_batch, spatial_dimension, 4]
static Value tryToUnflip(
    ConversionPatternRewriter &rewriter, Location loc, Value boundingBoxes) {
  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);

  MemRefBoundsIndexCapture bbBounds(boundingBoxes);
  IndexExpr bs = bbBounds.getDim(0); // batch size.
  IndexExpr ss = bbBounds.getDim(1); // spatial size.
  SmallVector<IndexExpr, 4> ubs;
  bbBounds.getDimList(ubs);
  LiteralIndexExpr zero(0), one(1), two(2), three(3);

  Value resMemRef = insertAllocAndDeallocSimple(rewriter, nullptr,
      boundingBoxes.getType().cast<MemRefType>(), loc, ubs,
      /*insertDealloc=*/false);

  ValueRange loopDef = createKrnl.defineLoops(2);
  createKrnl.iterateIE(loopDef, loopDef, {zero, zero}, {bs, ss},
      [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
        MathBuilder createMath(createKrnl);
        DimIndexExpr b(loopInd[0]), s(loopInd[1]);
        // Load a bounding box.
        Value y_min = createKrnl.loadIE(boundingBoxes, {b, s, zero});
        Value x_min = createKrnl.loadIE(boundingBoxes, {b, s, one});
        Value y_max = createKrnl.loadIE(boundingBoxes, {b, s, two});
        Value x_max = createKrnl.loadIE(boundingBoxes, {b, s, three});

        // Flip x.
        Value gtX = createMath.sgt(x_min, x_max);
        Value newXMin = createMath.select(gtX, x_max, x_min);
        Value newXMax = createMath.select(gtX, x_min, x_max);

        // Flip y.
        Value gtY = createMath.sgt(y_min, y_max);
        Value newYMin = createMath.select(gtY, y_max, y_min);
        Value newYMax = createMath.select(gtY, y_min, y_max);

        // Update the bounding box.
        createKrnl.storeIE(newYMin, resMemRef, {b, s, zero});
        createKrnl.storeIE(newXMin, resMemRef, {b, s, one});
        createKrnl.storeIE(newYMax, resMemRef, {b, s, two});
        createKrnl.storeIE(newXMax, resMemRef, {b, s, three});
      });
  return resMemRef;
}

struct ONNXNonMaxSuppressionOpLowering : public ConversionPattern {
  ONNXNonMaxSuppressionOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXNonMaxSuppressionOp::getOperationName(), 1, ctx) {
  }

  /// To understand how code is generated for NonMaxSuppression, look at the
  /// python implementation at the end of this file.
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXNonMaxSuppressionOp nmsOp = llvm::cast<ONNXNonMaxSuppressionOp>(op);
    ONNXNonMaxSuppressionOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Builder helper.
    IndexExprScope mainScope(&rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);
    MathBuilder createMath(createKrnl);
    MemRefBuilder createMemref(createKrnl);

    // Common information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = memRefType.getElementType();
    Type indexType = rewriter.getIndexType();
    Type boolType = rewriter.getI1Type();
    Type i64Type = rewriter.getI64Type();

    // Operation's operands.
    // Bounding boxes.
    Value boxes = operandAdaptor.boxes();
    // Scores.
    Value scores = operandAdaptor.scores();
    // Maximun number of output boxes per class.
    Value maxOutputBoxPerClass = operandAdaptor.max_output_boxes_per_class();
    if (maxOutputBoxPerClass.getType().isa<NoneType>()) {
      maxOutputBoxPerClass = createMemref.alloca(MemRefType::get({1}, i64Type));
      Value zero = createMath.constant(i64Type, 0);
      createKrnl.store(zero, maxOutputBoxPerClass, {});
    }
    // Score threshold.
    Type scoreType = scores.getType().cast<MemRefType>().getElementType();
    Value scoreThreshold = operandAdaptor.score_threshold();
    if (scoreThreshold.getType().isa<NoneType>()) {
      scoreThreshold = createMemref.alloca(MemRefType::get({1}, scoreType));
      Value zero = createMath.constant(scoreType, 0);
      createKrnl.store(zero, scoreThreshold, {});
    }
    // IOU threshold.
    Value iouThreshold = operandAdaptor.iou_threshold();
    if (iouThreshold.getType().isa<NoneType>()) {
      iouThreshold = createMemref.alloca(MemRefType::get({1}, scoreType));
      Value zero = createMath.constant(scoreType, 0);
      createKrnl.store(zero, iouThreshold, {});
    }
    // Mode: diagonal corners or center point.
    int64_t centerPointBox = nmsOp.center_point_box();

    // boxes: [num_of_batch, spatial_dimension, 4]
    // scores: [num_of_batch, num_of_class, spatial_dimension]
    MemRefBoundsIndexCapture boxBounds(boxes);
    MemRefBoundsIndexCapture scoreBounds(scores);
    IndexExpr bsIE = scoreBounds.getDim(0); // batch size.
    IndexExpr csIE = scoreBounds.getDim(1); // class size.
    IndexExpr ssIE = scoreBounds.getDim(2); // spatial size.
    Value bs = bsIE.getValue();
    Value cs = csIE.getValue();
    Value ss = ssIE.getValue();

    // Frequently used constants.
    Value zero = createMath.constantIndex(0);
    Value one = createMath.constantIndex(1);
    Value two = createMath.constantIndex(2);
    Value three = createMath.constantIndex(3);
    Value falseVal = createMath.constant(boolType, 0);
    Value trueVal = createMath.constant(boolType, 1);

    // Refine the number of output boxes per class by suppressing it using
    // spatial dimension size and score threshold.
    Value maxOutputPerClass =
        createMemref.alloca(MemRefType::get({}, indexType));
    // 1. Suppress by using spatial dimension size.
    Value x = createKrnl.load(maxOutputBoxPerClass, {zero});
    x = rewriter.create<arith::IndexCastOp>(loc, indexType, x);
    createKrnl.store(createMath.min(x, ss), maxOutputPerClass, {});
    // 2. Suppress by score threshold.
    // suppressByScores(rewriter, loc, scores, scoreThreshold,
    // maxOutputPerClass);

    // Sort scores in the descending order.
    Value order = emitArgSort(rewriter, loc, scores);

    // Bounding boxes may contain a mix of flipped and non-flipped boxes. Try to
    // unflip the flipped boxes.
    if (centerPointBox == 0)
      boxes = tryToUnflip(rewriter, loc, boxes);

    // Global parameters of NonMaxSuppression.
    Value scoreTH = createKrnl.load(scoreThreshold, {zero});
    Value iouTH = createKrnl.load(iouThreshold, {zero});
    Value MOPC = createKrnl.load(maxOutputPerClass, {});

    // The total number of output selected indices.
    IndexExpr numSelectedIndicesIE = bsIE * csIE * DimIndexExpr(MOPC);

    // Allocate a MemRef for the output. This MemRef is NOT the final output
    // since the number of selected indices has yet not suppressed by IOU. So
    // the first dimension size is larger than necessary.
    // Output shape : [num_selected_indices, 3]
    SmallVector<IndexExpr, 2> outputDims = {
        numSelectedIndicesIE, LiteralIndexExpr(3)};
    SmallVector<int64_t, 2> outputShape;
    if (numSelectedIndicesIE.isLiteral())
      outputShape.emplace_back(numSelectedIndicesIE.getLiteral());
    else
      outputShape.emplace_back(-1);
    outputShape.emplace_back(3);
    Value selectedMemRef = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get(outputShape, indexType), loc, outputDims,
        /*insertDealloc=*/true);
    // Initialize with value -1.
    createKrnl.memset(selectedMemRef, createMath.constantIndex(-1));

    // Effective number of selected indices. This is the final value for the 1st
    // dim of the output, which is suppressed by IOU during computation and
    // cannot be computed in advance.
    // Final output shape : [effective_num_selected_indices, 3]
    Value effectiveNumSelectedIndices =
        createMemref.alloca(MemRefType::get({}, indexType));
    createKrnl.store(zero, effectiveNumSelectedIndices, {});

    // Suppress by using IOU.
    // Iterate over all bounding boxes in the descending order of scores.
    ValueRange bcLoopDef = createKrnl.defineLoops(2);
    createKrnl.iterate(bcLoopDef, bcLoopDef, {zero, zero}, {bs, cs},
        [&](KrnlBuilder &createKrnl, ValueRange bcLoopInd) {
          MemRefBuilder createMemref(createKrnl);

          // Keep trace of the number of output boxes per class.
          Value effectiveMaxOutputPerClass =
              createMemref.alloca(MemRefType::get({}, indexType));
          createKrnl.store(zero, effectiveMaxOutputPerClass, {});
          // Keep trace of removed indices per class.
          DimIndexExpr ssIE(ss);
          SmallVector<IndexExpr, 1> dims = {ssIE};
          SmallVector<int64_t, 1> shapes = {-1};
          if (ssIE.isLiteral())
            shapes[0] = ssIE.getLiteral();
          Value removedIndices = insertAllocAndDeallocSimple(rewriter, nullptr,
              MemRefType::get(shapes, boolType), loc, dims,
              /*insertDealloc=*/true);
          createKrnl.memset(removedIndices, falseVal);

          // Iterate in the descending order of scores.
          ValueRange sLoopDef = createKrnl.defineLoops(1);
          createKrnl.iterate(sLoopDef, sLoopDef, {zero}, {ss},
              [&](KrnlBuilder &createKrnl, ValueRange sLoopInd) {
                Value b(bcLoopInd[0]), c(bcLoopInd[1]), s(sLoopInd[0]);
                AffineBuilderKrnlMem createAffine(createKrnl);
                MathBuilder createMath(createKrnl);

                Value score = createKrnl.load(scores, {b, c, s});

                // Check conditions to select a bounding box.
                // 1. Only bounding boxes whose score > score_threshold.
                Value checkScore = createMath.sgt(score, scoreTH);
                // 2. Have not yet got enough outputs.
                Value currentMOPC =
                    createKrnl.load(effectiveMaxOutputPerClass, {});
                Value checkMOPC = createMath.slt(currentMOPC, MOPC);
                // 3. Bounding box has not yet been removed.
                Value isRemoved = createKrnl.load(removedIndices, {s});
                Value isNotRemoved = createMath.eq(isRemoved, falseVal);

                // Only proceed if the box satisfies the above conditions.
                Value canSelectBox = createMath._and(
                    createMath._and(checkScore, checkMOPC), isNotRemoved);
                auto ifOp = rewriter.create<scf::IfOp>(
                    loc, canSelectBox, /*withElseRegion=*/false);
                rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());

                // Select the bounding box with the largest score.
                Value selectedBI = createKrnl.load(order, {b, c, s});
                SmallVector<Value, 4> selectedBox;
                for (int i = 0; i < 4; ++i) {
                  Value iVal = createMath.constantIndex(i);
                  Value x = createKrnl.load(boxes, {b, selectedBI, iVal});
                  selectedBox.emplace_back(x);
                }

                // Compute the position to store the selected box.
                // out_index = b * batch_size + c * max_output_per_class +
                //             effective_max_output_per_class;
                DimIndexExpr bIE(b), cIE(c), bsIE(bs);
                DimIndexExpr MOPCIE(MOPC), effectiveMOPCIE(currentMOPC);
                IndexExpr soIE = bIE * bsIE + cIE * MOPCIE + effectiveMOPCIE;

                // Store the index of the selected box to the output.
                // selected_indices[out_index] = [b, c, selected_box_index]
                Value soVal = soIE.getValue();
                createKrnl.store(b, selectedMemRef, {soVal, zero});
                createKrnl.store(c, selectedMemRef, {soVal, one});
                createKrnl.store(selectedBI, selectedMemRef, {soVal, two});

                // Update the number of output boxes per class.
                // effective_max_output_per_class += 1
                IndexExpr newEffectiveMOPCIE =
                    effectiveMOPCIE + LiteralIndexExpr(1);
                createKrnl.store(newEffectiveMOPCIE.getValue(),
                    effectiveMaxOutputPerClass, {});

                // Remove boxes overlapped too much with the selected box,
                // using IOU.
                ValueRange oLoopDef = createKrnl.defineLoops(1);
                createKrnl.iterate(oLoopDef, oLoopDef, {zero}, {ss},
                    [&](KrnlBuilder &createKrnl, ValueRange oLoopInd) {
                      Value o(oLoopInd[0]);
                      MathBuilder createMath(createKrnl);

                      // Only proceed if a box has not yet been removed.
                      Value isRemoved = createKrnl.load(removedIndices, {o});
                      Value isNotRemoved = createMath.eq(isRemoved, falseVal);
                      auto if1Op = rewriter.create<scf::IfOp>(
                          loc, isNotRemoved, /*withElseRegion=*/false);
                      rewriter.setInsertionPointToStart(
                          &if1Op.thenRegion().front());

                      // Pick the current box.
                      SmallVector<Value, 4> otherBox;
                      for (int i = 0; i < 4; ++i) {
                        Value iVal = createMath.constantIndex(i);
                        Value x = createKrnl.load(boxes, {b, o, iVal});
                        otherBox.emplace_back(x);
                      }

                      // Compute IOU between the selected and current boxes.
                      Value iou = emitIOU(
                          createMath, selectedBox, otherBox, centerPointBox);

                      // Only proceed if IOU > iou_threshold.
                      Value checkIOU = createMath.sgt(iou, iouTH);
                      auto if2Op = rewriter.create<scf::IfOp>(
                          loc, checkIOU, /*withElseRegion=*/false);
                      rewriter.setInsertionPointToStart(
                          &if2Op.thenRegion().front());

                      // If IOU is satified, mark the current box as removed.
                      createKrnl.store(trueVal, removedIndices, {o});
                    });
              });
          // Update the effective numbers of selected indices.
          Value effectiveMOPC = createKrnl.load(effectiveMaxOutputPerClass, {});
          Value effectiveNSI = createKrnl.load(effectiveNumSelectedIndices, {});
          Value newEffectiveNSI = createMath.add(effectiveNSI, effectiveMOPC);
          createKrnl.store(newEffectiveNSI, effectiveNumSelectedIndices, {});
        });

    // Insert allocation and deallocation for the final output.
    Value effectiveNSI = createKrnl.load(effectiveNumSelectedIndices, {});
    SmallVector<IndexExpr, 2> resDims = {
        DimIndexExpr(effectiveNSI), LiteralIndexExpr(3)};
    bool insertDealloc = checkInsertDealloc(op);
    Value resMemRef = insertAllocAndDeallocSimple(rewriter, op,
        MemRefType::get({-1, 3}, elementType), loc, resDims,
        /*insertDealloc=*/insertDealloc);

    // Copy data to the final ouput.
    ValueRange resLoopDef = createKrnl.defineLoops(2);
    createKrnl.iterate(resLoopDef, resLoopDef, {zero, zero},
        {effectiveNSI, three},
        [&](KrnlBuilder &createKrnl, ValueRange resLoopInd) {
          Value load = createKrnl.load(selectedMemRef, resLoopInd);
          Value res =
              rewriter.create<arith::IndexCastOp>(loc, elementType, load);
          createKrnl.store(res, resMemRef, resLoopInd);
        });

    rewriter.replaceOp(op, resMemRef);
    return success();
  }
};

void populateLoweringONNXNonMaxSuppressionOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXNonMaxSuppressionOpLowering>(ctx);
}

// clang-format off
// Below is a python implementation of NonMaxSuppression.
// import numpy as np
//
// def IOU(box1, box2, center_point_box=0):
//     if center_point_box == 0:
//         # The box data is supplied as [y1, x1, y2, x2]. (y1, x1) and (y2, x2)
//         # are the coordinates of the diagonal pair of bottom-left and top-right
//         # corners.
//         y1_min, x1_min, y1_max, x1_max = box1
//         y2_min, x2_min, y2_max, x2_max = box2
// 
//         area1 = (y1_max - y1_min) * (x1_max - x1_min)
//         area2 = (y2_max - y2_min) * (x2_max - x2_min)
//     else:
//         # The box data is supplied as [x_center, y_center, width, height].
//         x1_center, y1_center, w1, h1 = box1
//         x2_center, y2_center, w2, h2 = box2
// 
//         x1_min = x1_center - w1 / 2
//         x1_max = x1_center + w1 / 2
//         x2_min = x2_center - w2 / 2
//         x2_max = x2_center + w2 / 2
// 
//         y1_min = y1_center - h1 / 2
//         y1_max = y1_center + h1 / 2
//         y2_min = y2_center - h2 / 2
//         y2_max = y2_center + h2 / 2
// 
//         area1 = h1 * w1
//         area2 = h2 * w2
// 
//     intersection_x_min = max(x1_min, x2_min)
//     intersection_y_min = max(y1_min, y2_min)
//     intersection_x_max = min(x1_max, x2_max)
//     intersection_y_max = min(y1_max, y2_max)
//     intersection_area = max(intersection_x_max - intersection_x_min, 0) * \
//         max(intersection_y_max - intersection_y_min, 0)
// 
//     union_area = area1 + area2 - intersection_area + 1e-8
//     return intersection_area / union_area
// 
// 
// '''
// boxes :: [num_batch, spatial_dimension, 4]
// scores :: [num_batch, num_class, spatial_dimension]
// '''
// 
// 
// def nms(boxes, scores, max_output_boxes_per_class, iou_threshold,
//         score_threshold, center_point_box=0):
//     batch_size = scores.shape[0]
//     class_size = scores.shape[1]
//     spatial_size = boxes.shape[1]
// 
//     score_threshold = score_threshold[0]
//     iou_threshold = iou_threshold[0]
//     # Suppress by spatial dimension.
//     max_output_per_class = min(spatial_size, max_output_boxes_per_class[0])
//     # Suppress by scores
//     max_per_class_by_score = 0
//     for b in range(batch_size):
//         for c in range(class_size):
//             topk = 0
//             for s in range(spatial_size):
//                 if scores[b, c, s] > score_threshold:
//                     topk += 1
//             max_per_class_by_score = max(max_per_class_by_score, topk)
//     max_output_per_class = min(
//         max_output_per_class, max_per_class_by_score)
// 
//     # Sort scores in the descending order and get the sorted indices.
//     # order = np.argsort(-scores, axis=2)
//     order = np.full(scores.shape, -1)
//     for b in range(batch_size):
//         for c in range(class_size):
//             for i in range(spatial_size):
//                 order[b, c, i] = i
//     for b in range(batch_size):
//         for c in range(class_size):
//             for i in range(spatial_size - 1):
//                 for k in range(i+1, spatial_size):
//                      xOrd = order[b, c, i]
//                      yOrd = order[b, c, k]
//                      if (scores[b, c, xOrd] < scores[b, c, yOrd]):
//                          tmp = order[b, c, i]
//                          order[b, c, i] = order[b, c, k] 
//                          order[b, c, k] = tmp 
// 
// 
//     # Check if the coordinates are flipped. If so, flip them back.
//     if (center_point_box == 0):
//         new_boxes = np.empty(boxes.shape)
//         for b in range(batch_size):
//             for s in range(spatial_size):
//                 box = boxes[b, s]
//                 # Check whether the coordinates are flipped.
//                 y1_min, x1_min, y1_max, x1_max = box
//                 if (y1_min > y1_max):
//                     tmp = y1_min
//                     y1_min = y1_max
//                     y1_max = tmp
//                 if (x1_min > x1_max):
//                     tmp = x1_min
//                     x1_min = x1_max
//                     x1_max = tmp
//                 new_boxes[b, s] = [y1_min, x1_min, y1_max, x1_max]
//         boxes = new_boxes
// 
//     # Output: [num_selected_indices, 3]
//     # The selected index format is [batch_index, class_index, box_index].
//     num_selected_indices = batch_size * max_output_per_class * class_size
//     selected_indices_shape = (num_selected_indices, 3)
//     selected_indices = np.full(selected_indices_shape, -1).astype(np.int64)
//     effective_num_selected_indices = 0
//     for b in range(batch_size):
//         for c in range(class_size):
//             effective_max_output_per_class = 0
//             removed_indices = np.full((spatial_size), False)
//             for s in range(spatial_size):
//                 # Discard bounding boxes using score threshold.
//                 if (scores[b, c, s] <= score_threshold):
//                     continue
//                 # Have enough the number of outputs.
//                 if (effective_max_output_per_class >= max_output_per_class):
//                     continue
//                 # Removed, ignore.
//                 if removed_indices[s]:
//                     continue
// 
//                 # Pick the bounding box with the largest score.
//                 selected_box_index = order[b, c, s]
//                 selected_box = boxes[b, selected_box_index, :]
// 
//                 # Store the index of the picked box to the output.
//                 out_index = b * batch_size + c * max_output_per_class + effective_max_output_per_class
//                 selected_indices[out_index] = [b, c, selected_box_index]
//                 effective_max_output_per_class += 1
// 
//                 # Remove boxes overlapped too much with the selected box, using
//                 # IOU.
//                 for o in range(spatial_size):
//                     other_box = boxes[b, o, :]
//                     iou = IOU(selected_box, other_box, center_point_box)
//                     if (not removed_indices[o]) and (iou > iou_threshold):
//                         removed_indices[o] = True
//                     else:
//                         removed_indices[o] = removed_indices[o]
//             effective_num_selected_indices += effective_max_output_per_class
// 
//     # Since we cannot suppress by IOU in advance, so remove redundant score
//     # now.
//     res = np.empty((effective_num_selected_indices, 3))
//     for i in range(effective_num_selected_indices):
//         res[i] = selected_indices[i]
//     return res 
// 
// 
// print("testing nonmaxsuppression_center_point_box_format")
// center_point_box = 1
// boxes = np.array([[
//     [0.5, 0.5, 1.0, 1.0],
//     [0.5, 0.6, 1.0, 1.0],
//     [0.5, 0.4, 1.0, 1.0],
//     [0.5, 10.5, 1.0, 1.0],
//     [0.5, 10.6, 1.0, 1.0],
//     [0.5, 100.5, 1.0, 1.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold, center_point_box)
// np.testing.assert_allclose(selected_indices, out)
// 
// print("testing nonmaxsuppression_flipped_coordinates")
// boxes = np.array([[
//     [1.0, 1.0, 0.0, 0.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, 0.9, 1.0, -0.1],
//     [0.0, 10.0, 1.0, 11.0],
//     [1.0, 10.1, 0.0, 11.1],
//     [1.0, 101.0, 0.0, 100.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
// 
// print("testing nonmaxsuppression_identical_boxes")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
// 
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.0, 1.0, 1.0]
// ]]).astype(np.float32)
// scores = np.array(
//     [[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
// 
// print("testing nonmaxsuppression_limit_output_size")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, -0.1, 1.0, 0.9],
//     [0.0, 10.0, 1.0, 11.0],
//     [0.0, 10.1, 1.0, 11.1],
//     [0.0, 100.0, 1.0, 101.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([2]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
// 
// print("testing nonmaxsuppression_single_box")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
// 
// print("testing nonmaxsuppression_suppress_by_IOU")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, -0.1, 1.0, 0.9],
//     [0.0, 10.0, 1.0, 11.0],
//     [0.0, 10.1, 1.0, 11.1],
//     [0.0, 100.0, 1.0, 101.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
// 
// print("testing nonmaxsuppression_suppress_by_IOU_and_scores")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, -0.1, 1.0, 0.9],
//     [0.0, 10.0, 1.0, 11.0],
//     [0.0, 10.1, 1.0, 11.1],
//     [0.0, 100.0, 1.0, 101.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([3]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.4]).astype(np.float32)
// selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
// 
// print("testing nonmaxsuppression_two_batches")
// boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
//                    [0.0, 0.1, 1.0, 1.1],
//                    [0.0, -0.1, 1.0, 0.9],
//                    [0.0, 10.0, 1.0, 11.0],
//                    [0.0, 10.1, 1.0, 11.1],
//                    [0.0, 100.0, 1.0, 101.0]],
//                   [[0.0, 0.0, 1.0, 1.0],
//                    [0.0, 0.1, 1.0, 1.1],
//                    [0.0, -0.1, 1.0, 0.9],
//                    [0.0, 10.0, 1.0, 11.0],
//                    [0.0, 10.1, 1.0, 11.1],
//                    [0.0, 100.0, 1.0, 101.0]]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
//                    [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([2]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array(
//     [[0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
// 
// print("testing nonmaxsuppression_two_classes")
// boxes = np.array([[
//     [0.0, 0.0, 1.0, 1.0],
//     [0.0, 0.1, 1.0, 1.1],
//     [0.0, -0.1, 1.0, 0.9],
//     [0.0, 10.0, 1.0, 11.0],
//     [0.0, 10.1, 1.0, 11.1],
//     [0.0, 100.0, 1.0, 101.0]
// ]]).astype(np.float32)
// scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
//                     [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
// max_output_boxes_per_class = np.array([2]).astype(np.int64)
// iou_threshold = np.array([0.5]).astype(np.float32)
// score_threshold = np.array([0.0]).astype(np.float32)
// selected_indices = np.array(
//     [[0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]]).astype(np.int64)
// out = nms(boxes, scores, max_output_boxes_per_class,
//           iou_threshold, score_threshold)
// np.testing.assert_allclose(selected_indices, out)
// 
// 
// # if __name__ == "__main__":
// #     main()
// clang-format on
