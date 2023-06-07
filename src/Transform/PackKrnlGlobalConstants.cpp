/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- ElideKrnlGlobalConstants.cpp - Krnl Constant lobal Value Elision ---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// In practice, the constant values of Global Krnl operations may be large
// enough to hinder the readability of the MLIR intermediate representation.
//
// This file creates a pass which elides the explicit values of constant
// global operations. This pass has purely cosmetic purposes and should only be
// run to obtain a compact representation of the program when emitting Krnl
// dialect code. This pass should never be invoked on code meant to be run.
//
//===----------------------------------------------------------------------===//
#include <fstream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FileSystem.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"
#include "src/Transform/ElideKrnlGlobalConstants.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {

/*!
 *  Function pass that performs constant value elision of Krnl globals.
 */
class PackKrnlGlobalConstantsPass
    : public PassWrapper<PackKrnlGlobalConstantsPass, OperationPass<ModuleOp>> {
public:
  /// Make sure that we have a valid default constructor and copy constructor to
  /// make sure that the options are initialized properly.
  PackKrnlGlobalConstantsPass() = default;
  PackKrnlGlobalConstantsPass(const PackKrnlGlobalConstantsPass &pass) {}

  StringRef getArgument() const override { return "pack-krnl-constants"; }

  StringRef getDescription() const override {
    return "Elide the constant values of the Global Krnl operations.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(&getContext());

    // Packing constant arrays to packedConst.
    std::vector<char> packedConst;
    module.walk([&](KrnlGlobalOp op) {
      // Do not support packing aligned constants at this moment.
      if (op.getAlignmentAttr() &&
          op.getAlignmentAttr().getValue().getSExtValue() != 0) {
        return;
      }
      assert(op.getValue());
      op.setOffsetAttr(builder.getI64IntegerAttr(packedConst.size()));
      assert(op.getValue()->isa<DenseElementsAttr>());
      // || op.getValue()->isa<OpaqueElementsAttr>());
      if (op.getValue()->isa<DenseElementsAttr>()) {
        const auto &denseAttr = op.getValueAttr().cast<DenseElementsAttr>();
        auto numElements = denseAttr.getNumElements();
        if (numElements <= elisionThreshold || denseAttr.isSplat())
          return;

        // TODO(tjingrant) verify we can actually use the raw data.
        std::vector<char> rawData = denseAttr.getRawData();
        packedConst.insert(packedConst.end(), rawData.begin(), rawData.end());
      } else {
        // TODO(tung):
        // const auto &opaqueAttr = op.getValueAttr().cast<OpaqueElementsAttr>();
        // StringRef bytes = opaqueAttr.getValue();
        // packedConst.insert(
        //     packedConst.end(), bytes.bytes_begin(), bytes.bytes_end());
      }
    });

    // Remove value attributes from krnl constant op.
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<KrnlConstGlobalValueElision>(
        &getContext(), elisionThreshold);
    // Apply constant value elision.
    module.walk([&](func::FuncOp func) {
      applyPatternsAndFoldGreedily(func, std::move(patterns));
    });

    bool isLE = llvm::support::endian::system_endianness() ==
                llvm::support::endianness::little;
    mlir::OperationState state(module.getLoc(), "krnl.packed_const");
    KrnlPackedConstantOp::build(builder, state,
        builder.getIntegerType(/*width=*/64),
        /*size_in_bytes=*/builder.getI64IntegerAttr(packedConst.size()),
        /*is_le=*/builder.getBoolAttr(isLE),
        /*value=*/nullptr,
        /*file_name=*/nullptr);
    auto packedConstOp =
        llvm::cast<mlir::KrnlPackedConstantOp>(mlir::Operation::create(state));
    module.insert(module.begin(), packedConstOp);
    if (moveToFile) {
      std::string pathStr;
      if (filename.hasValue()) {
        pathStr = filename.getValue();
      } else {
        llvm::SmallVector<char, 10> path;
        llvm::sys::fs::createTemporaryFile("packed_const", "tmp", path);
        pathStr = std::string(path.begin(), path.end());
      }
      packedConstOp.setFileNameAttr(builder.getStringAttr(pathStr));
      std::ofstream outfile(pathStr, std::ofstream::binary);
      outfile.write(packedConst.data(), packedConst.size());
    } else {
      auto shapeTy =
          RankedTensorType::get({static_cast<int64_t>(packedConst.size())},
              builder.getIntegerType(8));
      auto denseAttr =
          DenseIntElementsAttr::get(shapeTy, llvm::makeArrayRef(packedConst));
      packedConstOp.setValueAttr(denseAttr);
    }
  }

  Option<bool> moveToFile{*this, "move-to-file",
      llvm::cl::desc("Whether to move the packed constant to a file."),
      llvm::cl::init(true)};
  Option<int64_t> elisionThreshold{*this, "elision-threshold",
      llvm::cl::desc(
          "A threshold value specifying the maximum number of elements a "
          "constant operation can hold as an attribute. If the number exceeds "
          "this threshold, constants will be packed together and, in the case "
          "where `move-to-file` option is enabled, stored as a  binary file on "
          "disk. This can help preserve readability of IR dump and improve "
          "compilation speed."),
      llvm::cl::init(KrnlConstGlobalValueElision::kDefaultElisionThreshold)};
  Option<std::string> filename{*this, "filename",
      llvm::cl::desc(
          "Specify a file in which the packed constant is to be stored.")};
};
} // namespace

std::unique_ptr<Pass> onnx_mlir::krnl::createPackKrnlGlobalConstantsPass() {
  return std::make_unique<PackKrnlGlobalConstantsPass>();
}
