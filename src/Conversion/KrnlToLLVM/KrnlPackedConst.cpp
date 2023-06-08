/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlGetRefOp.cpp - Lower KrnlGetRefOp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlGetRefOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/Path.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Support/KrnlSupport.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlPackedConstOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlPackedConstOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            KrnlPackedConstantOp::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    auto packedConstOp = llvm::dyn_cast<KrnlPackedConstantOp>(op);
    LLVM::GlobalOp globalBase;
    // Some frequently used types.
    Type llvmI8PtrTy = getI8PointerType(context);
    Type llvmI64Ty = IntegerType::get(context, 64);
    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      globalBase = rewriter.create<LLVM::GlobalOp>(loc, llvmI8PtrTy,
          /*isConstant=*/false, LLVM::Linkage::Internal, "packedConst",
          nullptr);
    }

    auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
    assert(mainFunc);

    rewriter.setInsertionPoint(
        &mainFunc.getBody().front(), mainFunc.getBody().front().begin());

    //  - Initialize the global constant base.
    Value basePtrAddr = rewriter.create<LLVM::AddressOfOp>(loc, globalBase);
    auto getEmbeddedConstPoolRef = create.llvm.getOrInsertSymbolRef(module,
        KrnlPackedConstantOp::getEmbeddedDataLoaderMethodName(), llvmI8PtrTy,
        ArrayRef<Type>{llvmI64Ty}, /*isVarArg=*/false);
    auto constPackSize = rewriter.create<LLVM::ConstantOp>(
        loc, IntegerType::get(context, 64), packedConstOp.getSizeInBytesAttr());
    Value alloc = rewriter
                      .create<func::CallOp>(loc, getEmbeddedConstPoolRef,
                          llvmI8PtrTy, ArrayRef<Value>({constPackSize}))
                      .getResult(0);
    rewriter.create<LLVM::StoreOp>(loc, alloc, basePtrAddr);
    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      // Record constant pack *file path* as a global variable (by recording the
      // file path string's underlying char array + its length).
      const auto &fileNameAttr = packedConstOp.getFileNameAttr();
      auto fileNameAttrArrayType = LLVM::LLVMArrayType::get(
          IntegerType::get(context, 8), fileNameAttr.getValue().size());
      rewriter.create<LLVM::GlobalOp>(loc, fileNameAttrArrayType,
          /*isConstant=*/true, LLVM::Linkage::External,
          mlir::KrnlPackedConstantOp::getConstPackFilePathSymbolName(),
          fileNameAttr);
      auto fileNameAttrIntType = IntegerType::get(context, 64);
      rewriter.create<LLVM::GlobalOp>(loc, fileNameAttrIntType,
          /*isConstant=*/true, LLVM::Linkage::External,
          mlir::KrnlPackedConstantOp::getConstPackFilePathStrLenSymbolName(),
          rewriter.getI64IntegerAttr(fileNameAttr.getValue().size()));

      // Record constant pack *file name* as a global variable (by recording the
      // file name string's underlying char array + its length).
      auto constPackFileName =
          llvm::sys::path::filename(fileNameAttr.getValue());
      auto fileNameArrayType = LLVM::LLVMArrayType::get(
          IntegerType::get(context, 8), constPackFileName.size());
      rewriter.create<LLVM::GlobalOp>(loc, fileNameArrayType,
          /*isConstant=*/true, LLVM::Linkage::External,
          mlir::KrnlPackedConstantOp::getConstPackFileNameSymbolName(),
          rewriter.getStringAttr(constPackFileName));
      auto fileNameIntType = IntegerType::get(context, 64);
      rewriter.create<LLVM::GlobalOp>(loc, fileNameIntType, /*isConstant=*/true,
          LLVM::Linkage::External,
          mlir::KrnlPackedConstantOp::getConstPackFileNameStrLenSymbolName(),
          rewriter.getI64IntegerAttr(constPackFileName.size()));

      auto type = IntegerType::get(context, 8);
      rewriter.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
          LLVM::Linkage::External,
          mlir::KrnlPackedConstantOp::getConstPackIsLESymbolName(),
          rewriter.getI8IntegerAttr(packedConstOp.getIsLe()));
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  static int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
    return (a.getValue()[i]).cast<IntegerAttr>().getInt();
  }
};

void populateLoweringKrnlPackedConstOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlPackedConstOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
