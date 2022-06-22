/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlEntryPoint.cpp - Lower KrnlEntryPointOp -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlEntryPointOp operator.
//
//===----------------------------------------------------------------------===//

#include <errno.h>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/JSON.h"

#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

extern uint64_t KRNL_ENTRY_POINT_ID;

class KrnlEntryPointOpLowering : public OpRewritePattern<KrnlEntryPointOp> {
public:
  using OpRewritePattern<KrnlEntryPointOp>::OpRewritePattern;
  ArrayRef<bool> outputOMTensorOwnerships;
  bool singleEntryPoint;
  SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps;
  SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps;
  SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps;
  bool verifyInputTensors;

  KrnlEntryPointOpLowering(TypeConverter typeConverter, MLIRContext *ctx,
      ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint,
      SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
      SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
      SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps, bool verifyInputTensors)
      : OpRewritePattern<KrnlEntryPointOp>(ctx),
        outputOMTensorOwnerships(outputOMTensorOwnerships),
        singleEntryPoint(singleEntryPoint), entryGlobalOps(entryGlobalOps),
        inSigGlobalOps(inSigGlobalOps), outSigGlobalOps(outSigGlobalOps),
        verifyInputTensors(verifyInputTensors) {}

  LogicalResult matchAndRewrite(
      KrnlEntryPointOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
    auto module = op->getParentOfType<ModuleOp>();
    auto *context = module.getContext();
    const RuntimeAPIRegistry &apiRegistry =
        RuntimeAPIRegistry::build(module, rewriter);
    auto numOutputs = op->getAttrOfType<IntegerAttr>(
                            KrnlEntryPointOp::getNumOutputsAttrName())
                          .getInt();

    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto int64Ty = IntegerType::get(context, 64);

    // Rewrite Krnl Entry Point Operation to an LLVM function with a dynamic
    // signature. The signature is dynamic because it remains the same no matter
    // what the model input/output schema look like. Such dynamic signature
    // takes a opaque ptr as input, representing a ptr to a data structure
    // containing a set of dynamic memrefs wrapped in a vector; similarly the
    // output is also a opaque ptr to a data structure with output memrefs
    // wrapped within it.
    auto staticEntryPointFuncName =
        op->getAttrOfType<SymbolRefAttr>(
              KrnlEntryPointOp::getEntryPointFuncAttrName())
            .getLeafReference()
            .getValue();

    // When there is only a single entry point function in a model, use
    // DEFAULT_DYN_ENTRY_POINT.
    std::string dynEntryPointName = "run_" + staticEntryPointFuncName.str();
    if (singleEntryPoint)
      dynEntryPointName = DEFAULT_DYN_ENTRY_POINT;

    // Record entry point name, input and output signatures in order to emit
    // signature-related functions later.
    recordEntryPointSignatures(module, dynEntryPointName, op, entryGlobalOps,
        inSigGlobalOps, outSigGlobalOps);

    // Start lowering the op.
    rewriter.eraseOp(op);
    auto dynEntryPointFuncTy =
        LLVM::LLVMFunctionType::get(opaquePtrTy, {opaquePtrTy}, false);
    LLVM::LLVMFuncOp dynamicEntryPointFunc =
        create.llvm.func(dynEntryPointName, dynEntryPointFuncTy);
    auto &entryPointEntryBlock =
        createEntryBlock(dynEntryPointFuncTy, dynamicEntryPointFunc, loc);
    rewriter.setInsertionPointToStart(&entryPointEntryBlock);

    // Emit code to initialize accelerators by calling OMInitCompatibleAccelX
    // where X is the accelerator name.
    // OMInitCompatibleAccelX's signature is `i64 (i64)`.
    if (Attribute maccelAttr =
            module->getAttrOfType<::mlir::Attribute>("onnx-mlir.accels")) {
      assert(
          maccelAttr.isa<ArrayAttr>() && "onnx-mlir.accels must be ArrayAttr");
      ArrayAttr accels = maccelAttr.cast<ArrayAttr>();
      Value zeroI64 = create.llvm.constant(int64Ty, 0);

      // Split the current block into IF, THEN, ELSE blocks.
      Block *ifBlock, *thenBlock, *elseBlock;
      for (uint64_t i = 0; i < accels.size(); ++i) {
        assert(accels[i].isa<StringAttr>() && "Attribute must be StringAttr");
        StringRef accelStr = accels.getValue()[i].cast<StringAttr>().getValue();
        std::pair<StringRef, StringRef> NameAndVersion = accelStr.split('-');
        uint64_t versionNumberInHex =
            std::stoul(NameAndVersion.second.str(), nullptr, 16);
        FlatSymbolRefAttr funcRef = getOrInsertOMInitCompatibleAccel(
            rewriter, module, NameAndVersion.first);

        // Split the current block into IF, THEN, ELSE blocks.
        ifBlock = rewriter.getInsertionBlock();
        thenBlock = ifBlock->splitBlock(rewriter.getInsertionPoint());
        elseBlock = rewriter.createBlock(
            thenBlock->getParent(), std::next(Region::iterator(thenBlock)));

        // Emit code for the IF block.
        rewriter.setInsertionPointToEnd(ifBlock);
        // Call OMInitCompatibleAccelX.
        Value versionNumberVal =
            create.llvm.constant(int64Ty, versionNumberInHex);
        Value isCompatible = create.llvm.call(
            int64Ty, funcRef, ArrayRef<Value>({versionNumberVal}));
        // Condition: if (OMInitCompatibleAccelX() != 0)
        Value notCompatible = rewriter.create<LLVM::ICmpOp>(
            loc, LLVM::ICmpPredicate::eq, isCompatible, zeroI64);
        // Branch the block into the THEN and ELSE blocks.
        rewriter.create<LLVM::CondBrOp>(loc, notCompatible, thenBlock,
            ValueRange(), elseBlock, ValueRange());

        // Emit code for the THEN block: return NULL.
        rewriter.setInsertionPointToStart(thenBlock);
        create.llvm._return(create.llvm.nullI8Ptr());

        // Emit code for thenELSE block: deal with other accelerators if any.
        rewriter.setInsertionPointToStart(elseBlock);
      }
    }

    // Based on the static entry point type signature, unpack dynamic memory
    // refs to corresponding static memory refs.
    auto wrappedStaticEntryPointFuncName =
        "_mlir_ciface_" + staticEntryPointFuncName.lower();
    auto *staticEntryPointFunc =
        module.lookupSymbol(wrappedStaticEntryPointFuncName);
    assert(staticEntryPointFunc &&
           isa<LLVM::LLVMFuncOp>(staticEntryPointFunc) &&
           "entry point func must exist and be an llvm func op");
    auto staticEntryPointTy = dyn_cast<LLVM::LLVMFuncOp>(staticEntryPointFunc)
                                  .getFunctionType()
                                  .dyn_cast<LLVM::LLVMFunctionType>();

    // Retrieve dynamic mem refs from wrapped input, and convert every one of
    // them to static mem refs.
    SmallVector<Value, 4> staticInputs;
    auto wrappedInput = entryPointEntryBlock.getArgument(0);

    // Emit code to verify every tensor in the wrapped input, e.g. verifying
    // shape and data type.
    if (verifyInputTensors) {
      StringAttr sigAttr = op->getAttrOfType<StringAttr>(
          KrnlEntryPointOp::getSignatureAttrName());
      llvm::StringRef inSigJSON;
      std::tie(inSigJSON, std::ignore) = sigAttr.getValue().split('@');
      emitVerificationCodeForInputTensors(
          module, rewriter, loc, apiRegistry, wrappedInput, inSigJSON);
    }

    Value omTensorPtrArr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_OMT_ARRAY, {wrappedInput});
    Value one = create.llvm.constant(int64Ty, 1);

    // Create a memref type for the return argument of the iface call
    Type memRefOutPtrTy = staticEntryPointTy.getParamType(0);
    Value ptrToOutMemRef =
        create.llvm._alloca(memRefOutPtrTy, one, /*alignment=*/0);
    staticInputs.emplace_back(ptrToOutMemRef);

    // Start with param 1 because 0 is the return value
    for (size_t i = 1; i < staticEntryPointTy.getNumParams(); i++) {
      // Call API function to retrieve the i-th dynamic memref.
      Value idxVal = create.llvm.constant(int64Ty, i - 1);

      Type omTensorPtrAddrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
      Value omTensorPtrAddr = create.llvm.getElemPtr(
          omTensorPtrAddrTy, omTensorPtrArr, ArrayRef<Value>({idxVal}));
      Value omTensorPtr = create.llvm.load(omTensorPtrAddr);

      // Create a (static) memref type corresponding to the i-th memref input to
      // the inference function on stack, and load it to memRef.
      Type memRefPtrTy = staticEntryPointTy.getParamType(i);

      Value ptrToMemRef =
          create.llvm._alloca(memRefPtrTy, one, /*alignment=*/0);

      // Fill in the memref underlying ptrToMemRef with information extracted
      // from omTensorPtr.
      fillPtrToMemRefWithOMTensor(
          omTensorPtr, ptrToMemRef, rewriter, loc, apiRegistry, module);

      // ptrToMemRef will be an input to main computation graph function.
      staticInputs.emplace_back(ptrToMemRef);
    }

    // Call static entry point with the memref ptrs created, and get output.
    create.llvm.call(
        ArrayRef<Type>({}), wrappedStaticEntryPointFuncName, staticInputs);
    Value outMemRefs = create.llvm.load(ptrToOutMemRef);
    auto outMemRefsType = outMemRefs.getType().dyn_cast<LLVM::LLVMStructType>();

    std::vector<mlir::Value> outMemRefList;
    if (numOutputs == 1) {
      // If only one output tensor exists, the tensor's corresponding memref
      // descriptor will be returned as is.
      outMemRefList.emplace_back(outMemRefs);
    } else {
      // Otherwise, if multiple tensors are to be returned, the returned value
      // is a struct. Multiple tensors' memref descriptors are packed into the
      // same struct. So we unpack them iteratively to outMemRefList.
      for (int i = 0; i < numOutputs; i++) {
        Type type = outMemRefsType.getBody()[i];
        Value extractOp = create.llvm.extractValue(type, outMemRefs, {i});
        outMemRefList.emplace_back(extractOp);
      }
    }

    Value numOutput = create.llvm.constant(int64Ty, outMemRefList.size());

    auto mallocSym = getOrInsertMalloc(rewriter, module);
    // TODO(tjingrant): get pointer size from data layout.
    size_t kPtrSize = 8;
    Value outputOmtPtrsArraySizeInByte =
        create.llvm.constant(int64Ty, outMemRefList.size() * kPtrSize);
    Value outOmtPtrsArr = create.llvm.call(
        LLVM::LLVMPointerType::get(IntegerType::get(module.getContext(), 8)),
        mallocSym, ArrayRef<Value>(outputOmtPtrsArraySizeInByte));
    outOmtPtrsArr = create.llvm.bitcastI8PtrPtr(outOmtPtrsArr);

    for (unsigned int i = 0; i < outMemRefList.size(); i++) {
      // Get the i-th memref returned, convert to a dynamic memref and store it
      // in the wrappedOutput.

      auto memRef = outMemRefList.at(i);
      auto outMemRefTy = memRef.getType().dyn_cast<LLVM::LLVMStructType>();
      auto outMemRefRank = krnl::getRankFromMemRefType(outMemRefTy);
      Value outMemRefRankVal = create.llvm.constant(int64Ty, outMemRefRank);
      Value outOMTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
          RuntimeAPI::API::CREATE_OMTENSOR, {outMemRefRankVal});
      // If output is a constant tensor or a block argument, OMTensor does not
      // own it.
      bool outOwning = outputOMTensorOwnerships[i];
      LLVM_DEBUG(llvm::dbgs() << "Output OMTensor " << i
                              << " with owning = " << outOwning << "\n");
      krnl::fillOMTensorWithMemRef(
          memRef, outOMTensor, outOwning, rewriter, loc, apiRegistry, module);

      Value idxVal = create.llvm.constant(int64Ty, i);

      auto omTensorPtrAddrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
      auto omTensorPtrAddr = create.llvm.getElemPtr(
          omTensorPtrAddrTy, outOmtPtrsArr, ArrayRef<Value>{idxVal});

      create.llvm.store(outOMTensor, omTensorPtrAddr);
    }

    // Create wrapped output.
    Value wrappedOutput = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::CREATE_OMTENSOR_LIST, {outOmtPtrsArr, numOutput, one});

    // Return wrapped output.
    create.llvm._return(wrappedOutput);
    return success();
  }

private:
  // Helper function to insert an entry block to LLVM function.
  // (TODO): upstream this to MLIR.
  Block &createEntryBlock(Type &dynEntryPoint,
      LLVM::LLVMFuncOp &dynamicEntryPointFunc, Location &loc) const {
    // Add entry block:
    auto *entryPointEntryBlock = new Block();
    auto dynEntryPointFuncType = dynEntryPoint.cast<LLVM::LLVMFunctionType>();
    dynamicEntryPointFunc.push_back(entryPointEntryBlock);
    llvm::SmallVector<Type, 4> argTypes;
    for (size_t i = 0; i < dynEntryPointFuncType.getNumParams(); i++)
      argTypes.emplace_back(dynEntryPointFuncType.getParamType(i));
    auto argLocs = llvm::SmallVector<Location, 4>(
        dynEntryPointFuncType.getNumParams(), loc);
    entryPointEntryBlock->addArguments(argTypes, argLocs);
    return *entryPointEntryBlock;
  }

  void fillPtrToMemRefWithOMTensor(Value &rtMemRef, Value &ptrToMemRef,
      PatternRewriter &rewriter, const Location &loc,
      const RuntimeAPIRegistry &apiRegistry, ModuleOp &module) const {
    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
    auto *context = module.getContext();
    auto memRefPtrTy = ptrToMemRef.getType().dyn_cast<LLVM::LLVMPointerType>();
    auto memRefTy = memRefPtrTy.getElementType();
    auto int64Ty = IntegerType::get(context, 64);

    Value memRef = rewriter.create<LLVM::UndefOp>(loc, memRefTy);

    // Set dataPtr and alignedDataPtr;
    Value dataPtr = RuntimeAPI::callApi(
        rewriter, loc, apiRegistry, RuntimeAPI::API::GET_DATA, {rtMemRef});
    dataPtr = create.llvm.bitcast(
        memRefTy.cast<LLVM::LLVMStructType>().getBody()[0], dataPtr);
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
        dataPtr, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(0)}));
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
        dataPtr, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(1)}));

    // Use zero offset now.
    Value zero = create.llvm.constant(int64Ty, 0);
    memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef, zero,
        rewriter.getArrayAttr({rewriter.getI64IntegerAttr(2)}));

    // Get rank, sizes array ptr and strides array ptr.
    auto rank =
        krnl::getRankFromMemRefType(memRefTy.cast<LLVM::LLVMStructType>());
    Value sizesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_DATA_SHAPE, {rtMemRef});
    Value stridesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_DATA_STRIDES, {rtMemRef});

    for (decltype(rank) i = 0; i < rank; i++) {
      Value dimIdx = create.llvm.constant(int64Ty, i);
      // Insert size of the dimension.
      Value dimSizePtr =
          create.llvm.getElemPtr(LLVM::LLVMPointerType::get(int64Ty),
              sizesArrayPtr, ArrayRef<Value>({dimIdx}));
      Value dimSize = create.llvm.load(dimSizePtr);
      memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
          dimSize,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(3), rewriter.getI64IntegerAttr(i)}));

      // Insert stride of the dimension.
      auto dimStridePtr =
          create.llvm.getElemPtr(LLVM::LLVMPointerType::get(int64Ty),
              stridesArrayPtr, ArrayRef<Value>({dimIdx}));
      auto dimStride = create.llvm.load(dimStridePtr);
      memRef = rewriter.create<LLVM::InsertValueOp>(loc, memRefTy, memRef,
          dimStride,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(4), rewriter.getI64IntegerAttr(i)}));
    }

    create.llvm.store(memRef, ptrToMemRef);
  }

  FlatSymbolRefAttr getOrInsertMalloc(
      PatternRewriter &rewriter, ModuleOp module) const {
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    // Insert the malloc/aligned_alloc declaration if it is not already present.
    auto allocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
    auto ctx = rewriter.getContext();
    LLVMTypeConverter converter(ctx);
    if (!allocFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      SmallVector<Type, 2> callArgTypes = {converter.getIndexType()};
      // aligned_alloc(size_t alignment, size_t size)
      auto voidPtrType = LLVM::LLVMPointerType::get(
          IntegerType::get(&converter.getContext(), 8));
      allocFunc = create.llvm.func(
          "malloc", LLVM::LLVMFunctionType::get(voidPtrType, callArgTypes,
                        /*isVarArg=*/false));
    }
    return SymbolRefAttr::get(ctx, "malloc");
  }

  FlatSymbolRefAttr getOrInsertOMInitAccel(
      PatternRewriter &rewriter, ModuleOp module, StringRef accelName) const {
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    std::string funcName = "OMInitAccel" + accelName.str();
    // OMInitAccelX's signature is `void ()`.
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    MLIRContext *ctx = rewriter.getContext();
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = create.llvm.func(funcName,
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {}));
    }
    return SymbolRefAttr::get(ctx, funcName);
  }

  FlatSymbolRefAttr getOrInsertOMInitCompatibleAccel(
      PatternRewriter &rewriter, ModuleOp module, StringRef accelName) const {
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    std::string funcName = "OMInitCompatibleAccel" + accelName.str();
    // OMInitCompatibleAccelX's signature is `i64 (i64)`.
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    MLIRContext *ctx = rewriter.getContext();
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      LLVM::LLVMFunctionType funcType =
          LLVM::LLVMFunctionType::get(IntegerType::get(ctx, 64),
              ArrayRef<mlir::Type>({IntegerType::get(ctx, 64)}),
              /*isVarArg=*/false);
      func = create.llvm.func(funcName, funcType);
    }
    return SymbolRefAttr::get(ctx, funcName);
  }

  // Emit code for `IF lhs != rhs THEN return null ELSE do nothing`
  void equalOrFailed(ModuleOp &module, PatternRewriter &rewriter, Location loc,
      Value lhs, Value rhs, std::string errorMsg = "",
      bool appendRHS = true) const {
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    // Split the current block into IF, THEN, END blocks.
    Block *ifBlock, *thenBlock, *endBlock;
    ifBlock = rewriter.getInsertionBlock();
    thenBlock = ifBlock->splitBlock(rewriter.getInsertionPoint());
    endBlock = rewriter.createBlock(
        thenBlock->getParent(), std::next(Region::iterator(thenBlock)));

    // Emit code for the IF block.
    rewriter.setInsertionPointToEnd(ifBlock);
    Value failed =
        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, lhs, rhs);

    // Branch the block into the THEN and END blocks.
    rewriter.create<LLVM::CondBrOp>(
        loc, failed, thenBlock, ValueRange(), endBlock, ValueRange());

    // Emit code for the THEN block: return NULL.
    rewriter.setInsertionPointToStart(thenBlock);
    // Print an error message.
    KrnlBuilder createKrnl(rewriter, loc);
    if (appendRHS)
      createKrnl.printf(StringRef(errorMsg), rhs, rewriter.getI64Type(), true);
    else
      createKrnl.printf(StringRef(errorMsg + "\n"));
    // Set errno.
    krnl::emitErrNo(module, rewriter, loc, EINVAL);
    // Return NULL.
    create.llvm._return(create.llvm.nullI8Ptr());

    // Emit code for the END block: continue with other generated code.
    rewriter.setInsertionPointToStart(endBlock);
  }

  void emitVerificationCodeForInputTensors(ModuleOp &module,
      PatternRewriter &rewriter, Location loc,
      const RuntimeAPIRegistry &apiRegistry, Value wrappedInput,
      StringRef inSigJSON) const {
    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
    Type int64Ty = rewriter.getI64Type();
    Type opaquePtrTy = LLVM::LLVMPointerType::get(rewriter.getI8Type());

    auto JSONInput = llvm::json::parse(inSigJSON.data());
    assert(JSONInput && "failed to parse json");
    auto JSONArray = JSONInput->getAsArray();
    assert(JSONArray && "failed to parse json as array");
    int64_t inputNum = JSONArray->size();

    // Verify the number of inputs.
    equalOrFailed(module, rewriter, loc,
        create.llvm.constant(int64Ty, inputNum),
        RuntimeAPI::callApi(rewriter, loc, apiRegistry,
            RuntimeAPI::API::GET_OMTENSOR_LIST_SIZE, {wrappedInput}),
        "Wrong number of input tensors: expect " + std::to_string(inputNum) +
            ", but got ");

    // Get a pointer to the list of input omTensors.
    Value omTensorPtrArr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::GET_OMT_ARRAY, {wrappedInput});
    for (int i = 0; i < inputNum; ++i) {
      // Call API function to retrieve the i-th omTensor.
      Value idxVal = create.llvm.constant(int64Ty, i);
      Value omTensorPtrAddr =
          create.llvm.getElemPtr(LLVM::LLVMPointerType::get(opaquePtrTy),
              omTensorPtrArr, ArrayRef<Value>({idxVal}));
      Value omTensorPtr = create.llvm.load(omTensorPtrAddr);

      // Verify data type.
      auto JSONItem = (*JSONArray)[i].getAsObject();
      auto JSONItemType = JSONItem->getString("type");
      assert(JSONItemType && "failed to get type");
      Type elemTy = parseType(JSONItemType.getValue(), rewriter.getContext());
      std::string elemTyStr;
      llvm::raw_string_ostream dstream(elemTyStr);
      dstream << elemTy;
      dstream.flush();
      onnx::TensorProto::DataType dtype = krnl::mlirTypeToOnnxType(elemTy);
      equalOrFailed(module, rewriter, loc, create.llvm.constant(int64Ty, dtype),
          RuntimeAPI::callApi(rewriter, loc, apiRegistry,
              RuntimeAPI::API::GET_DATA_TYPE, {omTensorPtr}),
          "Wrong data type for the input " + std::to_string(i) + ": expect " +
              elemTyStr,
          false);

      // Verify data rank.
      auto JSONDimArray = JSONItem->getArray("dims");
      int64_t rank = JSONDimArray->size();
      equalOrFailed(module, rewriter, loc, create.llvm.constant(int64Ty, rank),
          RuntimeAPI::callApi(rewriter, loc, apiRegistry,
              RuntimeAPI::API::GET_DATA_RANK, {omTensorPtr}),
          "Wrong rank for the input " + std::to_string(i) + ": expect " +
              std::to_string(rank) + ", but got ");

      // Verify dimensions.
      Value sizesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
          RuntimeAPI::API::GET_DATA_SHAPE, {omTensorPtr});
      for (int d = 0; d < rank; ++d) {
        auto JSONDimValue = (*JSONDimArray)[d].getAsInteger();
        assert(JSONDimValue && "failed to get value");
        int64_t dim = JSONDimValue.getValue();
        if (dim == -1)
          continue; // do not verify dynamic dimensions.
        Value dimIdx = create.llvm.constant(int64Ty, d);
        equalOrFailed(module, rewriter, loc, create.llvm.constant(int64Ty, dim),
            create.llvm.load(
                create.llvm.getElemPtr(LLVM::LLVMPointerType::get(int64Ty),
                    sizesArrayPtr, ArrayRef<Value>({dimIdx}))),
            "Wrong size for the dimension " + std::to_string(d) +
                " of the input " + std::to_string(i) + ": expect " +
                std::to_string(dim) + ", but got ");
      }
    }
  }

  void recordEntryPointSignatures(ModuleOp &module,
      std::string currentEntryPointName, KrnlEntryPointOp entryOp,
      SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
      SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
      SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps) const {
    Operation *op = entryOp.getOperation();
    MLIRContext *context = module.getContext();
    Location loc = module.getLoc();
    OpBuilder b(context);

    // Common information.
    Type i8Type = IntegerType::get(context, 8);

    // A helper function to emit a global constant operation storing a string.
    auto emitGlobalOp = [&context, &b, &loc, &i8Type](
                            std::string name, std::string value) {
      mlir::StringAttr valueAttr = mlir::StringAttr::get(context, value);
      Type valueArrayType = LLVM::LLVMArrayType::get(i8Type, value.size());
      LLVM::GlobalOp globalOp = b.create<LLVM::GlobalOp>(loc, valueArrayType,
          /*isConstant=*/true, LLVM::Linkage::External, name, valueAttr);
      return globalOp;
    };

    bool zOS = false;
    if (Attribute mtripleAttr =
            module->getAttrOfType<::mlir::Attribute>("llvm.target_triple"))
      zOS = llvm::Triple(mtripleAttr.cast<StringAttr>().getValue()).isOSzOS();

    // NULL terminated entry point name.
    std::string terminatedEntryPointName = currentEntryPointName + '\0';
    terminatedEntryPointName = (zOS) ? krnl::a2e_s(terminatedEntryPointName)
                                     : terminatedEntryPointName;

    // Input/output signature strings.
    StringAttr sigAttr =
        op->getAttrOfType<StringAttr>(KrnlEntryPointOp::getSignatureAttrName());
    llvm::StringRef signature = sigAttr.getValue();
    auto splitSig = signature.split('@');
    std::string inSignature =
        (zOS) ? krnl::a2e_s(splitSig.first.str()) : splitSig.first.str();
    std::string outSignature =
        (zOS) ? krnl::a2e_s(splitSig.second.str()) : splitSig.second.str();

    // For each entry point name, emit three global constants to store the entry
    // point name and input/output signatures. For the i-th entry point, these
    // constants are named as follows:
    // - Entry point name: `_entry_point_i`.
    // - Input signature: `_entry_point_i_in_sig`.
    // - Output signature: `_entry_point_i_out_sig`.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    // Global constants for entry point names.
    std::string entryVarName =
        "_entry_point_" + std::to_string(KRNL_ENTRY_POINT_ID);
    KRNL_ENTRY_POINT_ID++;
    LLVM::GlobalOp entryGlobalOp =
        emitGlobalOp(entryVarName, terminatedEntryPointName);
    entryGlobalOps.emplace_back(entryGlobalOp);

    // Global constants for input signatures.
    std::string inSigVarName = entryVarName + "_in_sig";
    LLVM::GlobalOp inSigGlobalOp = emitGlobalOp(inSigVarName, inSignature);
    inSigGlobalOps.emplace_back(inSigGlobalOp);

    // Global constants for output signatures.
    std::string outSigVarName = entryVarName + "_out_sig";
    LLVM::GlobalOp outSigGlobalOp = emitGlobalOp(outSigVarName, outSignature);
    outSigGlobalOps.emplace_back(outSigGlobalOp);
  }
};

void populateLoweringKrnlEntryPointOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps, bool verifyInputTensors) {
  patterns.insert<KrnlEntryPointOpLowering>(typeConverter, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, verifyInputTensors);
}

} // namespace krnl
} // namespace onnx_mlir
