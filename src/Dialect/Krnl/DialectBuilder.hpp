/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------- DialectBuilder.hpp - Krnl Dialect Builder -----------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the Krnl Dialect Builder.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

//====-------------------- Support for Krnl Builder ----------------------===//

struct KrnlBuilder : public DialectBuilder {
  KrnlBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  KrnlBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  KrnlBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~KrnlBuilder() {}

  mlir::Value load(mlir::Value memref, mlir::ValueRange indices = {}) const;
  // When ranks of offsets<indices, add offsets to the least significant dims.
  mlir::Value load(mlir::Value memref, mlir::ValueRange indices,
      mlir::ValueRange offsets) const;
  mlir::Value loadIE(
      mlir::Value memref, mlir::ArrayRef<IndexExpr> indices) const;
  void store(
      mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {}) const;
  // When ranks of offsets<indices, add offsets to the least significant dims.
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices,
      mlir::ValueRange offsets) const;
  void storeIE(mlir::Value val, mlir::Value memref,
      mlir::ArrayRef<IndexExpr> indices) const;

  void seqstore(mlir::Value element, mlir::Value seq, mlir::Value index) const;
  void seqstore(mlir::Value element, mlir::Value seq, IndexExpr index) const;

  mlir::Value vectorTypeCast(mlir::Value sourceMemref, int64_t vectorLen) const;

  mlir::ValueRange defineLoops(int64_t originalLoopNum) const;
  mlir::ValueRange block(mlir::Value loop, int64_t blockSize) const;
  void permute(mlir::ValueRange loops, mlir::ArrayRef<int64_t> map) const;
  mlir::ValueRange getInductionVarValue(mlir::ValueRange loops) const;
  void parallel(mlir::ValueRange loops) const;

  // Lambda passes loop indices as 2nd parameter.
  void iterate(mlir::ValueRange originalLoops, mlir::ValueRange optimizedLoops,
      mlir::ValueRange lbs, mlir::ValueRange ubs,
      mlir::function_ref<void(
          KrnlBuilder &createKrnl, mlir::ValueRange indices)>
          bodyBuilderFn) const;
  mlir::KrnlIterateOp iterate(
      const krnl::KrnlIterateOperandPack &operands) const;

  // Lambda passes loop indices as 2nd parameter.
  void iterateIE(mlir::ValueRange originalLoops,
      mlir::ValueRange optimizedLoops, mlir::ArrayRef<IndexExpr> lbs,
      mlir::ArrayRef<IndexExpr> ubs,
      mlir::function_ref<void(
          KrnlBuilder &createKrnl, mlir::ValueRange indices)>
          bodyBuilderFn) const;

  void copyToBuffer(
      // Buffer and source memory. Source memref may have a higher rank than
      // buffer.
      mlir::Value bufferMemref, mlir::Value sourceMemref,
      // Indices that points to the first data to be copied from source.
      // Starts has the same rank as sourceMemref.
      mlir::ValueRange starts,
      // If padding is needed, value to pad.
      mlir::Value padValue,
      // Now the bufferMemref may be larger than the actual data to be stored
      // in the buffer, if the user want to pad the data to a higher size.
      // TileSize enables the user to
      mlir::ArrayRef<int64_t> tileSize, mlir::ArrayRef<int64_t> padToNext,
      bool transpose = false) const;
  void copyToBuffer(mlir::Value bufferMemref, mlir::Value sourceMemref,
      mlir::ValueRange starts, mlir::Value padValue,
      bool transpose = false) const;

  void copyFromBuffer(mlir::Value bufferMemref, mlir::Value memref,
      mlir::ValueRange starts, mlir::ArrayRef<int64_t> tileSize) const;
  void copyFromBuffer(mlir::Value bufferMemref, mlir::Value memref,
      mlir::ValueRange starts) const;

  void matmul(
      // The a/b/cStart are the indices at the beginning of the buffer/mem
      // A/B/C.
      mlir::Value A, mlir::ValueRange aStart, mlir::Value B,
      mlir::ValueRange bStart, mlir::Value C, mlir::ValueRange cStart,
      // Loops are the krnl loop indices that this matmul replaces
      mlir::ValueRange loops,
      // the computeStarts indicate the i/j/k indices pointing to the beginning
      // of the matmul computation.
      mlir::ValueRange computeStarts,
      // The globalUBs are the global bounds on the original I, J, K
      // dimensions.
      mlir::ValueRange globalUBs,
      // If not the full A, B, C buffers are used by this matmul, meaning the
      // matmul uses a subtile of the buffers, this compute tile size
      // specifies the actual size of the i/j/k computations. Empty means
      // compute tiles encompass the entire buffer A, B, and C as defined by
      // their tile sizes.
      mlir::ArrayRef<int64_t> computeTileSize,
      // If buffers A, B, or C were padded, then the tile sizes give the size
      // of the non-padded data, basically the size of the data when the tile
      // is full. Partial tiles (due to computation on the edges of the
      // matrices) are handled differently (using the UBs), so no need to
      // worry about this. Empty means no padding was used.
      mlir::ArrayRef<int64_t> aTileSize, mlir::ArrayRef<int64_t> bTileSize,
      mlir::ArrayRef<int64_t> cTileSize,
      // Optimizations for code gen.
      bool simdize, bool unroll, bool overCompute) const;
  void matmul(mlir::Value A, mlir::ValueRange aStart, mlir::Value B,
      mlir::ValueRange bStart, mlir::Value C, mlir::ValueRange cStart,
      mlir::ValueRange loops, mlir::ValueRange computeStarts,
      mlir::ValueRange globalUBs, bool simdize, bool unroll,
      bool overCompute) const;

  mlir::KrnlMovableOp movable() const;

  mlir::KrnlGetRefOp getRef(mlir::Type type, mlir::Value memref,
      mlir::Value offset, mlir::ValueRange indices = {}) const;

  mlir::Value constant(mlir::MemRefType type, mlir::StringRef name,
      std::optional<mlir::Attribute> value,
      std::optional<mlir::IntegerAttr> offset = std::nullopt,
      std::optional<mlir::IntegerAttr> alignment = std::nullopt) const;

  // C library functions.
  void memcpy(mlir::Value dest, mlir::Value src, mlir::Value numElems) const;
  void memcpy(mlir::Value dest, mlir::Value src, mlir::Value numElems,
      mlir::Value destOffset, mlir::Value srcOffset) const;
  void memset(mlir::Value dest, mlir::Value val, bool delayed = false) const;
  mlir::Value strncmp(
      mlir::Value str1, mlir::Value str2, mlir::Value len) const;
  mlir::Value strlen(mlir::Value str) const;
  void printf(mlir::StringRef msg) const;
  void printf(mlir::StringRef msg, mlir::Value input, mlir::Type inputType,
      bool endsWithNewLine = false) const;
  void printf(mlir::Value input, mlir::Type inputType) const;

  // Onnx-mlir runtime functions.
  void randomNormal(mlir::Value alloc, mlir::Value numberOfRandomValues,
      mlir::Value mean, mlir::Value scale, mlir::Value seed) const;
  mlir::Value findIndex(
      mlir::Value input, mlir::Value G, mlir::Value V, mlir::Value len) const;
  void printTensor(mlir::StringRef msg, mlir::Value input) const;
};

//====--- Support for Affine Builder with Krnl Mem Ops ------------------===//

// We use here a Affine builder that generates Krnl Load and Store ops instead
// of the affine memory ops directly. This is because we can still generate
// Krnl Ops while lowering the dialect, and the big advantage of the Krnl memory
// operations is that they distinguish themselves if they are affine or not.
using AffineBuilderKrnlMem =
    GenericAffineBuilder<mlir::KrnlLoadOp, mlir::KrnlStoreOp>;

// =============================================================================
// IndexExpr Builder for building
// =============================================================================

struct IndexExprBuilderForKrnl : IndexExprBuilder {
  IndexExprBuilderForKrnl(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForKrnl(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(b, loc) {}
  IndexExprBuilderForKrnl(const DialectBuilder &db) : IndexExprBuilder(db) {}
  virtual ~IndexExprBuilderForKrnl() {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) final;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) final;
};

// =============================================================================
// MultiDialectBuilder for Krnl
// =============================================================================

// Recursive class specialized for AffineBuilderKrnlMem refereed to as
// affineKMem.
template <class... Ts>
struct MultiDialectBuilder<AffineBuilderKrnlMem, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), affineKMem(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), affineKMem(db) {}
  AffineBuilderKrnlMem affineKMem;
};

// Recursive class specialized for KrnlBuilder referred to as krnl.
template <class... Ts>
struct MultiDialectBuilder<KrnlBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), krnl(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), krnl(db) {}
  KrnlBuilder krnl;
};

// Recursive class specialized for IndexExprBuilderForKrnl referred to as
// krnlIE.
template <class... Ts>
struct MultiDialectBuilder<IndexExprBuilderForKrnl, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), krnlIE(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), krnlIE(db) {}
  IndexExprBuilderForKrnl krnlIE;
};

} // namespace onnx_mlir
