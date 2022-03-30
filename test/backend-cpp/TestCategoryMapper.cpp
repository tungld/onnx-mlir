/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- TestCategoryMapper.cpp - Test CategoryMapper Implementation -----===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains end-to-end tests for the CategoryMapper operator.
//
//===----------------------------------------------------------------------===//

#include "test/modellib/ModelLib.hpp"
#include "llvm/Support/FileSystem.h"

static const std::string SharedLibBaseName("./TestCategoryMapper_main_graph");

static bool testInt64ToStr() {
  using CategoryMapperBuilder =
      onnx_mlir::test::CategoryMapperLibBuilder<int64_t, const char *>;

  const CategoryMapperBuilder::CMAttributes attributes = {{1, 2, 3, 4, 5},
      {"cat", "dog", "human", "tiger", "beaver"}, -1, "unknown"};
  const ArrayRef<int64_t> input = {1, 2, 3, 4, 5};
  const ArrayRef<const char *> expResult = {
      "cat", "dog", "human", "tiger", "beaver"};

  CategoryMapperBuilder categoryMapper(
      SharedLibBaseName, attributes, input, expResult);

  bool passed = categoryMapper.build() && categoryMapper.compileAndLoad() &&
                categoryMapper.prepareInputs() && categoryMapper.run() &&
                categoryMapper.verifyOutputs();
  if (!passed)
    llvm::errs() << __func__ << " failed\n";

  return passed;
}

static bool testStrToInt64() {
  using CategoryMapperBuilder =
      onnx_mlir::test::CategoryMapperLibBuilder<const char *, int64_t>;

  const CategoryMapperBuilder::CMAttributes attributes = {{1, 2, 3, 4, 5},
      {"cat", "dog", "human", "tiger", "beaver"}, -1, "unknown"};
  const ArrayRef<const char *> input = {
      "dog", "human", "cat", "beaver", "tiger", "bird"};
  const ArrayRef<int64_t> expResult = {2, 3, 1, 5, 4, -1};

  CategoryMapperBuilder categoryMapper(
      SharedLibBaseName, attributes, input, expResult);

  bool passed = categoryMapper.build() && categoryMapper.compileAndLoad() &&
                categoryMapper.prepareInputs() && categoryMapper.run() &&
                categoryMapper.verifyOutputs();
  if (!passed)
    llvm::errs() << __func__ << " failed\n";

  return passed;
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(
      onnx_mlir::test::ModelLibBuilder::getSharedLibName(SharedLibBaseName));

  setCompilerOption(onnx_mlir::OptionKind::CompilerOptLevel, "3");
  registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestCategoryMapper\n", nullptr, "TEST_ARGS");

  bool rc = testInt64ToStr();
  rc &= testStrToInt64();

  return rc ? 0 : 1;
}
