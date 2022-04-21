// RUN: onnx-mlir --maccel=NNPA --printIR %s | FileCheck %s

// CHECK: module attributes {llvm.data_layout = "E-{{.*}}", llvm.target_triple = "{{.*}}", "onnx-mlir.accels" = ["NNPA"]} {
module {
}

