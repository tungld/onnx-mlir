// RUN: onnx-mlir-opt --maccel=NNPA --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func @test_sub(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_sub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Sub"([[VAR_0_]], [[VAR_1_]]) : (tensor<10x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>, tensor<10x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<10x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<10x10xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

// COM:  Do not lower broadcasting onnx.Sub to zHigh.
func @test_sub_not_lowered_diff_shape(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub_not_lowered_diff_shape
}

// -----

/// Do not lower onnx.Sub to zHigh if inputs have unknown dimensions
/// because we cannot statically check whether it is really broadcasting or not.
func @test_sub_not_lowered_unknown_dims(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub_not_lowered_unknown_dims
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func @test_exceed_limit_sub(%arg0 : tensor<32769x10xf32>, %arg1 : tensor<32769x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<32769x10xf32>, tensor<32769x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_exceed_limit_sub
// CHECK:        "onnx.Sub"
}
