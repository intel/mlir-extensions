// RUN: imex-opt -convert-xegpu-to-xevm -split-input-file %s | FileCheck %s

#sg_map_a_f16 = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#sg_map_b_f16 = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
#sg_map_c_f32 = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>

gpu.module @load_store_check {
    func.func @dpas(%a_loaded: vector<8x1xf16>, %b_loaded: vector<8x2xf16>, %c_loaded: vector<8x1xf32>) -> vector<8x1xf32> {
        // Loads are checked in a separate test.
        // Cast arguments to SIMT-style vectors.
        //CHECK: %[[CAST_A:.*]] = vector.shape_cast %arg0 : vector<8x1xf16> to vector<8xf16>
        //CHECK-NEXT: %[[CAST_B:.*]] = vector.shape_cast %arg1 : vector<8x2xf16> to vector<16xf16>
        //CHECK-NEXT: %[[CAST_C:.*]] = vector.shape_cast %arg2 : vector<8x1xf32> to vector<8xf32>
        //CHECK-NEXT: %[[D:.*]] = xevm.dpas %[[CAST_C]], %[[CAST_A]], %[[CAST_B]] {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xf16>, vector<16xf16>) -> vector<8xf32>
        // Cast result back to expected shape
        //CHECK-NEXT: %[[CAST_D:.*]] = vector.shape_cast %[[D]] : vector<8xf32> to vector<8x1xf32>
        %d = xegpu.dpas %a_loaded, %b_loaded, %c_loaded {sg_map_a = #sg_map_a_f16, sg_map_b = #sg_map_b_f16, sg_map_c = #sg_map_c_f32} : vector<8x1xf16>, vector<8x2xf16>, vector<8x1xf32> -> vector<8x1xf32>
        return %d : vector<8x1xf32>
    }
}
