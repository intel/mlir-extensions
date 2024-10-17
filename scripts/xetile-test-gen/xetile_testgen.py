import os
import math
import glob
import sys
import argparse
import pandas as pd
import xetile_testgen_utils as utils
from xetile_testgen_utils import WGLevel,SGLevel,Tile,Grid,TestParams

class CPUCode:
    def __init__(self, params):
        self.params = params

    def gen_k_loop(self, step):
        ab_vals = ""
        if self.params.A.type == self.params.C.type:
            ab_vals = f"""
                %a_val = memref.load %A[%i, %k_dpas] : memref<{self.params.A.memref()}>
                %b_val = memref.load %B[%b_k_dpas, %j] : memref<{self.params.B.memref()}>\n"""
        else:
            ab_vals = f"""
                        %a_loaded = memref.load %A[%i, %k_dpas] : memref<{self.params.A.memref()}>
                        %b_loaded = memref.load %B[%b_k_dpas, %j] : memref<{self.params.B.memref()}>
                        %a_val = arith.extf %a_loaded : {self.params.A.type} to {self.params.C.type}
                        %b_val = arith.extf %b_loaded : {self.params.B.type} to {self.params.C.type}\n"""
        op_suffix = 'i' if self.params.A.type_str == 'i' else 'f'
        k_loop = f"""
                %c_val = scf.for %k_tile = %c0 to %c{self.params.A.y} step %c{step} iter_args(%c_partial = %c_curr) -> {self.params.C.type} {{
                    %c_val_dpas = scf.for %k = %c0 to %c{step} step %c1 iter_args(%c_dpas_partial = %c_partial) -> {self.params.C.type} {{
                        %k_dpas = arith.addi %k_tile, %k : index
                        %b_k_dpas = arith.addi %B_offset_x, %k_dpas: index
                        {ab_vals}
                        %t = arith.mul{op_suffix} %a_val, %b_val : {self.params.C.type}
                        %c_sum = arith.add{op_suffix} %t, %c_dpas_partial : {self.params.C.type}
                        scf.yield %c_sum : {self.params.C.type}
                    }}
                    scf.yield %c_val_dpas : {self.params.C.type}
                }}
        """
        return k_loop

    def get_cpu_validation_code(self):
        if self.params.test_args.validate:
            cpu_validation = f"""
        call @cpu_reference(%A, %B, %C_ref) : (memref<{self.params.A.memref()}>, memref<{self.params.B.memref()}>, memref<{self.params.C.memref()}>) -> ()
        %cast_C = memref.cast %2 : memref<{self.params.C.memref()}> to memref<*x{self.params.C.type}>
        %cast_C_ref = memref.cast %C_ref : memref<{self.params.C.memref()}> to memref<*x{self.params.C.type}>
        call @printAllclose{self.params.C.runtime_call_suffix}(%cast_C, %cast_C_ref) : (memref<*x{self.params.C.type}>, memref<*x{self.params.C.type}>) -> ()
        """
        else:
            cpu_validation = "// Generated without validation"
        return cpu_validation

    def get_cpu_initialization_code(self):
        A_value = ""
        if self.params.A.type_str == 'i':
            A_value = f"""%val = index.castu %j : index to i{self.params.A.type_nbit}"""
        elif self.params.A.type_str == 'bf':
            A_value = f"""%t = index.castu %j : index to i{self.params.A.type_nbit}
                          %val = arith.bitcast %t : i{self.params.A.type_nbit} to {self.params.A.type}\n"""
        else:
            A_value = f"""%t = index.castu %j : index to i{self.params.A.type_nbit}
                          %val = arith.uitofp %t : i{self.params.A.type_nbit} to {self.params.A.type}\n"""

        B_value = ""
        if self.params.B.type_str == 'i':
            B_value = f"""%c1_i = index.castu %c1 : index to i{self.params.B.type_nbit}
                          %matrix_idx_i = index.castu %matrix_idx_in_batch : index to i{self.params.B.type_nbit}
                          %val = arith.addi %c1_i, %matrix_idx_i : {self.params.B.type_nbit}\n"""
        elif self.params.B.type_str == 'bf':
            B_value = f"""%c1_i = index.castu %c1 : index to i{self.params.B.type_nbit}
                          %matrix_idx_i = index.castu %matrix_idx_in_batch : index to i{self.params.B.type_nbit}
                          %matrix_val = arith.addi %c1_i, %matrix_idx_i : {self.params.B.type_nbit}
                          %val = arith.bitcast %matrix_val : i{self.params.B.type_nbit} to {self.params.B.type} \n"""
        else:
            B_value = f"""%matrix_idx_i = index.castu %matrix_idx_in_batch : index to i{self.params.B.type_nbit}
                    %matrix_idx = arith.uitofp %matrix_idx_i: i{self.params.B.type_nbit} to {self.params.B.type}
                    %val = arith.addf %c_{self.params.B.type}_1, %matrix_idx : {self.params.B.type} \n"""

        if self.params.test_args.validate:
            A_B_init_code = f"""
        // Matrix A : A[i, j] = j, since we store column value for A, we don't need to adjust for batch (which is collapsed x-dim)
        scf.for %i = %c0 to %c{self.params.A.x} step %c1 {{
            scf.for %j = %c0 to %c{self.params.A.y} step %c1 {{
                {A_value}
                memref.store %val, %A[%i, %j] : memref<{self.params.A.memref()}>
            }}
        }}

        // Matrix B : identity matrix
        scf.for %i = %c0 to %c{self.params.B.x} step %c1 {{
            %matrix_idx_in_batch = arith.divui %i, %c{self.params.B.get_x_dim_for_one_in_batch()} : index
            scf.for %j = %c0 to %c{self.params.B.y} step %c1 {{
                // Batched means collapsed x-dim, we want each matrix in the batch to be identity
                %i_batch_local = arith.remui %i, %c{self.params.B.get_x_dim_for_one_in_batch()} : index
                %i_i32_batch_local = index.castu %i_batch_local : index to i32
                %j_i32 = index.castu %j : index to i32
                %i_j_same = arith.cmpi eq, %i_i32_batch_local, %j_i32 : i32
                scf.if %i_j_same {{
                    {B_value}
                    memref.store %val, %B[%i, %j] : memref<{self.params.B.memref()}>
                }} else {{
                    memref.store %c_{self.params.B.type}_0, %B[%i, %j] : memref<{self.params.B.memref()}>
                }}
            }}
        }}
            """
        else:
            A_B_init_code = f"""
        %c_gen_int = arith.constant 0 : i1
        %cf_lower = arith.constant -0.5 : f32
        %cf_upper = arith.constant 0.5 : f32

        %A_random = memref.cast %A : memref<{self.params.A.memref()}> to memref<*x{self.params.A.type}>
        call @fillResource1DRandom{self.params.A.runtime_call_suffix}(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*x{self.params.A.type}>, f32, f32, i1) -> ()

        %B_random = memref.cast %B : memref<{self.params.B.memref()}> to memref<*x{self.params.B.type}>
        call @fillResource1DRandom{self.params.B.runtime_call_suffix}(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*x{self.params.B.type}>, f32, f32, i1) -> ()
        """
        return A_B_init_code


class GPUCode:
    def __init__(self, params):
        self.params = params
        prefetch_limits = {'min_x':8, 'max_x':32, 'min_y': 16, 'max_y': 32}

        if self.params.test_args.code_version == "prefetch":
            self.A_prefetch_tile, self.A_prefetch_layout_for_SG_tile = utils.get_prefetch_size_and_layout_in_SG_tile(self.params.SG.layout, self.params.SG.A_tile, prefetch_limits)
            self.B_prefetch_tile, self.B_prefetch_layout_for_SG_tile = utils.get_prefetch_size_and_layout_in_SG_tile(self.params.SG.layout, self.params.SG.B_tile, prefetch_limits, wg_slice_direction='x')
            self.params.constants += (self.B_prefetch_tile.x, self.B_prefetch_tile.y,
                                    self.B_prefetch_layout_for_SG_tile.x, self.B_prefetch_layout_for_SG_tile.y,
                                    self.A_prefetch_tile.x, self.A_prefetch_tile.y,
                                    self.A_prefetch_layout_for_SG_tile.x, self.A_prefetch_layout_for_SG_tile.y
                                    )
            self.params.constants = list(set(self.params.constants))

            self.A_wg_slice = Tile(x=self.params.WG.A_tile.x, y=self.params.k_loop_step, type=self.params.WG.A_tile.type)
            self.B_wg_slice = Tile(x=self.params.k_loop_step, y=self.params.WG.B_tile.y, type=self.params.WG.B_tile.type)
            if self.params.test_args.print_debug:
                print(f"MKN: {input_matrices['M']},{input_matrices['K']},{input_matrices['N']}")
                print(f"Matrix shapes: A={self.params.A}, B={self.params.B}, C={self.params.C}")
                print(f"{self.params.WG}")
                print(f"{self.params.SG}")
                print(f"k-loop step: {self.params.k_loop_step}")
                print(f"A WG slice {self.A_wg_slice} moves along y (→) in {self.params.WG.A_tile}")
                print(f"A Prefetch size: {self.A_prefetch_tile}, Prefetch layout for {self.A_prefetch_layout_for_SG_tile} SG Tile: {self.params.SG.A_tile}")
                print(f"B WG slice {self.B_wg_slice} moves along x (↓) in {self.params.WG.B_tile}")
                print(f"B Prefetch size: {self.B_prefetch_tile}, Prefetch layout for {self.B_prefetch_layout_for_SG_tile} SG Tile: {self.params.SG.B_tile}")
                print('_' * 60)

    def gen_prefetch_prologue(self):
        template = f"""
            %wg_tile_offset_x_local = arith.muli %wg_id_x, %c{self.params.WG.C_tile.x} : index
            %wg_tile_offset_x_global = arith.addi %C_global_start_x, %wg_tile_offset_x_local : index // Global means accounted for batch
            %wg_tile_offset_y = arith.muli %wg_id_y, %c{self.params.WG.C_tile.y} : index

            %local_sg_id_temp = arith.muli %local_sg_id_x, %c{self.params.SG.layout.y} : index
            %local_sg_id = arith.addi %local_sg_id_temp, %local_sg_id_y : index

            // (N x k-step) slice of A matrix moves along y (→), so all columns of a row of SG layout have to collaborate on prefetch
            // Start of the SG within Slice
            %A_sg_start_x_offset = arith.muli %local_sg_id_x, %c{self.params.SG.A_tile.x} : index
            // X-layout index of the prefetch tile within SG
            %A_sg_prefetch_tile_x_idx = arith.divui %local_sg_id_y, %c{self.A_prefetch_layout_for_SG_tile.y} : index
            // Row start of the prefetch tile within SG
            %A_sg_prefetch_tile_x_start = arith.muli %A_sg_prefetch_tile_x_idx, %c{self.A_prefetch_tile.x} : index
            // Row start of the prefetch tile within Slice
            %A_slice_prefetch_tile_x_start = arith.addi %A_sg_start_x_offset, %A_sg_prefetch_tile_x_start : index
            // Row start of the prefetch tile within Workgroup (global means accounted for batch)
            %A_prefetch_tile_x_start_global = arith.addi %wg_tile_offset_x_global, %A_slice_prefetch_tile_x_start : index

            // Y-layout index of the prefetch tile within SG
            %A_sg_prefetch_tile_y_idx = arith.remui %local_sg_id_y, %c{self.A_prefetch_layout_for_SG_tile.y} : index
            // Column start of the prefetch tile within SG/Slice (here we expect their Y-shape to match)
            %A_slice_prefetch_tile_y_start = arith.muli %A_sg_prefetch_tile_y_idx, %c{self.A_prefetch_tile.y} : index

            %A_sg_prefetch_tile_iter0 = xetile.init_tile %A[%A_prefetch_tile_x_start_global, %A_slice_prefetch_tile_y_start] : memref<{self.params.A.memref()}> -> !xetile.tile<{self.A_prefetch_tile.memref()}>
            xetile.prefetch_tile %A_sg_prefetch_tile_iter0 : !xetile.tile<{self.A_prefetch_tile.memref()}>
            %A_sg_prefetch_tile_iter1 = xetile.update_tile_offset %A_sg_prefetch_tile_iter0, [%c0, %c{self.params.k_loop_step}] : !xetile.tile<{self.A_prefetch_tile.memref()}>
            xetile.prefetch_tile %A_sg_prefetch_tile_iter1 : !xetile.tile<{self.A_prefetch_tile.memref()}>
            %A_sg_prefetch_tile_iter2 = xetile.update_tile_offset %A_sg_prefetch_tile_iter1, [%c0, %c{self.params.k_loop_step}] : !xetile.tile<{self.A_prefetch_tile.memref()}>
            xetile.prefetch_tile %A_sg_prefetch_tile_iter2 : !xetile.tile<{self.A_prefetch_tile.memref()}>
            %A_sg_prefetch_tile_iter3 = xetile.update_tile_offset %A_sg_prefetch_tile_iter2, [%c0, %c{self.params.k_loop_step}] : !xetile.tile<{self.A_prefetch_tile.memref()}>


            // (k-step x N) slice of A matrix moves along x (↓), so all rows of a column of SG layout have to collaborate on prefetch
            // X-layout index of the prefetch tile within SG
            %B_sg_prefetch_tile_x_idx = arith.remui %local_sg_id_x, %c{self.B_prefetch_layout_for_SG_tile.x} : index
            // Column start of the prefetch tile within SG/Slice (here we expect their Y-shape to match)
            %B_slice_prefetch_tile_x_start_local = arith.muli %B_sg_prefetch_tile_x_idx, %c{self.B_prefetch_tile.x} : index

            // Adjust for batch
            %B_slice_prefetch_tile_x_start_global = arith.addi %B_global_start_x, %B_slice_prefetch_tile_x_start_local : index

            // Start of the SG within Slice
            %B_sg_start_y_offset = arith.muli %local_sg_id_y, %c{self.params.SG.B_tile.y} : index
            // X-layout index of the prefetch tile within SG
            %B_sg_prefetch_tile_y_idx = arith.divui %local_sg_id_x, %c{self.B_prefetch_layout_for_SG_tile.x} : index
            // Row start of the prefetch tile within SG
            %B_sg_prefetch_tile_y_start = arith.muli %B_sg_prefetch_tile_y_idx, %c{self.B_prefetch_tile.y} : index
            // Row start of the prefetch tile within Slice
            %B_slice_prefetch_tile_y_start = arith.addi %B_sg_start_y_offset, %B_sg_prefetch_tile_y_start : index
            // Row start of the prefetch tile within Workgroup
            %B_prefetch_tile_y_start = arith.addi %wg_tile_offset_y, %B_slice_prefetch_tile_y_start : index

            %B_sg_prefetch_tile_iter0 = xetile.init_tile %B[%B_slice_prefetch_tile_x_start_global, %B_prefetch_tile_y_start] : memref<{self.params.B.memref()}> -> !xetile.tile<{self.B_prefetch_tile.memref()}>
            xetile.prefetch_tile %B_sg_prefetch_tile_iter0 : !xetile.tile<{self.B_prefetch_tile.memref()}>
            %B_sg_prefetch_tile_iter1 = xetile.update_tile_offset %B_sg_prefetch_tile_iter0, [%c{self.params.k_loop_step}, %c0] : !xetile.tile<{self.B_prefetch_tile.memref()}>
            xetile.prefetch_tile %B_sg_prefetch_tile_iter1 : !xetile.tile<{self.B_prefetch_tile.memref()}>
            %B_sg_prefetch_tile_iter2 = xetile.update_tile_offset %B_sg_prefetch_tile_iter1, [%c{self.params.k_loop_step}, %c0] : !xetile.tile<{self.B_prefetch_tile.memref()}>
            xetile.prefetch_tile %B_sg_prefetch_tile_iter2 : !xetile.tile<{self.B_prefetch_tile.memref()}>
            %B_sg_prefetch_tile_iter3 = xetile.update_tile_offset %B_sg_prefetch_tile_iter2, [%c{self.params.k_loop_step}, %c0] : !xetile.tile<{self.B_prefetch_tile.memref()}>
        """
        return template

    def gen_prologue(self):
        template = f"""
            {utils.generate_constants(self.params.constants, depth=6)}
            %c0_i32 = arith.constant 0 : i32
            %wg_id_x = gpu.block_id x
            %wg_id_y = gpu.block_id y
            %wg_id_z = gpu.block_id z
            // Batch size is z-dim in workgroups.
            // All of the tiling happens within one matrix, if batch is >1, then the general logic is not affected,
            //  but loads, stores or tile initializations from matrix x-dim need to be adjusted.

            %global_sg_id_x = gpu.global_id x
            %global_sg_id_y = gpu.global_id y
            %local_sg_id_x = arith.remui %global_sg_id_x, %c{self.params.SG.layout.x} : index
            %local_sg_id_y = arith.remui %global_sg_id_y, %c{self.params.SG.layout.y} : index

            // Batch size is embedded in x-dimension of matrix shape, so we need to adjust: WG_id along z-dimension indicates matrix_id in our batch
            %B_global_start_x = arith.muli %c{self.params.B.get_x_dim_for_one_in_batch()}, %wg_id_z : index
            %C_global_start_x = arith.muli %c{self.params.C.get_x_dim_for_one_in_batch()}, %wg_id_z : index
            %C_sg_tile_offset_x_local = arith.muli %global_sg_id_x, %c{self.params.SG.C_tile.x} : index
            %C_sg_tile_offset_x_global = arith.addi %C_global_start_x, %C_sg_tile_offset_x_local : index

            %C_sg_tile_offset_y = arith.muli %global_sg_id_y, %c{self.params.SG.C_tile.y} : index // y-dim of a matrix is not affected by the batch size, no need to adjust
            """
        if self.params.test_args.code_version == "prefetch":
            template += f"\n{self.gen_prefetch_prologue()}"
        template += f"""
            %A_sg_init_tile = xetile.init_tile %A[%C_sg_tile_offset_x_global, %c0] : memref<{self.params.A.memref()}> -> !xetile.tile<{self.params.SG.A_tile.memref()}>
            %B_sg_init_tile = xetile.init_tile %B[%B_global_start_x, %C_sg_tile_offset_y] : memref<{self.params.B.memref()}> -> !xetile.tile<{self.params.SG.B_tile.memref()}>

            %c_sg_tile = xetile.init_tile %C[%C_sg_tile_offset_x_global, %C_sg_tile_offset_y] : memref<{self.params.C.memref()}> -> !xetile.tile<{self.params.SG.C_tile.memref()}>
            // %c_init_val = xetile.load_tile %c_sg_tile : !xetile.tile<{self.params.SG.C_tile.memref()}> -> vector<{self.params.SG.C_tile.memref()}>
            %c_init_val = arith.constant dense<0.0> : vector<{self.params.SG.C_tile.memref()}>
"""
        if self.params.test_args.code_version == "prefetch":
            dpas_shape = Grid(x=8, y=16)
            number_of_dpas_c_tile = math.ceil(self.params.SG.C_tile.x / dpas_shape.x) * math.ceil(self.params.SG.C_tile.y/ dpas_shape.y)
            template += f"""
            xegpu.alloc_nbarrier {number_of_dpas_c_tile}
            %nbarrier_id = arith.constant 1 : i8
            %nthreads = arith.constant {self.params.SG.layout.num_elements()} : i8
            %nbarrier = xegpu.init_nbarrier %nbarrier_id, %nthreads : i8, i8 -> !xegpu.nbarrier
            """
        return template

    def gen_k_loop(self):
        barrier_every_X_iter = math.ceil(self.params.WG.C_tile.x / self.params.k_loop_step)
        barrier_condition = barrier_every_X_iter * self.params.k_loop_step
        num_loop_args = 5 if self.params.test_args.code_version == "prefetch" else 3

        template = f"""
        // K loop advances in {self.params.k_loop_step} steps
        %k_loop_result:{num_loop_args} = scf.for %k = %c0 to %c{self.params.A.y} step %c{self.params.k_loop_step} iter_args (
            %A_tile = %A_sg_init_tile,
            %B_tile = %B_sg_init_tile,
            %c_val = %c_init_val"""
        if self.params.test_args.code_version == "prefetch":
            template += f""",
            %A_prefetch_tile = %A_sg_prefetch_tile_iter3, %B_prefetch_tile = %B_sg_prefetch_tile_iter3"""
        template += f"""    ) ->
            (!xetile.tile<{self.params.SG.A_tile.memref()}>, !xetile.tile<{self.params.SG.B_tile.memref()}>,
            vector<{self.params.SG.C_tile.memref()}>"""
        if self.params.test_args.code_version == "prefetch":
            template += f""",
            !xetile.tile<{self.A_prefetch_tile.memref()}>, !xetile.tile<{self.B_prefetch_tile.memref()}>"""
        template += """
            ){"""
        if self.params.test_args.code_version == "prefetch":
            template += f"""
            // all SGs must arrive here first
            %every_{barrier_every_X_iter}th_iter = arith.remui %k, %c{barrier_condition} : index
            %every_{barrier_every_X_iter}th_iter_i32 = arith.index_cast %every_{barrier_every_X_iter}th_iter : index to i32
            %every_{barrier_every_X_iter}th_iter_cond = arith.cmpi eq, %every_{barrier_every_X_iter}th_iter_i32, %c0_i32 : i32
            scf.if %every_{barrier_every_X_iter}th_iter_cond  {{
            xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
            }}"""

        template += f"""
            %a_val = xetile.load_tile %A_tile : !xetile.tile<{self.params.SG.A_tile.memref()}> -> vector<{self.params.SG.A_tile.memref()}>
            %b_val = xetile.load_tile %B_tile  : !xetile.tile<{self.params.SG.B_tile.memref()}> -> vector<{self.params.SG.B_tile.memref()}>
        """

        if self.params.test_args.code_version == "prefetch":
            template += f"""
            xegpu.compile_hint
            xetile.prefetch_tile %A_prefetch_tile : !xetile.tile<{self.A_prefetch_tile.memref()}>
            xetile.prefetch_tile %B_prefetch_tile : !xetile.tile<{self.B_prefetch_tile.memref()}>
            xegpu.compile_hint

            %next_A_prefetch_tile = xetile.update_tile_offset %A_prefetch_tile, [%c0, %c{self.params.k_loop_step}] : !xetile.tile<{self.A_prefetch_tile.memref()}>
            %next_B_prefetch_tile = xetile.update_tile_offset %B_prefetch_tile, [%c{self.params.k_loop_step}, %c0] : !xetile.tile<{self.B_prefetch_tile.memref()}>
        """

        template += f"""
            %next_A_tile = xetile.update_tile_offset %A_tile, [%c0, %c{self.params.k_loop_step}]  : !xetile.tile<{self.params.SG.A_tile.memref()}>
            %next_B_tile = xetile.update_tile_offset %B_tile, [%c{self.params.k_loop_step}, %c0]  : !xetile.tile<{self.params.SG.B_tile.memref()}>

            xegpu.compile_hint
            %new_c_val = xetile.tile_mma %a_val, %b_val, %c_val : vector<{self.params.SG.A_tile.memref()}>, vector<{self.params.SG.B_tile.memref()}>, vector<{self.params.SG.C_tile.memref()}> -> vector<{self.params.SG.C_tile.memref()}>
            xegpu.compile_hint
        """

        if self.params.test_args.code_version == "prefetch":
            template += f"""
            //  barrier wait
            scf.if %every_{barrier_every_X_iter}th_iter_cond {{
            xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
            }}
        """

        template += f"""
            scf.yield %next_A_tile, %next_B_tile, %new_c_val"""
        if self.params.test_args.code_version == "prefetch":
            template += f""",%next_A_prefetch_tile, %next_B_prefetch_tile
            """

        template += f"""
                : !xetile.tile<{self.params.SG.A_tile.memref()}>, !xetile.tile<{self.params.SG.B_tile.memref()}>, vector<{self.params.SG.C_tile.memref()}>"""
        if self.params.test_args.code_version == "prefetch":
            template += f""", !xetile.tile<{self.A_prefetch_tile.memref()}>, !xetile.tile<{self.B_prefetch_tile.memref()}>
            """
        template += """
            }"""
        return template

    def gen_epilogue(self):
        template = f"""xetile.store_tile %k_loop_result#2 , %c_sg_tile : vector<{self.params.SG.C_tile.memref()}>, !xetile.tile<{self.params.SG.C_tile.memref()}>"""
        return template


class GEMM_test:
    def __init__(self, input_params, test_args):
        self.params = TestParams(input_params, test_args)
        self.cpu_code = CPUCode(self.params)
        self.gpu_code = GPUCode(self.params)

    def gen_test_driver(self):
        template = f"""func.func @test(%A: memref<{self.params.A.memref()}>, %B: memref<{self.params.B.memref()}>, %C: memref<{self.params.C.memref()}>) -> memref<{self.params.C.memref()}> attributes {{llvm.emit_c_interface}} {{
        {utils.generate_constants(self.params.constants, depth=4)}
        %A_gpu = gpu.alloc  host_shared () : memref<{self.params.A.memref()}>  // Batch means collapsed x-dim
        memref.copy %A, %A_gpu : memref<{self.params.A.memref()}> to memref<{self.params.A.memref()}>
        %B_gpu = gpu.alloc  host_shared () : memref<{self.params.B.memref()}>
        memref.copy %B, %B_gpu : memref<{self.params.B.memref()}> to memref<{self.params.B.memref()}>
        %C_gpu = gpu.alloc  host_shared () : memref<{self.params.C.memref()}>
        memref.copy %C, %C_gpu : memref<{self.params.C.memref()}> to memref<{self.params.C.memref()}>
        gpu.launch_func  @test_kernel::@test_kernel blocks in (%c{self.params.block_count_m}, %c{self.params.block_count_n}, %c{self.params.batch_size}) threads in (%c{self.params.thread_count_m}, %c{self.params.thread_count_n}, %c1) args(%A_gpu : memref<{self.params.A.memref()}>, %B_gpu : memref<{self.params.B.memref()}>, %C_gpu : memref<{self.params.C.memref()}>)
        gpu.dealloc  %A_gpu : memref<{self.params.A.memref()}>
        gpu.dealloc  %B_gpu : memref<{self.params.B.memref()}>
        return %C_gpu : memref<{self.params.C.memref()}>
    }}"""
        return template

    def get_cpu_reference(self):
        dpas_step = 16
        self.params.constants.append(dpas_step)
        self.params.constants = list(set(self.params.constants))
        template = f"""func.func @cpu_reference(%A : memref<{self.params.A.memref()}>, %B : memref<{self.params.B.memref()}>, %C : memref<{self.params.C.memref()}>) {{
        {utils.generate_constants(self.params.constants, depth=4)}
        scf.for %i = %c0 to %c{self.params.C.x} step %c1 {{
            %matrix_idx_in_batch = arith.divui %i, %c{self.params.C.get_x_dim_for_one_in_batch()} : index
            %B_offset_x = arith.muli %matrix_idx_in_batch, %c{self.params.B.get_x_dim_for_one_in_batch()} : index

            scf.for %j = %c0 to %c{self.params.C.y} step %c1 {{
                %c_curr = memref.load %C[%i, %j] : memref<{self.params.C.memref()}>
                {self.cpu_code.gen_k_loop(step = dpas_step)}
                memref.store %c_val , %C[%i, %j] : memref<{self.params.C.memref()}>
            }}
        }}
        return
    }}"""
        return template

    def gen_gpu_module(self):
        template = f"""gpu.module @test_kernel attributes {{spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>}} {{
        gpu.func @test_kernel(%A: memref<{self.params.A.memref()}>, %B: memref<{self.params.B.memref()}>, %C: memref<{self.params.C.memref()}>) kernel attributes {{VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}} {{
        {self.gpu_code.gen_prologue()}
        {self.gpu_code.gen_k_loop()}
        {self.gpu_code.gen_epilogue()}
        gpu.return
        }}
    }}"""
        return template

    def gen_main(self):
        if self.params.A.type_str == 'i':
            list_of_constants=[0,1,2]
        else:
            list_of_constants=[0.0,1.0,2.0]

        C_zero = "0" if self.params.C.type_str == 'i' else "0.0"
        template = f"""func.func @main() attributes {{llvm.emit_c_interface}} {{
        {utils.generate_constants(self.params.constants, depth=4)}
        {utils.generate_constants(list_of_constants=list_of_constants, prefix=f"_{self.params.A.type}_", cst_type=self.params.A.type, depth=4)}
        %A = memref.alloc() : memref<{self.params.A.memref()}>
        %B = memref.alloc() : memref<{self.params.B.memref()}>
        %C = memref.alloc() : memref<{self.params.C.memref()}>
        %C_ref = memref.alloc() : memref<{self.params.C.memref()}>
        {self.cpu_code.get_cpu_initialization_code()}
        // intialize matrix C and C_ref ; C[i, j] = 0. Since all values are 0, no need to account for batch
        %c0_{self.params.C.type} = arith.constant {C_zero} : {self.params.C.type}
        scf.for %i = %c0 to %c{self.params.C.x} step %c1 {{
            scf.for %j = %c0 to %c{self.params.C.y} step %c1 {{
                memref.store %c0_{self.params.C.type}, %C[%i, %j] : memref<{self.params.C.memref()}>
                memref.store %c0_{self.params.C.type}, %C_ref[%i, %j] : memref<{self.params.C.memref()}>
            }}
        }}

        // run GPU
        %2 = call @test(%A, %B, %C) : (memref<{self.params.A.memref()}>, memref<{self.params.B.memref()}>, memref<{self.params.C.memref()}>) -> memref<{self.params.C.memref()}>
        // run CPU
        {self.cpu_code.get_cpu_validation_code()}
        memref.dealloc %A : memref<{self.params.A.memref()}>
        memref.dealloc %B : memref<{self.params.B.memref()}>
        memref.dealloc %C : memref<{self.params.C.memref()}>
        memref.dealloc %C_ref : memref<{self.params.C.memref()}>
        return
    }}"""
        return template

    def generate_mlir(self, output_dir):
        template = f"""module @gemm attributes {{gpu.container_module}} {{
    {self.gen_test_driver()}

    {self.gen_gpu_module()}

    {self.get_cpu_reference()}

    {self.gen_main()}
    func.func private @printAllclose{self.params.C.runtime_call_suffix}(memref<*x{self.params.C.type}>, memref<*x{self.params.C.type}>) attributes {{llvm.emit_c_interface}}
    func.func private @fillResource1DRandom{self.params.A.runtime_call_suffix}(memref<*x{self.params.A.type}>, f32, f32, i1) attributes {{llvm.emit_c_interface}}
}}
    """
        with open(f"{os.path.join(output_dir, utils.get_filename(self.params.input_params))}.mlir", 'w') as file:
            file.write(template)
        return template

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', '-t', help="path to a csv file with test cases", type=str, default="input_shapes.csv")
    parser.add_argument('--code_version', '-c', help="code version to generate", type=str, default="baseline")
    parser.add_argument('--print_debug', '-d', help="print debug info for GPU module (e.g., tile sizes) per test", type=int, default=0)
    parser.add_argument('--validate', '-v', help="if validate=1, then CPU will also perform GEMM and compare the result with GPU's result", type=int, default=0)
    parser.add_argument('--default_tests', '-l', help="if default_tests=1, additionally generates default cases for 4kx4k and 1kx1k GEMMs", type=int, default=0)
    parser.add_argument('--output_gemm_dir', '-o', help="name of the output folder, default 'generated-gemm'.", type=str, default="generated-gemm")
    args = parser.parse_args()
    # print(parser.format_help())
    output_dir = f"{args.output_gemm_dir}/{args.code_version}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for filename in glob.glob(output_dir+"/*"):
          os.remove(filename)
    # A: (mxk)
    # B: (kxn)
    # C: (mxn)

    if args.default_tests :
        # Hardcoded test 4k
        input_matrices = {
            "M": 4096,
            "K": 4096,
            "N": 4096,
            "wgm": 256,
            "wgn": 256,
            "sgm": 32,
            "sgn": 64,
            "sgk": 32,
            "type_AB" : "f16",
            "type_C" : "f32",
            "batchSize": 1
        }
        GEMM_test(input_matrices, args).generate_mlir(output_dir)
        # Hardcoded test 1k
        input_matrices = {
            "M": 1024,
            "K": 1024,
            "N": 1024,
            "wgm": 16,
            "wgn": 32,
            "sgm": 16,
            "sgn": 16,
            "sgk": 32,
            "type_AB" : "f16",
            "type_C" : "f32",
            "batchSize": 7
        }
        GEMM_test(input_matrices, args).generate_mlir(output_dir)

    if args.test_csv:
        skipped = []
        shapes = pd.read_csv(args.test_csv, comment='#')
        for index, gemm_shapes in shapes.iterrows():
            # Skip large GEMMs when validating on CPU as it takes minutes even for a 4k GEMM
            if args.validate and (gemm_shapes['M'] > 5000 or gemm_shapes['N'] > 5000 or gemm_shapes['K'] > 5000):
                skipped.append(f"({gemm_shapes['M']},{gemm_shapes['K']},{gemm_shapes['N']})")
                continue
            input_matrices = {
                "M": gemm_shapes['M'],
                "K": gemm_shapes['K'],
                "N": gemm_shapes['N'],
                "wgm": gemm_shapes['wgm'],
                "wgn": gemm_shapes['wgn'],
                "sgm": gemm_shapes['sgm'],
                "sgn": gemm_shapes['sgn'],
                "sgk": gemm_shapes['sgk'],
                "type_AB" : "f16",#gemm_shapes['dtype'],
                "type_C" : "f32",#gemm_shapes['dtype'],
                "batchSize": gemm_shapes['BatchSize']
            }
            try:
                GEMM_test(input_matrices, args).generate_mlir(output_dir)
            except Exception as e:
                print(f"Error when generating testcase {utils.get_filename(input_matrices)}:")
                print(f"  {e}")
        if skipped:
            print(f"Skipped {len(skipped)} tests, (M,K,N): {skipped}")
