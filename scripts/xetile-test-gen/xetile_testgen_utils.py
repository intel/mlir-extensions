import math

class Grid:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}x{self.y})"

    def num_elements(self):
        return self.x*self.y

class Tile(Grid):
    def __init__(self, x=0, y=0, batch_dim=1, type=""):
        super().__init__(x*batch_dim,y)
        self.type_str = ''.join(filter(lambda x: x.isalpha(), type))
        self.type_nbit = ''.join(filter(lambda x: x.isdigit(), type))
        self.type = type
        self.runtime_call_suffix = self.type_str.upper() + self.type_nbit
        self.batch_size = batch_dim

    def memref(self):
        return f"{self.x}x{self.y}x{self.type}"

    def get_x_dim_for_one_in_batch(self):
        return self.x // self.batch_size

    def flattened_memref(self):
        return f"{self.x*self.y}x{self.type}"

class WGLevel():
    def __init__(self, input_matrices):
        block_size_m, block_size_n, block_count_m, block_count_n = self.get_block_sizes_counts(input_matrices)
        k_dim = input_matrices['K']
        self.layout = Grid(x=block_count_m, y=block_count_n)
        self.A_tile = Tile(x=block_size_m, y=k_dim, type=input_matrices["type_AB"])
        self.B_tile = Tile(x=k_dim, y=block_size_n, type=input_matrices["type_AB"])
        self.C_tile = Tile(x=block_size_m, y=block_size_n, type=input_matrices["type_C"])

    def __str__(self):
        depth = 3
        return f"Blocks (WG layout): {self.layout}" + f"\n{' ' * depth}" + \
               f"Tile A: {self.A_tile}" + f"\n{' ' * depth}" + \
               f"Tile B: {self.B_tile}" + f"\n{' ' * depth}" + \
               f"Tile C: {self.C_tile}"

    def get_block_sizes_counts(self, input_matrices : dict):
      block_size_m, block_size_n = [input_matrices["wgm"], input_matrices["wgn"]]
      block_count_m, block_count_n = [math.ceil(input_matrices["M"] / block_size_m), math.ceil(input_matrices["N"] / block_size_n)]
      return block_size_m, block_size_n, block_count_m, block_count_n

class SGLevel():
    def __init__(self, input_matrices, parent_wg_level):
        self.parent_wg = parent_wg_level
        thread_size_m, thread_size_n, thread_size_k, thread_count_m, thread_count_n = self.get_thread_size_count(input_matrices)
        self.layout = Grid(x=thread_count_m, y=thread_count_n)
        self.A_tile = Tile(x=thread_size_m, y=thread_size_k, type=input_matrices["type_AB"])
        self.B_tile = Tile(x=thread_size_k, y=thread_size_n, type=input_matrices["type_AB"])
        self.C_tile = Tile(x=thread_size_m, y=thread_size_n, type=input_matrices["type_C"])
        self.k_loop_step = thread_size_k

    def __str__(self):
        depth = 3
        return f"Threads (SG layout): {self.layout}" + f"\n{' ' * depth}" + \
               f"Tile A: {self.A_tile}" + f"\n{' ' * depth}" + \
               f"Tile B: {self.B_tile}" + f"\n{' ' * depth}" + \
               f"Tile C: {self.C_tile}"

    def get_thread_size_count(self, input_matrices : dict):
        block_size_m, block_size_n, _, _ = self.parent_wg.get_block_sizes_counts(input_matrices)
        thread_size_m, thread_size_n, thread_size_k = [input_matrices["sgm"], input_matrices["sgn"], input_matrices["sgk"]]
        thread_count_m, thread_count_n = [math.ceil(block_size_m / thread_size_m),
                                          math.ceil(block_size_n / thread_size_n)]
        return thread_size_m, thread_size_n, thread_size_k, thread_count_m, thread_count_n

def generate_constants(list_of_constants: list, prefix="", depth=2, cst_type="index"):
    """
    Generates constants of the specified type from the given list.
    If empty list is passed, returns empty string.
      list_of_constants : constants to generate, should be all of one type
      prefix : constant name prefix (%c_prefix_...)
      depth : number of double spaces to generate pretty code
      cst_type : type of constants ("index", "f16", ...)
    """
    if not list_of_constants:
        return ""
    const_definitions = ""
    for idx, cst in enumerate(list_of_constants):
        shift_text_by = ' ' * depth * 2 * (idx>0)
        if cst_type.startswith('f'):
            const_definitions += f"{shift_text_by}%c{prefix}{int(cst)} = arith.constant {cst:.1f} : {cst_type}\n"
        else:
            const_definitions += f"{shift_text_by}%c{prefix}{cst} = arith.constant {cst} : {cst_type}\n"
    return const_definitions[:-1]

def within_limits(tile, prefetch_shape_limits):
    return (tile.x >= prefetch_shape_limits['min_x'] and tile.x <= prefetch_shape_limits['max_x']) and \
           (tile.y >= prefetch_shape_limits['min_y'] and tile.y <= prefetch_shape_limits['max_y'])

def get_prefetch_size_and_layout_in_SG_tile(SG_layout, SG_tile, prefetch_shape_limits, wg_slice_direction = "y"):
    # Get the number of SGs that will collaboratively prefetch a SG tile, wg_slice_direction y means A matrix, because its WG slice moves along y (→).
    num_collaborators = SG_layout.y if wg_slice_direction == "y" else SG_layout.x
    prefetch_tile = Tile()
    prefetch_layout_in_SG_tile = Grid()
    # First we try to get a prefetch tile with the largest Y dimension
    prefetch_tile.y = prefetch_shape_limits['max_y']
    prefetch_layout_in_SG_tile.y = SG_tile.y / prefetch_tile.y
    # Largest Y dimension might have been too large and now we have <1 tile along Y,
    # correct it by reducing Y of the prefetch tile until we have at least one tile along Y dimension
    error_msg_failed_size_selection = f"Couldn't find a proper prefetch shape for SG layout:{SG_layout}, for " +\
                                  f"matrix: {'A' if wg_slice_direction == 'y' else 'B'}, SG tile:{SG_tile}"
    while prefetch_layout_in_SG_tile.y < 1:
      prefetch_tile.y /= 2
      prefetch_layout_in_SG_tile.y = SG_tile.y / prefetch_tile.y
      # We can only reduce Y dimension up to a limit
      if prefetch_tile.y < prefetch_shape_limits['min_y']:
          raise Exception(error_msg_failed_size_selection)

    # We now have the largest possible Y dimension (for a given SG Tile shape), proceed to X dimension.
    # We have fixed num_collaborators and we already know the Y layout for prefetch tiles, so X layout is easy to calculate.
    prefetch_layout_in_SG_tile.x = num_collaborators / prefetch_layout_in_SG_tile.y
    prefetch_tile.x = SG_tile.x / prefetch_layout_in_SG_tile.x
    # Since we have constant "area" of the prefetch and we initially tried to make Y dimension of our prefetch tile as large as possible,
    # the resulting X dimension of our prefetch tile may turn out too small, correct it by reducing Y dimension and correspondingly increasing X dimension
    while not(within_limits(prefetch_tile, prefetch_shape_limits) and prefetch_layout_in_SG_tile.num_elements() == num_collaborators):
        prefetch_tile.y /= 2
        prefetch_layout_in_SG_tile.y = SG_tile.y / prefetch_tile.y
        prefetch_tile.x *= 2
        prefetch_layout_in_SG_tile.x = SG_tile.x / prefetch_tile.x
        # We can only reduce Y dimension down to a limit
        if prefetch_tile.y < prefetch_shape_limits['min_y']:
            raise Exception(error_msg_failed_size_selection)

    # We might have leftover for imperfect shapes, ignore it for now by rounding down with int()
    prefetch_tile = Tile(x=int(prefetch_tile.x), y=int(prefetch_tile.y), type=SG_tile.type)
    prefetch_layout_in_SG_tile = Grid(x=int(prefetch_layout_in_SG_tile.x), y=int(prefetch_layout_in_SG_tile.y))
    return prefetch_tile, prefetch_layout_in_SG_tile

def get_filename(input_params):
    wg_sg_suffix = f"wgm{input_params['wgm']}_wgn{input_params['wgn']}_sgm{input_params['sgm']}_sgn{input_params['sgn']}_sgk{input_params['sgk']}"
    matrix_shapes = f"{input_params['M']}x{input_params['K']}x{input_params['N']}"
    out_filename = f"sg_gemm_{input_params['batchSize']}_{matrix_shapes}_{wg_sg_suffix}"
    return out_filename

class TestParams:
    def __init__(self, input_params, test_args):
        self.test_args = test_args
        self.input_params = input_params
        self.WG = WGLevel(input_params)
        self.SG = SGLevel(input_params, self.WG)
        self.k_loop_step = self.SG.k_loop_step

        self.A = Tile(input_params["M"], input_params["K"], batch_dim=input_params["batchSize"], type=input_params["type_AB"])
        self.B = Tile(input_params["K"], input_params["N"], batch_dim=input_params["batchSize"], type=input_params["type_AB"])
        self.C = Tile(input_params["M"], input_params["N"], batch_dim=input_params["batchSize"], type=input_params["type_C"])
        self.batch_size = input_params["batchSize"]

        self.block_size_m, self.block_size_n, self.block_count_m, self.block_count_n = self.WG.get_block_sizes_counts(self.input_params)
        self.thread_size_m, self.thread_size_n, self.thread_size_k, self.thread_count_m, self.thread_count_n = self.SG.get_thread_size_count(self.input_params)
        self.constants = list(set([
            0,1,self.batch_size,
            self.C.get_x_dim_for_one_in_batch(), self.B.get_x_dim_for_one_in_batch(),
            self.A.x,self.A.y,
            self.B.y,self.B.y,
            self.C.x,self.C.y,
            self.block_size_m, self.block_size_n, self.block_count_m, self.block_count_n,
            self.thread_size_m, self.thread_size_n, self.thread_size_k, self.thread_count_m, self.thread_count_n,
            self.WG.C_tile.x, self.WG.C_tile.y,
            self.SG.layout.x, self.SG.layout.y,
            self.SG.C_tile.x, self.SG.C_tile.y,
            self.SG.A_tile.x, self.SG.A_tile.y,
            self.SG.B_tile.x, self.SG.B_tile.y
        ]))
