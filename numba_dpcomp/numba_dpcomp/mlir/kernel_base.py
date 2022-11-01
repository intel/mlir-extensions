# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy


def _raise_error(desc):
    raise ValueError(desc)


def _process_dims(dims):
    if isinstance(dims, int):
        return (dims,)
    elif isinstance(dims, (list, tuple)):
        n = len(dims)
        if n > 3:
            _raise_error(f"Invalid dimentions count: {n}")
        return tuple(dims)
    else:
        _raise_error(f"Invalid dimentions type: {type(dims)}")


class KernelBase:
    def __init__(self, func):
        self.global_size = ()
        self.local_size = ()
        self.py_func = func

    def copy(self):
        return copy.copy(self)

    def configure(self, global_size, local_size):
        global_dim_count = len(global_size)
        local_dim_count = len(local_size)
        assert local_dim_count <= global_dim_count
        if local_dim_count != 0 and local_dim_count < global_dim_count:
            local_size = tuple(
                local_size[i] if i < local_dim_count else 1
                for i in range(global_dim_count)
            )
        ret = self.copy()
        ret.global_size = tuple(global_size)
        ret.local_size = tuple(local_size)
        return ret

    def check_call_args(self, args, kwargs):
        if kwargs:
            _raise_error("kwargs not supported")

    def __getitem__(self, args):
        nargs = len(args)
        if nargs < 1 or nargs > 2:
            _raise_error(f"Invalid kernel arguments count: {nargs}")

        gs = _process_dims(args[0])
        ls = _process_dims(args[1]) if nargs > 1 else ()
        return self.configure(gs, ls)
