# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from numba.extending import overload_classmethod, register_model, typeof_impl
from numba.core import errors
from numba.core.types.npytypes import Array
from numba.core.datamodel.models import ArrayModel
from numba.np import numpy_support
from numba.np.arrayobj import intrin_alloc
from numba.core.imputils import lower_cast


class FixedArray(Array):
    """
    Type class for Numpy arrays.
    """

    def __init__(
        self, dtype, ndim, layout, fixed_dims, readonly=False, name=None, aligned=True
    ):
        self.fixed_dims = fixed_dims
        super(FixedArray, self).__init__(
            dtype, ndim, layout, readonly=readonly, name=name, aligned=aligned
        )

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        return FixedArray(
            dtype=dtype,
            ndim=ndim,
            layout=layout,
            fixed_dims=(None,) * ndim,
            readonly=readonly,
            aligned=self.aligned,
        )

    @property
    def key(self):
        return super().key + (self.fixed_dims,)


register_model(FixedArray)(ArrayModel)


@lower_cast(FixedArray, Array)
def array_to_array(context, builder, fromty, toty, val):
    return val


@lower_cast(Array, FixedArray)
def array_to_array(context, builder, fromty, toty, val):
    return val


@lower_cast(FixedArray, FixedArray)
def array_to_array(context, builder, fromty, toty, val):
    return val


@overload_classmethod(FixedArray, "_allocate")
def _ol_array_allocate(cls, allocsize, align):
    """Implements a Numba-only default target (cpu) classmethod on the array type."""

    def impl(cls, allocsize, align):
        return intrin_alloc(allocsize, align)

    return impl


def get_fixed_dims(shape):
    return tuple(d if (d == 0 or d == 1) else None for d in shape)


@typeof_impl.register(np.ndarray)
def _typeof_ndarray(val, c):
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except errors.NumbaNotImplementedError:
        raise errors.NumbaValueError(f"Unsupported array dtype: {val.dtype}")
    layout = numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    fixed_dims = get_fixed_dims(val.shape)
    return FixedArray(dtype, val.ndim, layout, fixed_dims=fixed_dims, readonly=readonly)
