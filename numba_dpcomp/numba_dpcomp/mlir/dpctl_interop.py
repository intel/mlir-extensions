# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


try:
    import dpctl.tensor.numpy_usm_shared as nus
    from dpctl.tensor.numpy_usm_shared import class_list, functions_list, ndarray
    from dpctl.tensor import usm_ndarray
    import dpctl
    from dpctl.memory import MemoryUSMShared
    _is_dpctl_available = True
except ImportError:
    _is_dpctl_available = False

if _is_dpctl_available:
    import builtins
    import functools
    import importlib
    import inspect
    import sys
    from ctypes.util import find_library
    from inspect import getmembers, isbuiltin, isclass, isfunction
    from numbers import Number
    from types import BuiltinFunctionType as bftype
    from types import FunctionType as ftype



    import llvmlite.binding as llb
    import llvmlite.llvmpy.core as lc
    import numba
    import numpy as np
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils, config, types, typing
    from numba.core.datamodel.registry import (
        register_default as register_model_default,
    )
    from numba.core.imputils import builtin_registry as lower_registry
    from numba.core.overload_glue import _overload_glue
    from numba.core.pythonapi import box, unbox
    from numba.core.typing.arraydecl import normalize_shape
    from numba.core.typing.npydecl import registry as typing_registry
    from numba.core.typing.templates import (
        AttributeTemplate,
        CallableTemplate,
        bound_function,
    )
    from numba.core.typing.templates import builtin_registry as templates_registry
    from numba.core.typing.templates import signature
    from numba.extending import (
        intrinsic,
        lower_builtin,
        overload_classmethod,
        register_model,
        type_callable,
        typeof_impl,
    )
    from numba.np import numpy_support
    from numba.np.arrayobj import _array_copy

    from numba.core.datamodel.models import StructModel
    from numba.core.types.npytypes import Array

    debug = config.DEBUG


    def dprint(*args):
        if debug:
            print(*args)
            sys.stdout.flush()

    class DPPYArray(Array):
        """
        Type class for DPPY arrays.
        """

        def __init__(
            self,
            dtype,
            ndim,
            layout,
            readonly=False,
            name=None,
            aligned=True,
            addrspace=None,
        ):
            self.addrspace = addrspace
            super(DPPYArray, self).__init__(
                dtype,
                ndim,
                layout,
                readonly=readonly,
                name=name,
                aligned=aligned,
            )

        def copy(
            self, dtype=None, ndim=None, layout=None, readonly=None, addrspace=None
        ):
            if dtype is None:
                dtype = self.dtype
            if ndim is None:
                ndim = self.ndim
            if layout is None:
                layout = self.layout
            if readonly is None:
                readonly = not self.mutable
            if addrspace is None:
                addrspace = self.addrspace
            return DPPYArray(
                dtype=dtype,
                ndim=ndim,
                layout=layout,
                readonly=readonly,
                aligned=self.aligned,
                addrspace=addrspace,
            )

        @property
        def key(self):
            return (
                self.dtype,
                self.ndim,
                self.layout,
                self.mutable,
                self.aligned,
                self.addrspace,
            )

        @property
        def box_type(self):
            return np.ndarray

        def is_precise(self):
            return self.dtype.is_precise()


    class DPPYArrayModel(StructModel):
        def __init__(self, dmm, fe_type):
            ndim = fe_type.ndim
            members = [
                (
                    "meminfo",
                    types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
                ),
                (
                    "parent",
                    types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
                ),
                ("nitems", types.intp),
                ("itemsize", types.intp),
                (
                    "data",
                    types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
                ),
                ("shape", types.UniTuple(types.intp, ndim)),
                ("strides", types.UniTuple(types.intp, ndim)),
            ]
            super(DPPYArrayModel, self).__init__(dmm, fe_type, members)

    class USMNdArrayType(DPPYArray):
        """
        USMNdArrayType(dtype, ndim, layout, usm_type,
                        readonly=False, name=None,
                        aligned=True, addrspace=None)
        creates Numba type to represent ``dpctl.tensor.usm_ndarray``.
        """

        def __init__(
            self,
            dtype,
            ndim,
            layout,
            usm_type,
            readonly=False,
            name=None,
            aligned=True,
            addrspace=None,
        ):
            self.usm_type = usm_type
            # This name defines how this type will be shown in Numba's type dumps.
            name = "USM:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
            super(USMNdArrayType, self).__init__(
                dtype,
                ndim,
                layout,
                readonly=readonly,
                name=name,
                addrspace=addrspace,
            )

        def copy(self, *args, **kwargs):
            return super(USMNdArrayType, self).copy(*args, **kwargs)


    # This tells Numba to use the DPPYArray data layout for object of type USMNdArrayType.
    register_model(USMNdArrayType)(DPPYArrayModel)
    # dppy_target.spirv_data_model_manager.register(USMNdArrayType, DPPYArrayModel)


    @typeof_impl.register(usm_ndarray)
    def typeof_usm_ndarray(val, c):
        """
        This function creates the Numba type (USMNdArrayType) when a usm_ndarray is passed.
        """
        try:
            dtype = numpy_support.from_dtype(val.dtype)
        except NotImplementedError:
            raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
        layout = "C"
        readonly = False
        return USMNdArrayType(
            dtype,
            val.ndim,
            layout,
            val.usm_type,
            readonly=readonly,
            # addrspace=address_space.GLOBAL,
            addrspace=None,
        )

    class UsmSharedArrayType(DPPYArray):
        """Creates a Numba type for Numpy arrays that are stored in USM shared
        memory.  We inherit from Numba's existing Numpy array type but overload
        how this type is printed during dumping of typing information and we
        implement the special __array_ufunc__ function to determine who this
        type gets combined with scalars and regular Numpy types.
        We re-use Numpy functions as well but those are going to return Numpy
        arrays allocated in USM and we use the overloaded copy function to
        convert such USM-backed Numpy arrays into typed USM arrays."""

        def __init__(
            self,
            dtype,
            ndim,
            layout,
            readonly=False,
            name=None,
            aligned=True,
            addrspace=None,
        ):
            # This name defines how this type will be shown in Numba's type dumps.
            name = "UsmArray:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
            super(UsmSharedArrayType, self).__init__(
                dtype,
                ndim,
                layout,
                # py_type=ndarray,
                readonly=readonly,
                name=name,
                addrspace=addrspace,
            )

        def copy(self, *args, **kwargs):
            retty = super(UsmSharedArrayType, self).copy(*args, **kwargs)
            if isinstance(retty, types.Array):
                return UsmSharedArrayType(
                    dtype=retty.dtype, ndim=retty.ndim, layout=retty.layout
                )
            else:
                return retty

        # Tell Numba typing how to combine UsmSharedArrayType with other ndarray types.
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            if method == "__call__":
                for inp in inputs:
                    if not isinstance(
                        inp, (UsmSharedArrayType, types.Array, types.Number)
                    ):
                        return None

                return UsmSharedArrayType
            else:
                return None

        @property
        def box_type(self):
            return ndarray

    # This tells Numba how to create a UsmSharedArrayType when a usmarray is passed
    # into a njit function.
    @typeof_impl.register(ndarray)
    def typeof_ta_ndarray(val, c):
        try:
            dtype = numpy_support.from_dtype(val.dtype)
        except NotImplementedError:
            raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
        layout = numpy_support.map_layout(val)
        readonly = not val.flags.writeable
        return UsmSharedArrayType(dtype, val.ndim, layout, readonly=readonly)

    # This tells Numba to use the default Numpy ndarray data layout for
    # object of type UsmArray.
    # register_model(UsmSharedArrayType)(DPPYArrayModel)
    register_model(UsmSharedArrayType)(numba.core.datamodel.models.ArrayModel)
    # dppy_target.spirv_data_model_manager.register(UsmSharedArrayType, DPPYArrayModel)
    # dppy_target.spirv_data_model_manager.register(
    #     UsmSharedArrayType, numba.core.datamodel.models.ArrayModel
    # )

    # This tells Numba how to convert from its native representation
    # of a UsmArray in a njit function back to a Python UsmArray.
    @box(UsmSharedArrayType)
    def box_array(typ, val, c):
        nativearycls = c.context.make_array(typ)
        nativeary = nativearycls(c.context, c.builder, value=val)
        if c.context.enable_nrt:
            np_dtype = numpy_support.as_dtype(typ.dtype)
            dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
            # Steals NRT ref
            newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
            return newary
        else:
            parent = nativeary.parent
            c.pyapi.incref(parent)
            return parent


    @unbox(UsmSharedArrayType)
    def unbox_array1(typ, obj, c):
        print('unbox_array UsmSharedArrayType',flush=True)
        assert False

    @unbox(USMNdArrayType)
    def unbox_array2(typ, obj, c):
        print('unbox_array USMNdArrayType',flush=True)
        assert False

    _registered = False

    def numba_register():
        return
        global _registered
        if _registered:
            return

        _registered = True
        # numba_register_typing()
        # numba_register_lower_builtin()


else: # _is_dpctl_available
    def numba_register():
        pass
