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
    from dpctl.tensor import usm_ndarray

    _is_dpctl_available = True
except ImportError:
    _is_dpctl_available = False

if _is_dpctl_available:
    import llvmlite.llvmpy.core as lc
    import numba
    import numpy as np
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils, config, types, typing

    from numba.core.pythonapi import box, unbox, NativeValue

    from numba.extending import register_model, typeof_impl
    from numba.np import numpy_support

    from numba.core.datamodel.models import StructModel
    from numba.core.types.npytypes import Array

    class USMNdArrayBaseType(Array):
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
        ):
            super(USMNdArrayBaseType, self).__init__(
                dtype,
                ndim,
                layout,
                readonly=readonly,
                name=name,
                aligned=aligned,
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
            return USMNdArrayBaseType(
                dtype=dtype,
                ndim=ndim,
                layout=layout,
                readonly=readonly,
                aligned=self.aligned,
            )

        @property
        def key(self):
            return (
                self.dtype,
                self.ndim,
                self.layout,
                self.mutable,
                self.aligned,
            )

        @property
        def box_type(self):
            return np.ndarray

        def is_precise(self):
            return self.dtype.is_precise()

    class USMNdArrayModel(StructModel):
        def __init__(self, dmm, fe_type):
            ndim = fe_type.ndim
            members = [
                ("meminfo", types.MemInfoPointer(fe_type.dtype)),
                ("parent", types.pyobject),
                ("nitems", types.intp),
                ("itemsize", types.intp),
                ("data", types.CPointer(fe_type.dtype)),
                ("shape", types.UniTuple(types.intp, ndim)),
                ("strides", types.UniTuple(types.intp, ndim)),
            ]
            super(USMNdArrayModel, self).__init__(dmm, fe_type, members)

    class USMNdArrayType(USMNdArrayBaseType):
        """
        USMNdArrayType(dtype, ndim, layout, usm_type,
                        readonly=False, name=None,
                        aligned=True)
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
            )

        def copy(self, *args, **kwargs):
            return super(USMNdArrayType, self).copy(*args, **kwargs)

    register_model(USMNdArrayType)(USMNdArrayModel)

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
        )

    def adapt_sycl_array_from_python(pyapi, ary, ptr):
        assert pyapi.context.enable_nrt
        fnty = lc.Type.function(lc.Type.int(), [pyapi.pyobj, pyapi.voidptr])
        fn = pyapi._get_function(fnty, name="dpcompUnboxSyclInterface")
        fn.args[0].add_attribute(lc.ATTR_NO_CAPTURE)
        fn.args[1].add_attribute(lc.ATTR_NO_CAPTURE)
        return pyapi.builder.call(fn, (ary, ptr))

    @unbox(USMNdArrayType)
    def unbox_array(typ, obj, c):
        nativearycls = c.context.make_array(typ)
        nativeary = nativearycls(c.context, c.builder)
        aryptr = nativeary._getpointer()

        ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
        errcode = adapt_sycl_array_from_python(c.pyapi, obj, ptr)
        failed = cgutils.is_not_null(c.builder, errcode)

        # Handle error
        with c.builder.if_then(failed, likely=False):
            c.pyapi.err_set_string(
                "PyExc_TypeError",
                "can't unbox array from PyObject into "
                "native value.  The object maybe of a "
                "different type",
            )
        return NativeValue(c.builder.load(aryptr), is_error=failed)

    def _get_filter_string(array):
        if isinstance(array, usm_ndarray):
            return array.device.sycl_device.filter_string

        return None

    def check_usm_ndarray_args(args):
        devs = set(s for s in map(_get_filter_string, args) if s is not None)
        if len(devs) > 1:
            dev_names = ", ".join(devs)
            err_str = f"usm_ndarray arguments have incompatibe devices: {dev_names}"
            raise ValueError(err_str)

else:  # _is_dpctl_available

    USMNdArrayType = None  # dummy

    def check_usm_ndarray_args(args):
        # dpctl is not loaded, nothing to do
        pass
