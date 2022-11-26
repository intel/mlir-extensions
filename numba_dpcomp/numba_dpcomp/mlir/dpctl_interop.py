# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    import dpctl
    from dpctl.tensor import usm_ndarray

    _is_dpctl_available = True
except ImportError:
    _is_dpctl_available = False

if _is_dpctl_available:
    import numba
    import numpy as np
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils, config, types, typing

    from numba.core.pythonapi import box, unbox, NativeValue

    from numba.extending import register_model, typeof_impl
    from numba.np import numpy_support

    from numba.core.datamodel.models import StructModel

    from . import array_type

    def _get_filter_string(array):
        if isinstance(array, usm_ndarray):
            return array.device.sycl_device.filter_string

        return None

    class USMNdArrayBaseType(array_type.FixedArray):
        """
        Type class for DPPY arrays.
        """

        def __init__(
            self,
            dtype,
            ndim,
            layout,
            fixed_dims,
            readonly=False,
            name=None,
            aligned=True,
            filter_string=None,
        ):
            super(USMNdArrayBaseType, self).__init__(
                dtype,
                ndim,
                layout,
                fixed_dims,
                readonly=readonly,
                name=name,
                aligned=aligned,
            )

            self.filter_string = filter_string

        @property
        def key(self):
            return super().key + (self.filter_string,)

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
            fixed_dims,
            readonly=False,
            name=None,
            aligned=True,
            filter_string=None,
        ):
            self.usm_type = usm_type
            # This name defines how this type will be shown in Numba's type dumps.
            name = "USM:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
            super(USMNdArrayType, self).__init__(
                dtype,
                ndim,
                layout,
                fixed_dims,
                readonly=readonly,
                name=name,
                filter_string=filter_string,
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
            return USMNdArrayType(
                dtype=dtype,
                ndim=ndim,
                layout=layout,
                usm_type=self.usm_type,
                fixed_dims=(None,) * ndim,
                readonly=readonly,
                aligned=self.aligned,
                filter_string=self.filter_string,
            )

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
        layout = "C"  # TODO: infer layout
        readonly = False
        filter_string = _get_filter_string(val)
        assert filter_string is not None
        fixed_dims = array_type.get_fixed_dims(val.shape)
        return USMNdArrayType(
            dtype,
            val.ndim,
            layout,
            val.usm_type,
            fixed_dims,
            readonly=readonly,
            filter_string=filter_string,
        )

    def adapt_sycl_array_from_python(pyapi, ary, ptr):
        assert pyapi.context.enable_nrt
        fnty = ir.FunctionType(ir.IntType(32), [pyapi.pyobj, pyapi.voidptr])
        fn = pyapi._get_function(fnty, name="dpcompUnboxSyclInterface")
        fn.args[0].add_attribute("nocapture")
        fn.args[1].add_attribute("nocapture")
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

    def check_usm_ndarray_args(args):
        devs = set(s for s in map(_get_filter_string, args) if s is not None)
        if len(devs) > 1:
            dev_names = ", ".join(devs)
            err_str = f"usm_ndarray arguments have incompatibe devices: {dev_names}"
            raise ValueError(err_str)

    def get_default_device_name():
        return dpctl.select_default_device().filter_string

else:  # _is_dpctl_available

    USMNdArrayType = None  # dummy

    def check_usm_ndarray_args(args):
        # dpctl is not loaded, nothing to do
        pass

    def get_default_device_name():
        # TODO: deprecated
        return "level_zero:gpu:0"
