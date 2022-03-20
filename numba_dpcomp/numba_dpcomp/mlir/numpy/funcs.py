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

from ..linalg_builder import (
    FuncRegistry,
    is_literal,
    broadcast_type_arrays,
    get_array_type,
    get_val_type,
    eltwise,
    convert_array,
    asarray,
    is_int,
    is_float,
    dtype_str,
    DYNAMIC_DIM,
)
from ..func_registry import add_func

import numpy
import math
from numba import prange

from numba.core import types
from numba.core.typing.templates import ConcreteTemplate, signature, infer_global

add_func(prange, "numba.prange")

registry = FuncRegistry()


def register_func(name, orig_func=None):
    global registry
    return registry.register_func(name, orig_func)


def register_attr(name):
    global registry
    return registry.register_attr(name)


def promote_int(t, b):
    if is_int(t, b):
        return b.int64
    return t


def _linalg_index(dim):
    pass


@infer_global(_linalg_index)
class _LinalgIndexId(ConcreteTemplate):
    cases = [signature(types.int64, types.int64)]


@register_func("_linalg_index", _linalg_index)
def linalg_index_impl(builder, dim):
    if isinstance(dim, int):
        return builder.linalg_index(dim)


def _fix_axis(axis, num_dims):
    if axis < 0:
        axis = axis + num_dims
    assert axis >= 0 and axis < num_dims
    return axis


def _array_reduce(builder, arg, axis, body, get_init_value):
    if axis is None:
        shape = arg.shape
        num_dims = len(shape)
        iterators = ["reduction" for _ in range(num_dims)]
        dims = ",".join(["d%s" % i for i in range(num_dims)])
        expr1 = f"({dims}) -> ({dims})"
        expr2 = f"({dims}) -> (0)"
        maps = [expr1, expr2]
        res_type = promote_int(arg.dtype, builder)
        init = builder.from_elements(get_init_value(builder, res_type), res_type)
        res = builder.linalg_generic(arg, init, iterators, maps, body)
        return builder.extract(res, 0)
    elif isinstance(axis, int):
        shape = arg.shape
        num_dims = len(shape)
        axis = _fix_axis(axis, num_dims)
        iterators = [
            ("reduction" if i == axis else "parallel") for i in range(num_dims)
        ]
        dims1 = ",".join(["d%s" % i for i in range(num_dims)])
        dims2 = ",".join(["d%s" % i for i in range(num_dims) if i != axis])
        expr1 = f"({dims1}) -> ({dims1})"
        expr2 = f"({dims1}) -> ({dims2})"
        maps = [expr1, expr2]
        res_shape = tuple(shape[i] for i in range(len(shape)) if i != axis)
        res_type = promote_int(arg.dtype, builder)
        init = builder.init_tensor(
            res_shape, res_type, get_init_value(builder, res_type)
        )
        return builder.linalg_generic(arg, init, iterators, maps, body)


@register_func("array.sum")
@register_func("numpy.sum", numpy.sum)
def sum_impl(builder, arg, axis=None):
    return _array_reduce(builder, arg, axis, lambda a, b: a + b, lambda b, t: 0)


@register_func("numpy.flip", numpy.flip)
def flip_impl(builder, arg, axis=None):
    shape = arg.shape
    num_dims = len(shape)
    if axis is None:
        axis = (True,) * num_dims
    elif isinstance(axis, int):
        axis = _fix_axis(axis, num_dims)
        l = [False] * num_dims
        l[axis] = True
        axis = tuple(l)
    else:
        return

    offsets = [0 if not axis[i] else shape[i] - 1 for i in range(num_dims)]
    strides = [1 if not axis[i] else -1 for i in range(num_dims)]
    return builder.subview(arg, offsets, shape, strides)


def _get_numpy_type(builder, dtype):
    types = [
        (builder.int8, numpy.int8),
        (builder.int16, numpy.int16),
        (builder.int32, numpy.int32),
        (builder.int64, numpy.int64),
        (builder.uint8, numpy.uint8),
        (builder.uint16, numpy.uint16),
        (builder.uint32, numpy.uint32),
        (builder.uint64, numpy.uint64),
        (builder.float32, numpy.float32),
        (builder.float64, numpy.float64),
    ]
    for t, nt in types:
        if t == dtype:
            return nt
    raise ValueError(f"Cannot convert type to numpy: {str(dtype)}")


def _get_max_init_value(builder, dtype):
    if (dtype == builder.float32) or (dtype == builder.float64):
        return -math.inf
    return numpy.iinfo(_get_numpy_type(builder, dtype)).min


@register_func("array.max")
@register_func("numpy.amax", numpy.amax)
def min_impl(builder, arg, axis=None):
    return _array_reduce(
        builder, arg, axis, lambda a, b: a if a > b else b, _get_max_init_value
    )


def _get_min_init_value(builder, dtype):
    if (dtype == builder.float32) or (dtype == builder.float64):
        return math.inf
    return numpy.iinfo(_get_numpy_type(builder, dtype)).max


@register_func("array.min")
@register_func("numpy.amin", numpy.amin)
def min_impl(builder, arg, axis=None):
    return _array_reduce(
        builder, arg, axis, lambda a, b: a if a < b else b, _get_min_init_value
    )


@register_func("array.mean")
@register_func("numpy.mean", numpy.mean)
def mean_impl(builder, arg, axis=None):
    return sum_impl(builder, arg, axis) / size_impl(builder, arg)


def _gen_unary_ops():
    def f64_type(builder, t):
        if is_float(t, builder):
            return t
        return builder.float64

    def bool_type(builder, t):
        return builder.bool

    unary_ops = [
        (register_func("numpy.sqrt", numpy.sqrt), f64_type, lambda a, b: math.sqrt(a)),
        (register_func("numpy.square", numpy.square), None, lambda a, b: a * a),
        (register_func("numpy.log", numpy.log), f64_type, lambda a, b: math.log(a)),
        (register_func("numpy.sin", numpy.sin), f64_type, lambda a, b: math.sin(a)),
        (register_func("numpy.cos", numpy.cos), f64_type, lambda a, b: math.cos(a)),
        (register_func("numpy.exp", numpy.exp), f64_type, lambda a, b: math.exp(a)),
        (register_func("numpy.tanh", numpy.tanh), f64_type, lambda a, b: math.tanh(a)),
        (register_func("numpy.abs", numpy.abs), None, lambda a, b: abs(a)),
        (register_func("numpy.negative", numpy.negative), None, lambda a, b: -a),
        (
            register_func("numpy.logical_not", numpy.logical_not),
            bool_type,
            lambda a, b: not bool(a),
        ),
    ]

    def make_func(init, body):
        def func(builder, arg):
            init_type = None if init is None else init(builder, arg.dtype)
            return eltwise(builder, arg, body, init_type)

        return func

    for reg, f64, body in unary_ops:
        reg(make_func(f64, body))


_gen_unary_ops()
del _gen_unary_ops


def _select_float_type(builder, a, b):
    # TODO: hack for numba
    da = get_array_type(builder, a)
    db = get_array_type(builder, b)
    if da == db:
        return da
    if is_float(da, builder) and not is_float(db, builder):
        return da
    if is_float(db, builder) and not is_float(da, builder):
        return db
    return broadcast_type_arrays(builder, (a, b))


def _gen_binary_ops():
    def bool_type(builder, a, b):
        return builder.bool

    def select_float_type_f64(builder, a, b):
        # TODO: hack for numba
        da = get_array_type(builder, a)
        db = get_array_type(builder, b)
        if da == db and is_float(da, builder):
            return da
        if is_float(da, builder) and not is_float(db, builder):
            return da
        if is_float(db, builder) and not is_float(da, builder):
            return db
        return builder.float64

    binary_ops = [
        (
            register_func("numpy.add", numpy.add),
            _select_float_type,
            lambda a, b, c: a + b,
        ),
        (register_func("operator.add"), _select_float_type, lambda a, b, c: a + b),
        (
            register_func("numpy.subtract", numpy.subtract),
            _select_float_type,
            lambda a, b, c: a - b,
        ),
        (register_func("operator.sub"), _select_float_type, lambda a, b, c: a - b),
        (
            register_func("numpy.multiply", numpy.multiply),
            _select_float_type,
            lambda a, b, c: a * b,
        ),
        (register_func("operator.mul"), _select_float_type, lambda a, b, c: a * b),
        (
            register_func("numpy.true_divide", numpy.true_divide),
            select_float_type_f64,
            lambda a, b, c: a / b,
        ),
        (
            register_func("operator.truediv"),
            select_float_type_f64,
            lambda a, b, c: a / b,
        ),
        (
            register_func("numpy.power", numpy.power),
            _select_float_type,
            lambda a, b, c: a ** b,
        ),
        (register_func("operator.pow"), _select_float_type, lambda a, b, c: a ** b),
        (
            register_func("numpy.arctan2", numpy.arctan2),
            select_float_type_f64,
            lambda a, b, c: math.atan2(a, b),
        ),
        (
            register_func("numpy.minimum", numpy.minimum),
            _select_float_type,
            lambda a, b, c: min(a, b),
        ),
        (
            register_func("numpy.maximum", numpy.maximum),
            _select_float_type,
            lambda a, b, c: max(a, b),
        ),
        (
            register_func("numpy.logical_and", numpy.logical_and),
            bool_type,
            lambda a, b, c: a and b,
        ),
        (register_func("operator.and"), bool_type, lambda a, b, c: a and b),
        (
            register_func("numpy.logical_or", numpy.logical_or),
            bool_type,
            lambda a, b, c: a or b,
        ),
        (register_func("operator.or"), bool_type, lambda a, b, c: a or b),
        (
            register_func("numpy.logical_xor", numpy.logical_xor),
            bool_type,
            lambda a, b, c: bool(a) != bool(b),
        ),
        (register_func("operator.xor"), bool_type, lambda a, b, c: bool(a) != bool(b)),
        (register_func("operator.lt"), bool_type, lambda a, b, c: a < b),
        (register_func("operator.le"), bool_type, lambda a, b, c: a <= b),
        (register_func("operator.gt"), bool_type, lambda a, b, c: a > b),
        (register_func("operator.ge"), bool_type, lambda a, b, c: a >= b),
        (register_func("operator.eq"), bool_type, lambda a, b, c: a == b),
        (register_func("operator.ne"), bool_type, lambda a, b, c: a != b),
    ]

    def make_func(init, body):
        def func(builder, arg1, arg2):
            init_type = None if init is None else init(builder, arg1, arg2)
            return eltwise(builder, (arg1, arg2), body, init_type)

        return func

    for reg, init, body in binary_ops:
        reg(make_func(init, body))


_gen_binary_ops()
del _gen_binary_ops


def _init_impl(builder, shape, dtype, init=None):
    if dtype is None:
        dtype = builder.float64

    try:
        len(shape)  # will raise if not available
    except:
        shape = (shape,)

    if init is None:
        return builder.init_tensor(shape, dtype)
    else:
        init = builder.cast(init, dtype)
        return builder.init_tensor(shape, dtype, init)


@register_func("numpy.empty", numpy.empty)
def empty_impl(builder, shape, dtype=None):
    return _init_impl(builder, shape, dtype)


@register_func("numpy.empty_like", numpy.empty_like)
def empty_like_impl(builder, arr):
    return _init_impl(builder, arr.shape, arr.dtype)


@register_func("numpy.zeros", numpy.zeros)
def zeros_impl(builder, shape, dtype=None):
    return _init_impl(builder, shape, dtype, 0)


@register_func("numpy.zeros_like", numpy.zeros_like)
def zeros_like_impl(builder, arr):
    return _init_impl(builder, arr.shape, arr.dtype, 0)


@register_func("numpy.ones", numpy.ones)
def ones_impl(builder, shape, dtype=None):
    return _init_impl(builder, shape, dtype, 1)


@register_func("numpy.ones_like", numpy.ones_like)
def ones_like_impl(builder, arr):
    return _init_impl(builder, arr.shape, arr.dtype, 1)


@register_func("numpy.eye", numpy.eye)
def eye_impl(builder, N, M=None, k=0, dtype=None):
    if M is None:
        M = N

    if dtype is None:
        dtype = builder.float64

    init = builder.init_tensor((N, M), dtype)
    idx = builder.from_elements(k, builder.int64)

    iterators = ["parallel"] * 2
    maps = ["(d0, d1) -> (0)", "(d0, d1) -> (d0, d1)"]

    def body(a, b):
        i = _linalg_index(0)
        j = _linalg_index(1)
        return 1 if (j - i) == a else 0

    return builder.linalg_generic(idx, init, iterators, maps, body)


def _matmul2d(builder, a, b, shape1, shape2):
    iterators = ["parallel", "parallel", "reduction"]
    expr1 = "(d0,d1,d2) -> (d0,d2)"
    expr2 = "(d0,d1,d2) -> (d2,d1)"
    expr3 = "(d0,d1,d2) -> (d0,d1)"
    maps = [expr1, expr2, expr3]
    res_shape = (shape1[0], shape2[1])
    dtype = broadcast_type_arrays(builder, (a, b))
    init = builder.init_tensor(res_shape, dtype, 0)

    def body(a, b, c):
        return a * b + c

    return builder.linalg_generic((a, b), init, iterators, maps, body)


@register_func("numpy.dot", numpy.dot)
def dot_impl(builder, a, b):
    shape1 = a.shape
    shape2 = b.shape
    if len(shape1) == 1 and len(shape2) == 1:
        iterators = ["reduction"]
        expr1 = "(d0) -> (d0)"
        expr2 = "(d0) -> (0)"
        maps = [expr1, expr1, expr2]
        init = builder.from_elements(0, a.dtype)

        def body(a, b, c):
            return a * b + c

        res = builder.linalg_generic((a, b), init, iterators, maps, body)
        return builder.extract(res, 0)
    if len(shape1) == 2 and len(shape2) == 2:
        return _matmul2d(builder, a, b, shape1, shape2)


@register_func("operator.matmul")
def matmul_impl(builder, a, b):
    shape1 = a.shape
    shape2 = b.shape
    dim1 = len(shape1)
    dim2 = len(shape2)
    if dim1 > 2 or dim2 > 2:
        return

    if dim1 == 1:
        x = shape2[0]
        y = shape1[0]
        dst_shape = (x, y)
        tmp = builder.init_tensor(dst_shape, a.dtype, 1)
        tmp_a = builder.reshape(a, (1, y))
        tmp = builder.insert(tmp_a, tmp, (x - 1, 0), (1, 1))
        a = tmp
    if dim2 == 1:
        x = shape2[0]
        y = shape1[0]
        dst_shape = (x, y)
        tmp = builder.init_tensor(dst_shape, b.dtype, 1)
        tmp_b = builder.reshape(b, (x, 1))
        tmp = builder.insert(tmp_b, tmp, (0, 0), (1, 1))
        b = tmp

    res = _matmul2d(builder, a, b, a.shape, b.shape)

    if dim1 == 1 and dim2 == 1:
        res = builder.extract(res, (shape2[0] - 1, 0))
    elif dim1 == 1:
        res = builder.subview(res, (shape2[0] - 1, 0), (1, shape2[1]), result_rank=1)
    elif dim2 == 1:
        res = builder.subview(res, (0, 0), (shape1[0], 1), result_rank=1)

    return res


@register_func("numpy.where", numpy.where)
def where_impl(builder, cond, x, y):
    cond, x, y = builder.broadcast(cond, x, y, result_type=None)
    x, y = builder.broadcast(x, y, result_type=broadcast_type_arrays(builder, (x, y)))

    def body(c, x, y, r):
        return x if c else y

    return eltwise(builder, (cond, x, y), body)


def _is_scalar(a):
    try:
        return len(a.shape) == 0
    except:
        return True


@register_func("numpy.outer", numpy.outer)
def outer_impl(builder, x, y):
    def flatten(a):
        if _is_scalar(a):
            return builder.from_elements([a], get_val_type(builder, a))

        return flatten_impl(builder, a)

    x = flatten(x)
    y = flatten(y)

    res_type = _select_float_type(builder, x, y)
    init = builder.init_tensor((x.shape[0], y.shape[0]), res_type)

    iterators = ["parallel", "parallel"]
    expr1 = "(d0,d1) -> (d0)"
    expr2 = "(d0,d1) -> (d1)"
    expr3 = "(d0,d1) -> (d0,d1)"
    maps = [expr1, expr2, expr3]

    def body(a, b, c):
        return a * b

    return builder.linalg_generic((x, y), init, iterators, maps, body)


@register_attr("array.shape")
def shape_impl(builder, arg):
    shape = arg.shape
    return tuple(builder.cast(shape[i], builder.int64) for i in range(len(shape)))


@register_attr("array.size")
def size_impl(builder, arg):
    shape = arg.shape
    res = 1
    for i in range(len(shape)):
        res = res * shape[i]
    return builder.cast(res, builder.int64)


@register_func("array.copy")
def shape_impl(builder, arg):
    return builder.force_copy(arg)


@register_attr("array.T")
def transpose_impl(builder, arg):
    shape = arg.shape
    dims = len(shape)
    if dims == 1:
        return arg
    if dims == 2:
        iterators = ["parallel", "parallel"]
        expr1 = "(d0,d1) -> (d0,d1)"
        expr2 = "(d0,d1) -> (d1,d0)"
        maps = [expr1, expr2]
        res_shape = (shape[1], shape[0])
        init = builder.init_tensor(res_shape, arg.dtype)

        def body(a, b):
            return a

        return builder.linalg_generic(arg, init, iterators, maps, body)


@register_attr("array.dtype")
def dtype_impl(builder, arg):
    return arg.dtype


@register_func("array.reshape")
@register_func("numpy.reshape", numpy.reshape)
def reshape_impl(builder, arg, new_shape):
    return builder.reshape(arg, new_shape)


# @register_attr('array.flat')
@register_func("array.flatten")
def flatten_impl(builder, arg):
    size = size_impl(builder, arg)
    return builder.reshape(arg, (size))


@register_func("array.__getitem__")
def getitem_impl(builder, arr, index):
    if index.dtype != builder.bool:
        return

    arr = flatten_impl(builder, arr)
    index = flatten_impl(builder, index)

    def func(a, ind):
        s = a.size
        res = numpy.empty((s,), a.dtype)
        curr = 0
        for i in range(s):
            if ind[i]:
                res[curr] = a[i]
                curr += 1
        return res[0:curr]

    return builder.inline_func(func, arr.type, arr, index)


@register_func("numpy.linalg.eig", numpy.linalg.eig)
def eig_impl(builder, arg):
    shape = arg.shape
    if len(shape) == 2:
        dtype = arg.dtype
        func_name = f"dpcompLinalgEig_{dtype_str(builder, dtype)}"
        size = shape[0]
        vals = builder.init_tensor([size], dtype)
        vecs = builder.init_tensor([size, size], dtype)
        return builder.external_call(func_name, arg, (vals, vecs))


@register_func("numpy.atleast_2d", numpy.atleast_2d)
def atleast2d_impl(builder, arr):
    shape = arr.shape
    dims = len(shape)
    if dims == 0:
        return builder.init_tensor([1, 1], arr.dtype, arr)
    elif dims == 1:
        init = builder.init_tensor([1, shape[0]], arr.dtype)
        iterators = ["parallel", "parallel"]
        expr1 = "(d0,d1) -> (d1)"
        expr2 = "(d0,d1) -> (d0,d1)"
        maps = [expr1, expr2]
        return builder.linalg_generic(arr, init, iterators, maps, lambda a, b: a)
    else:
        return arr


@register_func("numpy.concatenate", numpy.concatenate)
def concat_impl(builder, arrays, axis=0):
    if isinstance(axis, int):
        shapes = [a.shape for a in arrays]
        num_dims = len(shapes[0])
        dtype = broadcast_type_arrays(builder, arrays)
        new_len = sum((s[axis] for s in shapes), 0)
        new_shape = [
            new_len if i == axis else shapes[0][i] for i in range(len(shapes[0]))
        ]
        res = builder.init_tensor(new_shape, dtype)
        offsets = [0] * num_dims
        strides = [1] * num_dims
        for sizes, array in zip(shapes, arrays):
            res = builder.insert(array, res, offsets, strides)
            offsets[axis] += sizes[axis]
        return res


def _cov_get_ddof_func(ddof_is_none):
    if ddof_is_none:

        def ddof_func(bias, ddof):
            if bias:
                return 0
            else:
                return 1

    else:

        def ddof_func(bias, ddof):
            return ddof

    return ddof_func


def _cov_impl_inner(X, ddof):
    # determine the normalization factor
    fact = X.shape[1] - ddof

    # numpy warns if less than 0 and floors at 0
    fact = max(fact, 0.0)

    # _row_wise_average
    m, n = X.shape
    R = numpy.empty((m, 1), dtype=X.dtype)

    for i in prange(m):
        R[i, 0] = numpy.sum(X[i, :]) / n

    # de-mean
    X = X - R

    c = numpy.dot(X, X.T)
    # c = numpy.dot(X, numpy.conj(X.T))
    c = c * numpy.true_divide(1, fact)
    return c


def _prepare_cov_input(builder, m, y, rowvar):
    def get_func():
        if y is None:
            dtype = m.dtype

            def _prepare_cov_input_impl(m, y, rowvar):
                m_arr = numpy.atleast_2d(m)

                if not rowvar:
                    m_arr = m_arr.T

                return m_arr

        else:
            dtype = broadcast_type_arrays(builder, (m, y))

            def _prepare_cov_input_impl(m, y, rowvar):
                m_arr = numpy.atleast_2d(m)
                y_arr = numpy.atleast_2d(y)

                # transpose if asked to and not a (1, n) vector - this looks
                # wrong as you might end up transposing one and not the other,
                # but it's what numpy does
                if not rowvar:
                    if m_arr.shape[0] != 1:
                        m_arr = m_arr.T
                    if y_arr.shape[0] != 1:
                        y_arr = y_arr.T

                m_rows, m_cols = m_arr.shape
                y_rows, y_cols = y_arr.shape

                # if m_cols != y_cols:
                #     raise ValueError("m and y have incompatible dimensions")

                # allocate and fill output array
                return numpy.concatenate((m_arr, y_arr), axis=0)

        return dtype, _prepare_cov_input_impl

    dtype, func = get_func()
    array_type = builder.array_type([DYNAMIC_DIM, DYNAMIC_DIM], dtype)
    if is_int(dtype, builder):
        dtype = builder.float64
    res = builder.inline_func(func, array_type, m, y, rowvar)
    return convert_array(builder, res, dtype)


def _cov_scalar_result_expected(mandatory_input, optional_input):
    opt_is_none = optional_input is None

    if len(mandatory_input.shape) == 1:
        return opt_is_none

    return False


@register_func("numpy.cov", numpy.cov)
def cov_impl(builder, m, y=None, rowvar=True, bias=False, ddof=None):
    m = asarray(builder, m)
    if not y is None:
        y = asarray(builder, y)
    X = _prepare_cov_input(builder, m, y, rowvar)
    ddof = builder.inline_func(
        _cov_get_ddof_func(ddof is None), builder.int64, bias, ddof
    )
    array_type = builder.array_type([DYNAMIC_DIM, DYNAMIC_DIM], X.dtype)
    res = builder.inline_func(_cov_impl_inner, array_type, X, ddof)
    if _cov_scalar_result_expected(m, y):
        res = res[0, 0]
    return res
