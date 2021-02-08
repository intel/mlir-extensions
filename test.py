import numba
import numpy as np

_tests_total = 0
_tests_passes = 0
_failed_tests = []

def ret(a):
    return a

def const():
    return 42

def sum1(a):
    return a + 42

def sum2(a, b):
    return a + b

def cond(a, b):
    if a > b:
        return a
    else:
        return b

def var(a):
    c = 1
    c = c + a
    return c

def jump(a, b):
    c = 3
    if a > 5:
        c = c + a
    c = c + b
    return c

sum2_jit = numba.njit()(sum2)

def call(a, b, c):
    return sum2_jit(a, sum2_jit(b, c))

def tuple(a,b,c):
    t = (a,b,c)
    return t[0] + t[1] + t[2]

def arr_loop():
    res = 0
    arr = [1,2,3]
    for i in arr:
        res = res + i
    return res

def range_loop(n):
    res = 0
    res1 = 2
    for i in range(n):
        if i > 5:
            res = res + i
        else:
            res1 = res1 + i * 2
    return res + res1

def range_loop_nested(a, b, c):
    res = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                res = res + i + j * 10 + k * 100
    return res

def np_getitem(a, b):
    return a[b]

def np_getitem2(a, b, c):
    return b[c]

def np_sum(a):
    return a.sum()

def np_add(a, b):
    return np.add(a, b).sum()

def np_add2(a, b, c):
    t = np.add(a, b)
    return np.add(t, c).sum()

def test(func, params):
    global _tests_total
    global _tests_passes
    global _failed_tests
    _tests_total += 1
    test_name = f'{func.__name__} {params}'
    print('test', test_name, '... ', end='')
    result = func(*params)
    wrapped = numba.njit()(func)
    try:
        res = wrapped(*params)
        if (res != result):
            raise Exception(f'Invalid value "{res}", expected "{result}"')
        print('SUCCESS')
        _tests_passes += 1
    except Exception as e:
        print(e)
        print('FAILED')
        _failed_tests.append(test_name)

print('=========================================================')

test(ret, (7,))
test(const, ())
test(sum1, (5,))
test(sum2, (3,4))
test(cond, (5,6))
test(cond, (8,7))
test(var, (8,))
test(jump, (1,8))
test(jump, (7,8))
test(call, (1,2,3))
test(tuple, (1,2,3))
test(tuple, (1,2.0,3))
test(arr_loop, ())
test(range_loop, (8,))
test(range_loop_nested, (8,9,10))
test(sum2, (np.asarray([1,2,3]),np.asarray([4,5,6])))
test(np_getitem, (np.asarray([1,2,3]),1))
test(np_getitem2, (np.asarray([1,2,3]),np.asarray([4,5,6]),1))
test(np_sum, (np.asarray([1,2,3]),))
test(np_add, (np.asarray([1,2,3]),np.asarray([4,5,6])))
test(np_add2, (np.asarray([1,2,3]),np.asarray([4,5,6]),np.asarray([7,8,9])))

print(f'Tests passed: {_tests_passes}/{_tests_total}')
if (len(_failed_tests) != 0):
    print('Failed:')
    for t in _failed_tests:
        print(t)
