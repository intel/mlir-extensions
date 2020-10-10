from numba import runtests

def test(*args, **kwargs):
    return runtests.main("numba.mlir.tests", *args, **kwargs)
