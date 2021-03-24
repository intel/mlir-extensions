import ctypes
import os
import atexit
from numba.np.ufunc.parallel import get_thread_count
import llvmlite.binding as ll

def load_runtume_lib():
    runtime_search_paths = ['']

    try:
        runtime_search_paths += os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        pass

    lib_name = 'dpcomp-runtime'

    for path in runtime_search_paths:
        lib_path = lib_name if len(path) == 0 else os.path.join(path, lib_name)
        try:
            return ctypes.CDLL(lib_path)
        except:
            pass

    return None

runtime_lib = load_runtume_lib();
assert not runtime_lib is None

_init_func = runtime_lib.dpcomp_parallel_init
_init_func.argtypes = [ctypes.c_int]
_init_func(get_thread_count())

_finalize_func = runtime_lib.dpcomp_parallel_finalize

_parallel_for_func = runtime_lib.dpcomp_parallel_for
ll.add_symbol('dpcomp_parallel_for', ctypes.cast(_parallel_for_func, ctypes.c_void_p).value)

@atexit.register
def _cleanup():
    _finalize_func()

