from os import environ

def _readenv(name, ctor, default):
    value = environ.get(name)
    if value is None:
        return default() if callable(default) else default
    try:
        return ctor(value)
    except Exception:
        warnings.warn("environ %s defined but failed to parse '%s'" %
                      (name, value), RuntimeWarning)
        return default

USE_MLIR = _readenv('NUMBA_MLIR_ENABLE', int, 1)
PRINT_IR = _readenv('NUMBA_MLIR_PRINT_IR', int, 0)
