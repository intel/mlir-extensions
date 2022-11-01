<!--
SPDX-FileCopyrightText: 2022 Intel Corporation

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# numba dpcomp Python implementation

## numpy functions

*numpy/funcs.py* file contains implemented numpy functions.

General form is:
```
@register_func('numpy.<function_name>', numpy.<function_name>)
def <function_name>_impl(builder, <function_args: arg1, arg2, ..., argn>):
    def body(<function args + 1: arg1, arg2, ..., argn, arg(n+1)>):
        <implementation>
        return <result>

    return eltwise(builder, <function args>, body, <optional: return type>)
```

*Note:* Python math functions can be used.
In this case do not forget to add implemented function to *math_funcs.py* in *_funcs* list.

Tests should be added to *tests/test_numpy.py* file.
