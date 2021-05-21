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

USE_MLIR = _readenv('DPCOMP_ENABLE', int, 1)
DUMP_PLIER = _readenv('DPCOMP_DUMP_PLIER', int, 0)
PRINT_IR = _readenv('DPCOMP_PRINT_IR', int, 0)
DEBUG_TYPE = list(filter(None, _readenv('DPCOMP_DEBUG_TYPE', str, '').split(',')))
