## Adding a new dialect
```sh
# enter root directory of mlir-extension
cd mlir-extensions
python scripts/add_dialect.py <name-of-new-dialect>
```
This will
* generate directories in the appropriate directories (include/mlir/Dialect and lib/dialect)
* Extend/Create cmake infrastrcuture with defaults
* Create stub soruce files for IR/Ops

## How to build
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## License
This code is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
