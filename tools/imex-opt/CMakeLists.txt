get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(imex_dialect_libs GLOBAL PROPERTY IMEX_DIALECT_LIBS)
get_property(imex_conversion_libs GLOBAL PROPERTY IMEX_CONVERSION_LIBS)
get_property(imex_extension_libs GLOBAL PROPERTY IMEX_EXTENSION_LIBS)
set(LIBS
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        ${mlir_extension_libs}
        ${imex_dialect_libs}
        ${imex_conversion_libs}
        ${imex_extension_libs}
        MLIROptLib
        IMEXTransforms
        IMEXUtil
        )
add_imex_tool(imex-opt imex-opt.cpp)
llvm_update_compile_flags(imex-opt)
target_link_libraries(imex-opt PRIVATE ${LIBS} IMEXTransforms)
