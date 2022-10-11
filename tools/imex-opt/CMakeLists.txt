get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
	MLIROptLib
        IMEXTransforms
        )
add_llvm_executable(imex-opt imex-opt.cpp)

llvm_update_compile_flags(imex-opt)
target_link_libraries(imex-opt PRIVATE ${LIBS} IMEXTransforms)
