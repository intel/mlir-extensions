add_mlir_dialect(XeTileOps xetile)
add_mlir_doc(XeTileOps XeTileDialect Dialects/ -gen-dialect-doc -dialect=xetile)

set(LLVM_TARGET_DEFINITIONS XeTileOps.td)
mlir_tablegen(XeTileOpsAttrs.h.inc -gen-attrdef-decls --attrdefs-dialect=xetile)
mlir_tablegen(XeTileOpsAttrs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=xetile)

set(LLVM_TARGET_DEFINITIONS XeTileAttrs.td)
mlir_tablegen(XeTileOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(XeTileOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(XeTileOpsAttrsIncGen)
