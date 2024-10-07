add_imex_dialect_library(IMEXNDArrayDialect
  NDArrayOps.cpp
  SubviewOp.cpp
  InsertSliceOp.cpp
  CreateOp.cpp
  LinSpaceOp.cpp
  CastOp.cpp
  DimOp.cpp
  EWBinOp.cpp
  EWUnyOp.cpp
  PermuteDimsOp.cpp
  DeleteOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/NDArray

  DEPENDS
  MLIRNDArrayOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
