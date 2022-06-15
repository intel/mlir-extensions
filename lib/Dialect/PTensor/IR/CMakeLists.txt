add_mlir_dialect_library(MLIRPTensor
        PTensorOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/PTensor

        DEPENDS
        MLIRPTensorOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
