add_custom_target(imex-runner
                  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/imex-runner.py ${IMEX_BINARY_DIR}/bin
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/imex-runner.py
                  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/imex-runner.py)
