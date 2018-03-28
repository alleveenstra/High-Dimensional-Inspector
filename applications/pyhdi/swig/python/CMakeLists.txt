# Include python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})


set(CMAKE_SWIG_FLAGS "")
set_source_files_properties(../pyhdi.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(../pyhdi.i PROPERTIES SWIG_FLAGS "-includeall")
include_directories(../..)

# Add swig module
swig_add_module(pyhdi python ../pyhdi.i ../../hdi_wrapper.h)
swig_link_libraries(pyhdi hdi_wrapper ${PYTHON_LIBRARIES})

# Files to install with Python
set(PYTHON_INSTALL_FILES
        ${CMAKE_CURRENT_BINARY_DIR}/pyhdi.py
        ${CMAKE_CURRENT_BINARY_DIR}/_pyhdi.so)

# Configure setup.py and copy to output directory
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

# Declare install target for python
#install(TARGETS swig_example
#        COMMAND "${PYTHON_EXECUTABLE} setup.py"
#        COMPONENT swig-python)

# Install target to call setup.py
add_custom_target(install-python
        DEPENDS _pyhdi
        COMMAND python ${SETUP_PY_OUT} install)