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