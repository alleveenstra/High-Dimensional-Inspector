set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -Wall -Wextra -Wno-unused-parameter -fPIC")

if(APPLE)
else(APPLE)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        message (STATUS "OpenMP found")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif(APPLE)

add_library(hdi_wrapper hdi_wrapper.cpp)
set_target_properties(hdi_wrapper PROPERTIES PUBLIC_HEADER "hdi_wrapper.h")

include_directories ("${CMAKE_SOURCE_DIR}")
target_link_libraries (hdi_wrapper LINK_PUBLIC hdianalytics)
target_link_libraries (hdi_wrapper LINK_PUBLIC hdiclustering)
target_link_libraries (hdi_wrapper LINK_PUBLIC hdidata)
target_link_libraries (hdi_wrapper LINK_PUBLIC hdidimensionalityreduction)
target_link_libraries (hdi_wrapper LINK_PUBLIC hdiutils)
target_link_libraries (hdi_wrapper LINK_PUBLIC hdivisualization)

## Add install target
install(TARGETS hdi_wrapper
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        PUBLIC_HEADER DESTINATION include)

# Add swig subdirectory
add_subdirectory(swig)