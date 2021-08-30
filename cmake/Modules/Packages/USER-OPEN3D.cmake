#cmake_policy(PUSH)
#cmake_minimum_required(VERSION 3.18)

# The options need to be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
if(STATIC_WINDOWS_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

message("compiler version is ${CMAKE_CXX_COMPILER_VERSION}")
message("c++ version is ${CMAKE_CXX_STANDARD}")
message("Using CMake version ${CMAKE_VERSION}")

find_package(Open3D)
target_compile_definitions(lammps PRIVATE -DLAMMPS_OPEND3D)
cmake_policy(PUSH)
target_link_libraries(lammps PRIVATE Open3D::Open3D)
cmake_policy(POP)
#target_link_libraries(lammps PRIVATE ${OPEN3D_LIBRARIES})

message("compiler version is ${CMAKE_CXX_COMPILER_VERSION}")
message("c++ version is ${CMAKE_CXX_STANDARD}")
message("Using CMake version ${CMAKE_VERSION}")

# On Windows if BUILD_SHARED_LIBS is enabled, copy .dll files to the executable directory
if(WIN32)
    get_target_property(open3d_type Open3D::Open3D TYPE)
    if(open3d_type STREQUAL "SHARED_LIBRARY")
        message(STATUS "Copying Open3D.dll to ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
        add_custom_command(TARGET Draw POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy
                                   ${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll
                                   ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
    endif()
endif()
#cmake_policy(POP)
