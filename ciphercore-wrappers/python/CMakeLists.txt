cmake_minimum_required(VERSION 3.5)

project(CipherCorePython)

add_subdirectory(external/pybind11)

find_package(OpenSSL REQUIRED)

add_custom_target(
  ciphercore
  COMMAND cargo build --release
  COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/lib && cp target/release/libcadapter.a ${CMAKE_CURRENT_SOURCE_DIR}/lib/
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../"
  )

set(HEADER_DEPENDENCIES include/)
set(LIBRARY_DEPENDENCIES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcadapter.a OpenSSL::Crypto OpenSSL::SSL)
add_library(ciphercore_lib INTERFACE)
add_dependencies(ciphercore_lib ciphercore)
target_include_directories(ciphercore_lib INTERFACE ${HEADER_DEPENDENCIES})
target_link_libraries(ciphercore_lib INTERFACE ${LIBRARY_DEPENDENCIES})
target_compile_options(ciphercore_lib INTERFACE -O3 -Wall -std=c++17 -march=native -g -Wno-return-type-c-linkage)

pybind11_add_module(ciphercore_native src/ciphercore_native.cpp)
target_link_libraries("ciphercore_native" PRIVATE ciphercore_lib)
