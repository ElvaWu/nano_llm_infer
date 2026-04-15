# Install script for directory: /root/autodl-tmp/repos/nano_llm_infer/third_party/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/root/autodl-tmp/repos/nano_llm_infer/build/third_party/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/root/autodl-tmp/repos/nano_llm_infer/build/third_party/ggml/src/libggml.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-cpu.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-alloc.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-backend.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-blas.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-cann.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-cpp.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-cuda.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-opt.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-metal.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-rpc.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-virtgpu.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-sycl.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-vulkan.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-webgpu.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-zendnn.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/ggml-openvino.h"
    "/root/autodl-tmp/repos/nano_llm_infer/third_party/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/root/autodl-tmp/repos/nano_llm_infer/build/third_party/ggml/src/libggml-base.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/root/autodl-tmp/repos/nano_llm_infer/build/third_party/ggml/ggml-config.cmake"
    "/root/autodl-tmp/repos/nano_llm_infer/build/third_party/ggml/ggml-version.cmake"
    )
endif()

