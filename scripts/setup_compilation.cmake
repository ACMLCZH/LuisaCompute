find_package(Threads REQUIRED)

if (APPLE)
    if (LUISA_COMPUTE_ENABLE_METAL OR LUISA_COMPUTE_ENABLE_GUI)
        enable_language(OBJC)
        enable_language(OBJCXX)
    endif ()
endif ()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS ON)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64" OR
        CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    # enable AVX2 for embree on x64
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    else ()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mf16c")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mf16c")
    endif ()
else ()
    if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # workaround for Apple clang -Xarch_arm64 bug with precompiled headers
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-command-line-argument -Xarch_arm64 no-unused-command-line-argument")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument -Xarch_arm64 no-unused-command-line-argument")
    endif ()
endif ()

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    option(LUISA_COMPUTE_ENABLE_SCCACHE "Enable sccache to speed up compilation" ON)
    if (LUISA_COMPUTE_ENABLE_SCCACHE)
        find_program(SCCACHE_EXE sccache)
        if (SCCACHE_EXE)
            message(STATUS "Build with sccache: ${SCCACHE_EXE}")
            set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_EXE})
            set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_EXE})
            set(CMAKE_OBJC_COMPILER_LAUNCHER ${SCCACHE_EXE})
            set(CMAKE_OBJCXX_COMPILER_LAUNCHER ${SCCACHE_EXE})
        else ()
            message(STATUS "Could not find sccache")
        endif ()
    endif ()
endif ()

# LTO
option(LUISA_COMPUTE_ENABLE_LTO "Enable link-time optimization (for release builds only)" ON)
if (LUISA_COMPUTE_ENABLE_LTO)
    if (CMAKE_C_COMPILER_ID MATCHES "Clang" AND
            CMAKE_C_COMPILER_VERSION VERSION_LESS 15.0)
        set(LUISA_COMPUTE_ENABLE_LTO OFF)
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0)
        set(LUISA_COMPUTE_ENABLE_LTO OFF)
    endif ()
    if (NOT LUISA_COMPUTE_ENABLE_LTO)
        message(STATUS "LTO disabled for clang ${CMAKE_CXX_COMPILER_VERSION} due to non-default opaque pointer support")
    endif ()
endif ()
if (LUISA_COMPUTE_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LUISA_LTO_SUPPORTED OUTPUT LUISA_LTO_CHECK_OUTPUT)
    if (LUISA_LTO_SUPPORTED)
        message(STATUS "IPO/LTO enabled for release builds")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
    else ()
        message(STATUS "IPO/LTO not supported: ${LUISA_LTO_CHECK_OUTPUT}")
    endif ()
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

get_cmake_property(LUISA_COMPUTE_IS_MULTI_CONFIG GENERATOR_IS_MULTI_CONFIG)
if (LUISA_COMPUTE_IS_MULTI_CONFIG)
    foreach (config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${config} CONFIG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin/${config}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin/${config}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/lib/${config}")
        set(CMAKE_PDB_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/lib/${config}")
    endforeach ()
else ()
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release")
    endif ()
endif ()

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)
