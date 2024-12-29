set(LUISA_COMPUTE_XIR_SOURCES

        argument.cpp
        basic_block.cpp
        builder.cpp
        constant.cpp
        function.cpp
        instruction.cpp
        metadata.cpp
        module.cpp
        pool.cpp
        use.cpp
        user.cpp
        value.cpp

        # derived instructions
        instructions/alloca.cpp
        instructions/assert.cpp
        instructions/call.cpp
        instructions/cast.cpp
        instructions/gep.cpp
        instructions/intrinsic.cpp
        instructions/load.cpp
        instructions/loop.cpp
        instructions/phi.cpp
        instructions/print.cpp
        instructions/ray_query.cpp
        instructions/return.cpp
        instructions/store.cpp
        instructions/switch.cpp
        instructions/unreachable.cpp

        # derived metadata
        metadata/location.cpp
        metadata/comment.cpp
        metadata/name.cpp

        # translators
        translators/ast2xir.cpp
        translators/xir2text.cpp
)

add_library(luisa-compute-xir SHARED ${LUISA_COMPUTE_XIR_SOURCES})
target_link_libraries(luisa-compute-xir PUBLIC luisa-compute-ast)

target_compile_definitions(luisa-compute-xir PUBLIC LUISA_ENABLE_XIR=1)
target_compile_definitions(luisa-compute-xir PRIVATE LC_XIR_EXPORT_DLL=1)

set_target_properties(luisa-compute-xir PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-xir)

luisa_compute_install(xir)
