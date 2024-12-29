//
// Created by Mike on 2024/10/20.
//

#include "luisa/core/stl/unordered_map.h"

#include <luisa/core/logging.h>
#include <luisa/xir/instructions/intrinsic.h>

namespace luisa::compute::xir {

IntrinsicInst::IntrinsicInst(const Type *type, IntrinsicOp op,
                             luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{type}, _op{op} { set_operands(operands); }

luisa::string to_string(IntrinsicOp op) noexcept {
    switch (op) {
        case IntrinsicOp::NOP: return "nop";
        case IntrinsicOp::UNARY_PLUS: return "unary_plus";
        case IntrinsicOp::UNARY_MINUS: return "unary_minus";
        case IntrinsicOp::UNARY_NOT: return "unary_not";
        case IntrinsicOp::UNARY_BIT_NOT: return "unary_bit_not";
        case IntrinsicOp::BINARY_ADD: return "binary_add";
        case IntrinsicOp::BINARY_SUB: return "binary_sub";
        case IntrinsicOp::BINARY_MUL: return "binary_mul";
        case IntrinsicOp::BINARY_DIV: return "binary_div";
        case IntrinsicOp::BINARY_MOD: return "binary_mod";
        case IntrinsicOp::BINARY_AND: return "binary_and";
        case IntrinsicOp::BINARY_OR: return "binary_or";
        case IntrinsicOp::BINARY_BIT_AND: return "binary_bit_and";
        case IntrinsicOp::BINARY_BIT_OR: return "binary_bit_or";
        case IntrinsicOp::BINARY_BIT_XOR: return "binary_bit_xor";
        case IntrinsicOp::BINARY_SHIFT_LEFT: return "binary_shift_left";
        case IntrinsicOp::BINARY_SHIFT_RIGHT: return "binary_shift_right";
        case IntrinsicOp::BINARY_ROTATE_LEFT: return "binary_rotate_left";
        case IntrinsicOp::BINARY_ROTATE_RIGHT: return "binary_rotate_right";
        case IntrinsicOp::BINARY_LESS: return "binary_less";
        case IntrinsicOp::BINARY_GREATER: return "binary_greater";
        case IntrinsicOp::BINARY_LESS_EQUAL: return "binary_less_equal";
        case IntrinsicOp::BINARY_GREATER_EQUAL: return "binary_greater_equal";
        case IntrinsicOp::BINARY_EQUAL: return "binary_equal";
        case IntrinsicOp::BINARY_NOT_EQUAL: return "binary_not_equal";
        case IntrinsicOp::ASSUME: return "assume";
        case IntrinsicOp::ASSERT: return "assert";
        case IntrinsicOp::ADDRESS_OF: return "address_of";
        case IntrinsicOp::THREAD_ID: return "thread_id";
        case IntrinsicOp::BLOCK_ID: return "block_id";
        case IntrinsicOp::WARP_SIZE: return "warp_size";
        case IntrinsicOp::WARP_LANE_ID: return "warp_lane_id";
        case IntrinsicOp::DISPATCH_ID: return "dispatch_id";
        case IntrinsicOp::DISPATCH_SIZE: return "dispatch_size";
        case IntrinsicOp::KERNEL_ID: return "kernel_id";
        case IntrinsicOp::OBJECT_ID: return "object_id";
        case IntrinsicOp::BLOCK_SIZE: return "block_size";
        case IntrinsicOp::SYNCHRONIZE_BLOCK: return "synchronize_block";
        case IntrinsicOp::ALL: return "all";
        case IntrinsicOp::ANY: return "any";
        case IntrinsicOp::SELECT: return "select";
        case IntrinsicOp::CLAMP: return "clamp";
        case IntrinsicOp::SATURATE: return "saturate";
        case IntrinsicOp::LERP: return "lerp";
        case IntrinsicOp::SMOOTHSTEP: return "smoothstep";
        case IntrinsicOp::STEP: return "step";
        case IntrinsicOp::ABS: return "abs";
        case IntrinsicOp::MIN: return "min";
        case IntrinsicOp::MAX: return "max";
        case IntrinsicOp::CLZ: return "clz";
        case IntrinsicOp::CTZ: return "ctz";
        case IntrinsicOp::POPCOUNT: return "popcount";
        case IntrinsicOp::REVERSE: return "reverse";
        case IntrinsicOp::ISINF: return "isinf";
        case IntrinsicOp::ISNAN: return "isnan";
        case IntrinsicOp::ACOS: return "acos";
        case IntrinsicOp::ACOSH: return "acosh";
        case IntrinsicOp::ASIN: return "asin";
        case IntrinsicOp::ASINH: return "asinh";
        case IntrinsicOp::ATAN: return "atan";
        case IntrinsicOp::ATAN2: return "atan2";
        case IntrinsicOp::ATANH: return "atanh";
        case IntrinsicOp::COS: return "cos";
        case IntrinsicOp::COSH: return "cosh";
        case IntrinsicOp::SIN: return "sin";
        case IntrinsicOp::SINH: return "sinh";
        case IntrinsicOp::TAN: return "tan";
        case IntrinsicOp::TANH: return "tanh";
        case IntrinsicOp::EXP: return "exp";
        case IntrinsicOp::EXP2: return "exp2";
        case IntrinsicOp::EXP10: return "exp10";
        case IntrinsicOp::LOG: return "log";
        case IntrinsicOp::LOG2: return "log2";
        case IntrinsicOp::LOG10: return "log10";
        case IntrinsicOp::POW: return "pow";
        case IntrinsicOp::POW_INT: return "pow_int";
        case IntrinsicOp::SQRT: return "sqrt";
        case IntrinsicOp::RSQRT: return "rsqrt";
        case IntrinsicOp::CEIL: return "ceil";
        case IntrinsicOp::FLOOR: return "floor";
        case IntrinsicOp::FRACT: return "fract";
        case IntrinsicOp::TRUNC: return "trunc";
        case IntrinsicOp::ROUND: return "round";
        case IntrinsicOp::FMA: return "fma";
        case IntrinsicOp::COPYSIGN: return "copysign";
        case IntrinsicOp::CROSS: return "cross";
        case IntrinsicOp::DOT: return "dot";
        case IntrinsicOp::LENGTH: return "length";
        case IntrinsicOp::LENGTH_SQUARED: return "length_squared";
        case IntrinsicOp::NORMALIZE: return "normalize";
        case IntrinsicOp::FACEFORWARD: return "faceforward";
        case IntrinsicOp::REFLECT: return "reflect";
        case IntrinsicOp::REDUCE_SUM: return "reduce_sum";
        case IntrinsicOp::REDUCE_PRODUCT: return "reduce_product";
        case IntrinsicOp::REDUCE_MIN: return "reduce_min";
        case IntrinsicOp::REDUCE_MAX: return "reduce_max";
        case IntrinsicOp::OUTER_PRODUCT: return "outer_product";
        case IntrinsicOp::MATRIX_COMP_MUL: return "matrix_comp_mul";
        case IntrinsicOp::DETERMINANT: return "determinant";
        case IntrinsicOp::TRANSPOSE: return "transpose";
        case IntrinsicOp::INVERSE: return "inverse";
        case IntrinsicOp::ATOMIC_EXCHANGE: return "atomic_exchange";
        case IntrinsicOp::ATOMIC_COMPARE_EXCHANGE: return "atomic_compare_exchange";
        case IntrinsicOp::ATOMIC_FETCH_ADD: return "atomic_fetch_add";
        case IntrinsicOp::ATOMIC_FETCH_SUB: return "atomic_fetch_sub";
        case IntrinsicOp::ATOMIC_FETCH_AND: return "atomic_fetch_and";
        case IntrinsicOp::ATOMIC_FETCH_OR: return "atomic_fetch_or";
        case IntrinsicOp::ATOMIC_FETCH_XOR: return "atomic_fetch_xor";
        case IntrinsicOp::ATOMIC_FETCH_MIN: return "atomic_fetch_min";
        case IntrinsicOp::ATOMIC_FETCH_MAX: return "atomic_fetch_max";
        case IntrinsicOp::BUFFER_READ: return "buffer_read";
        case IntrinsicOp::BUFFER_WRITE: return "buffer_write";
        case IntrinsicOp::BUFFER_SIZE: return "buffer_size";
        case IntrinsicOp::BUFFER_ADDRESS: return "buffer_address";
        case IntrinsicOp::BYTE_BUFFER_READ: return "byte_buffer_read";
        case IntrinsicOp::BYTE_BUFFER_WRITE: return "byte_buffer_write";
        case IntrinsicOp::BYTE_BUFFER_SIZE: return "byte_buffer_size";
        case IntrinsicOp::TEXTURE_READ: return "texture_read";
        case IntrinsicOp::TEXTURE_WRITE: return "texture_write";
        case IntrinsicOp::TEXTURE_SIZE: return "texture_size";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE: return "bindless_texture2d_sample";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return "bindless_texture2d_sample_level";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return "bindless_texture2d_sample_grad";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: return "bindless_texture2d_sample_grad_level";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE: return "bindless_texture3d_sample";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return "bindless_texture3d_sample_level";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return "bindless_texture3d_sample_grad";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: return "bindless_texture3d_sample_grad_level";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: return "bindless_texture2d_sample_sampler";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: return "bindless_texture2d_sample_level_sampler";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: return "bindless_texture2d_sample_grad_sampler";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: return "bindless_texture2d_sample_grad_level_sampler";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: return "bindless_texture3d_sample_sampler";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: return "bindless_texture3d_sample_level_sampler";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: return "bindless_texture3d_sample_grad_sampler";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return "bindless_texture3d_sample_grad_level_sampler";
        case IntrinsicOp::BINDLESS_TEXTURE2D_READ: return "bindless_texture2d_read";
        case IntrinsicOp::BINDLESS_TEXTURE3D_READ: return "bindless_texture3d_read";
        case IntrinsicOp::BINDLESS_TEXTURE2D_READ_LEVEL: return "bindless_texture2d_read_level";
        case IntrinsicOp::BINDLESS_TEXTURE3D_READ_LEVEL: return "bindless_texture3d_read_level";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SIZE: return "bindless_texture2d_size";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SIZE: return "bindless_texture3d_size";
        case IntrinsicOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: return "bindless_texture2d_size_level";
        case IntrinsicOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return "bindless_texture3d_size_level";
        case IntrinsicOp::BINDLESS_BUFFER_READ: return "bindless_buffer_read";
        case IntrinsicOp::BINDLESS_BUFFER_WRITE: return "bindless_buffer_write";
        case IntrinsicOp::BINDLESS_BYTE_BUFFER_READ: return "bindless_byte_buffer_read";
        case IntrinsicOp::BINDLESS_BUFFER_SIZE: return "bindless_buffer_size";
        case IntrinsicOp::BINDLESS_BUFFER_TYPE: return "bindless_buffer_type";
        case IntrinsicOp::BINDLESS_BUFFER_ADDRESS: return "bindless_buffer_address";
        case IntrinsicOp::AGGREGATE: return "aggregate";
        case IntrinsicOp::SHUFFLE: return "shuffle";
        case IntrinsicOp::INSERT: return "insert";
        case IntrinsicOp::EXTRACT: return "extract";
        case IntrinsicOp::REQUIRES_GRADIENT: return "requires_gradient";
        case IntrinsicOp::GRADIENT: return "gradient";
        case IntrinsicOp::GRADIENT_MARKER: return "gradient_marker";
        case IntrinsicOp::ACCUMULATE_GRADIENT: return "accumulate_gradient";
        case IntrinsicOp::BACKWARD: return "backward";
        case IntrinsicOp::DETACH: return "detach";
        case IntrinsicOp::RAY_TRACING_INSTANCE_TRANSFORM: return "ray_tracing_instance_transform";
        case IntrinsicOp::RAY_TRACING_INSTANCE_USER_ID: return "ray_tracing_instance_user_id";
        case IntrinsicOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return "ray_tracing_instance_visibility_mask";
        case IntrinsicOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: return "ray_tracing_set_instance_transform";
        case IntrinsicOp::RAY_TRACING_SET_INSTANCE_VISIBILITY: return "ray_tracing_set_instance_visibility";
        case IntrinsicOp::RAY_TRACING_SET_INSTANCE_OPACITY: return "ray_tracing_set_instance_opacity";
        case IntrinsicOp::RAY_TRACING_SET_INSTANCE_USER_ID: return "ray_tracing_set_instance_user_id";
        case IntrinsicOp::RAY_TRACING_TRACE_CLOSEST: return "ray_tracing_trace_closest";
        case IntrinsicOp::RAY_TRACING_TRACE_ANY: return "ray_tracing_trace_any";
        case IntrinsicOp::RAY_TRACING_QUERY_ALL: return "ray_tracing_query_all";
        case IntrinsicOp::RAY_TRACING_QUERY_ANY: return "ray_tracing_query_any";
        case IntrinsicOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: return "ray_tracing_instance_motion_matrix";
        case IntrinsicOp::RAY_TRACING_INSTANCE_MOTION_SRT: return "ray_tracing_instance_motion_srt";
        case IntrinsicOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: return "ray_tracing_set_instance_motion_matrix";
        case IntrinsicOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: return "ray_tracing_set_instance_motion_srt";
        case IntrinsicOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: return "ray_tracing_trace_closest_motion_blur";
        case IntrinsicOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: return "ray_tracing_trace_any_motion_blur";
        case IntrinsicOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: return "ray_tracing_query_all_motion_blur";
        case IntrinsicOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: return "ray_tracing_query_any_motion_blur";
        case IntrinsicOp::RAY_QUERY_WORLD_SPACE_RAY: return "ray_query_world_space_ray";
        case IntrinsicOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT: return "ray_query_procedural_candidate_hit";
        case IntrinsicOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT: return "ray_query_triangle_candidate_hit";
        case IntrinsicOp::RAY_QUERY_COMMITTED_HIT: return "ray_query_committed_hit";
        case IntrinsicOp::RAY_QUERY_COMMIT_TRIANGLE: return "ray_query_commit_triangle";
        case IntrinsicOp::RAY_QUERY_COMMIT_PROCEDURAL: return "ray_query_commit_procedural";
        case IntrinsicOp::RAY_QUERY_TERMINATE: return "ray_query_terminate";
        case IntrinsicOp::RAY_QUERY_PROCEED: return "ray_query_proceed";
        case IntrinsicOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE: return "ray_query_is_triangle_candidate";
        case IntrinsicOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE: return "ray_query_is_procedural_candidate";
        case IntrinsicOp::RASTER_DISCARD: return "raster_discard";
        case IntrinsicOp::DDX: return "ddx";
        case IntrinsicOp::DDY: return "ddy";
        case IntrinsicOp::WARP_IS_FIRST_ACTIVE_LANE: return "warp_is_first_active_lane";
        case IntrinsicOp::WARP_FIRST_ACTIVE_LANE: return "warp_first_active_lane";
        case IntrinsicOp::WARP_ACTIVE_ALL_EQUAL: return "warp_active_all_equal";
        case IntrinsicOp::WARP_ACTIVE_BIT_AND: return "warp_active_bit_and";
        case IntrinsicOp::WARP_ACTIVE_BIT_OR: return "warp_active_bit_or";
        case IntrinsicOp::WARP_ACTIVE_BIT_XOR: return "warp_active_bit_xor";
        case IntrinsicOp::WARP_ACTIVE_COUNT_BITS: return "warp_active_count_bits";
        case IntrinsicOp::WARP_ACTIVE_MAX: return "warp_active_max";
        case IntrinsicOp::WARP_ACTIVE_MIN: return "warp_active_min";
        case IntrinsicOp::WARP_ACTIVE_PRODUCT: return "warp_active_product";
        case IntrinsicOp::WARP_ACTIVE_SUM: return "warp_active_sum";
        case IntrinsicOp::WARP_ACTIVE_ALL: return "warp_active_all";
        case IntrinsicOp::WARP_ACTIVE_ANY: return "warp_active_any";
        case IntrinsicOp::WARP_ACTIVE_BIT_MASK: return "warp_active_bit_mask";
        case IntrinsicOp::WARP_PREFIX_COUNT_BITS: return "warp_prefix_count_bits";
        case IntrinsicOp::WARP_PREFIX_SUM: return "warp_prefix_sum";
        case IntrinsicOp::WARP_PREFIX_PRODUCT: return "warp_prefix_product";
        case IntrinsicOp::WARP_READ_LANE: return "warp_read_lane";
        case IntrinsicOp::WARP_READ_FIRST_ACTIVE_LANE: return "warp_read_first_active_lane";
        case IntrinsicOp::INDIRECT_SET_DISPATCH_KERNEL: return "indirect_set_dispatch_kernel";
        case IntrinsicOp::INDIRECT_SET_DISPATCH_COUNT: return "indirect_set_dispatch_count";
        case IntrinsicOp::TEXTURE2D_SAMPLE: return "texture2d_sample";
        case IntrinsicOp::TEXTURE2D_SAMPLE_LEVEL: return "texture2d_sample_level";
        case IntrinsicOp::TEXTURE2D_SAMPLE_GRAD: return "texture2d_sample_grad";
        case IntrinsicOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: return "texture2d_sample_grad_level";
        case IntrinsicOp::TEXTURE3D_SAMPLE: return "texture3d_sample";
        case IntrinsicOp::TEXTURE3D_SAMPLE_LEVEL: return "texture3d_sample_level";
        case IntrinsicOp::TEXTURE3D_SAMPLE_GRAD: return "texture3d_sample_grad";
        case IntrinsicOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: return "texture3d_sample_grad_level";
        case IntrinsicOp::SHADER_EXECUTION_REORDER: return "shader_execution_reorder";
    }
    LUISA_ERROR_WITH_LOCATION("Unknown intrinsic operation: {}.",
                              static_cast<uint32_t>(op));
}

IntrinsicOp intrinsic_op_from_string(luisa::string_view name) noexcept {
    static const luisa::unordered_map<luisa::string, IntrinsicOp> m{
        {"nop", IntrinsicOp::NOP},
        {"unary_plus", IntrinsicOp::UNARY_PLUS},
        {"unary_minus", IntrinsicOp::UNARY_MINUS},
        {"unary_not", IntrinsicOp::UNARY_NOT},
        {"unary_bit_not", IntrinsicOp::UNARY_BIT_NOT},
        {"binary_add", IntrinsicOp::BINARY_ADD},
        {"binary_sub", IntrinsicOp::BINARY_SUB},
        {"binary_mul", IntrinsicOp::BINARY_MUL},
        {"binary_div", IntrinsicOp::BINARY_DIV},
        {"binary_mod", IntrinsicOp::BINARY_MOD},
        {"binary_and", IntrinsicOp::BINARY_AND},
        {"binary_or", IntrinsicOp::BINARY_OR},
        {"binary_bit_and", IntrinsicOp::BINARY_BIT_AND},
        {"binary_bit_or", IntrinsicOp::BINARY_BIT_OR},
        {"binary_bit_xor", IntrinsicOp::BINARY_BIT_XOR},
        {"binary_shift_left", IntrinsicOp::BINARY_SHIFT_LEFT},
        {"binary_shift_right", IntrinsicOp::BINARY_SHIFT_RIGHT},
        {"binary_rotate_left", IntrinsicOp::BINARY_ROTATE_LEFT},
        {"binary_rotate_right", IntrinsicOp::BINARY_ROTATE_RIGHT},
        {"binary_less", IntrinsicOp::BINARY_LESS},
        {"binary_greater", IntrinsicOp::BINARY_GREATER},
        {"binary_less_equal", IntrinsicOp::BINARY_LESS_EQUAL},
        {"binary_greater_equal", IntrinsicOp::BINARY_GREATER_EQUAL},
        {"binary_equal", IntrinsicOp::BINARY_EQUAL},
        {"binary_not_equal", IntrinsicOp::BINARY_NOT_EQUAL},
        {"assume", IntrinsicOp::ASSUME},
        {"assert", IntrinsicOp::ASSERT},
        {"address_of", IntrinsicOp::ADDRESS_OF},
        {"thread_id", IntrinsicOp::THREAD_ID},
        {"block_id", IntrinsicOp::BLOCK_ID},
        {"block_size", IntrinsicOp::BLOCK_SIZE},
        {"warp_size", IntrinsicOp::WARP_SIZE},
        {"warp_lane_id", IntrinsicOp::WARP_LANE_ID},
        {"dispatch_id", IntrinsicOp::DISPATCH_ID},
        {"dispatch_size", IntrinsicOp::DISPATCH_SIZE},
        {"kernel_id", IntrinsicOp::KERNEL_ID},
        {"object_id", IntrinsicOp::OBJECT_ID},
        {"synchronize_block", IntrinsicOp::SYNCHRONIZE_BLOCK},
        {"all", IntrinsicOp::ALL},
        {"any", IntrinsicOp::ANY},
        {"select", IntrinsicOp::SELECT},
        {"clamp", IntrinsicOp::CLAMP},
        {"saturate", IntrinsicOp::SATURATE},
        {"lerp", IntrinsicOp::LERP},
        {"smoothstep", IntrinsicOp::SMOOTHSTEP},
        {"step", IntrinsicOp::STEP},
        {"abs", IntrinsicOp::ABS},
        {"min", IntrinsicOp::MIN},
        {"max", IntrinsicOp::MAX},
        {"clz", IntrinsicOp::CLZ},
        {"ctz", IntrinsicOp::CTZ},
        {"popcount", IntrinsicOp::POPCOUNT},
        {"reverse", IntrinsicOp::REVERSE},
        {"isinf", IntrinsicOp::ISINF},
        {"isnan", IntrinsicOp::ISNAN},
        {"acos", IntrinsicOp::ACOS},
        {"acosh", IntrinsicOp::ACOSH},
        {"asin", IntrinsicOp::ASIN},
        {"asinh", IntrinsicOp::ASINH},
        {"atan", IntrinsicOp::ATAN},
        {"atan2", IntrinsicOp::ATAN2},
        {"atanh", IntrinsicOp::ATANH},
        {"cos", IntrinsicOp::COS},
        {"cosh", IntrinsicOp::COSH},
        {"sin", IntrinsicOp::SIN},
        {"sinh", IntrinsicOp::SINH},
        {"tan", IntrinsicOp::TAN},
        {"tanh", IntrinsicOp::TANH},
        {"exp", IntrinsicOp::EXP},
        {"exp2", IntrinsicOp::EXP2},
        {"exp10", IntrinsicOp::EXP10},
        {"log", IntrinsicOp::LOG},
        {"log2", IntrinsicOp::LOG2},
        {"log10", IntrinsicOp::LOG10},
        {"pow", IntrinsicOp::POW},
        {"pow_int", IntrinsicOp::POW_INT},
        {"sqrt", IntrinsicOp::SQRT},
        {"rsqrt", IntrinsicOp::RSQRT},
        {"ceil", IntrinsicOp::CEIL},
        {"floor", IntrinsicOp::FLOOR},
        {"fract", IntrinsicOp::FRACT},
        {"trunc", IntrinsicOp::TRUNC},
        {"round", IntrinsicOp::ROUND},
        {"fma", IntrinsicOp::FMA},
        {"copysign", IntrinsicOp::COPYSIGN},
        {"cross", IntrinsicOp::CROSS},
        {"dot", IntrinsicOp::DOT},
        {"length", IntrinsicOp::LENGTH},
        {"length_squared", IntrinsicOp::LENGTH_SQUARED},
        {"normalize", IntrinsicOp::NORMALIZE},
        {"faceforward", IntrinsicOp::FACEFORWARD},
        {"reflect", IntrinsicOp::REFLECT},
        {"reduce_sum", IntrinsicOp::REDUCE_SUM},
        {"reduce_product", IntrinsicOp::REDUCE_PRODUCT},
        {"reduce_min", IntrinsicOp::REDUCE_MIN},
        {"reduce_max", IntrinsicOp::REDUCE_MAX},
        {"outer_product", IntrinsicOp::OUTER_PRODUCT},
        {"matrix_comp_mul", IntrinsicOp::MATRIX_COMP_MUL},
        {"determinant", IntrinsicOp::DETERMINANT},
        {"transpose", IntrinsicOp::TRANSPOSE},
        {"inverse", IntrinsicOp::INVERSE},
        {"atomic_exchange", IntrinsicOp::ATOMIC_EXCHANGE},
        {"atomic_compare_exchange", IntrinsicOp::ATOMIC_COMPARE_EXCHANGE},
        {"atomic_fetch_add", IntrinsicOp::ATOMIC_FETCH_ADD},
        {"atomic_fetch_sub", IntrinsicOp::ATOMIC_FETCH_SUB},
        {"atomic_fetch_and", IntrinsicOp::ATOMIC_FETCH_AND},
        {"atomic_fetch_or", IntrinsicOp::ATOMIC_FETCH_OR},
        {"atomic_fetch_xor", IntrinsicOp::ATOMIC_FETCH_XOR},
        {"atomic_fetch_min", IntrinsicOp::ATOMIC_FETCH_MIN},
        {"atomic_fetch_max", IntrinsicOp::ATOMIC_FETCH_MAX},
        {"buffer_read", IntrinsicOp::BUFFER_READ},
        {"buffer_write", IntrinsicOp::BUFFER_WRITE},
        {"buffer_size", IntrinsicOp::BUFFER_SIZE},
        {"buffer_address", IntrinsicOp::BUFFER_ADDRESS},
        {"byte_buffer_read", IntrinsicOp::BYTE_BUFFER_READ},
        {"byte_buffer_write", IntrinsicOp::BYTE_BUFFER_WRITE},
        {"byte_buffer_size", IntrinsicOp::BYTE_BUFFER_SIZE},
        {"texture_read", IntrinsicOp::TEXTURE_READ},
        {"texture_write", IntrinsicOp::TEXTURE_WRITE},
        {"texture_size", IntrinsicOp::TEXTURE_SIZE},
        {"bindless_texture2d_sample", IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE},
        {"bindless_texture2d_sample_level", IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL},
        {"bindless_texture2d_sample_grad", IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD},
        {"bindless_texture2d_sample_grad_level", IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL},
        {"bindless_texture3d_sample", IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE},
        {"bindless_texture3d_sample_level", IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL},
        {"bindless_texture3d_sample_grad", IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD},
        {"bindless_texture3d_sample_grad_level", IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL},
        {"bindless_texture2d_sample_sampler", IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER},
        {"bindless_texture2d_sample_level_sampler", IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER},
        {"bindless_texture2d_sample_grad_sampler", IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER},
        {"bindless_texture2d_sample_grad_level_sampler", IntrinsicOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER},
        {"bindless_texture3d_sample_sampler", IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER},
        {"bindless_texture3d_sample_level_sampler", IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER},
        {"bindless_texture3d_sample_grad_sampler", IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER},
        {"bindless_texture3d_sample_grad_level_sampler", IntrinsicOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER},
        {"bindless_texture2d_read", IntrinsicOp::BINDLESS_TEXTURE2D_READ},
        {"bindless_texture3d_read", IntrinsicOp::BINDLESS_TEXTURE3D_READ},
        {"bindless_texture2d_read_level", IntrinsicOp::BINDLESS_TEXTURE2D_READ_LEVEL},
        {"bindless_texture3d_read_level", IntrinsicOp::BINDLESS_TEXTURE3D_READ_LEVEL},
        {"bindless_texture2d_size", IntrinsicOp::BINDLESS_TEXTURE2D_SIZE},
        {"bindless_texture3d_size", IntrinsicOp::BINDLESS_TEXTURE3D_SIZE},
        {"bindless_texture2d_size_level", IntrinsicOp::BINDLESS_TEXTURE2D_SIZE_LEVEL},
        {"bindless_texture3d_size_level", IntrinsicOp::BINDLESS_TEXTURE3D_SIZE_LEVEL},
        {"bindless_buffer_read", IntrinsicOp::BINDLESS_BUFFER_READ},
        {"bindless_buffer_write", IntrinsicOp::BINDLESS_BUFFER_WRITE},
        {"bindless_byte_buffer_read", IntrinsicOp::BINDLESS_BYTE_BUFFER_READ},
        {"bindless_buffer_size", IntrinsicOp::BINDLESS_BUFFER_SIZE},
        {"bindless_buffer_type", IntrinsicOp::BINDLESS_BUFFER_TYPE},
        {"bindless_buffer_address", IntrinsicOp::BINDLESS_BUFFER_ADDRESS},
        {"aggregate", IntrinsicOp::AGGREGATE},
        {"shuffle", IntrinsicOp::SHUFFLE},
        {"insert", IntrinsicOp::INSERT},
        {"extract", IntrinsicOp::EXTRACT},
        {"requires_gradient", IntrinsicOp::REQUIRES_GRADIENT},
        {"gradient", IntrinsicOp::GRADIENT},
        {"gradient_marker", IntrinsicOp::GRADIENT_MARKER},
        {"accumulate_gradient", IntrinsicOp::ACCUMULATE_GRADIENT},
        {"backward", IntrinsicOp::BACKWARD},
        {"detach", IntrinsicOp::DETACH},
        {"ray_tracing_instance_transform", IntrinsicOp::RAY_TRACING_INSTANCE_TRANSFORM},
        {"ray_tracing_instance_user_id", IntrinsicOp::RAY_TRACING_INSTANCE_USER_ID},
        {"ray_tracing_instance_visibility_mask", IntrinsicOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK},
        {"ray_tracing_set_instance_transform", IntrinsicOp::RAY_TRACING_SET_INSTANCE_TRANSFORM},
        {"ray_tracing_set_instance_visibility", IntrinsicOp::RAY_TRACING_SET_INSTANCE_VISIBILITY},
        {"ray_tracing_set_instance_opacity", IntrinsicOp::RAY_TRACING_SET_INSTANCE_OPACITY},
        {"ray_tracing_set_instance_user_id", IntrinsicOp::RAY_TRACING_SET_INSTANCE_USER_ID},
        {"ray_tracing_trace_closest", IntrinsicOp::RAY_TRACING_TRACE_CLOSEST},
        {"ray_tracing_trace_any", IntrinsicOp::RAY_TRACING_TRACE_ANY},
        {"ray_tracing_query_all", IntrinsicOp::RAY_TRACING_QUERY_ALL},
        {"ray_tracing_query_any", IntrinsicOp::RAY_TRACING_QUERY_ANY},
        {"ray_tracing_instance_motion_matrix", IntrinsicOp::RAY_TRACING_INSTANCE_MOTION_MATRIX},
        {"ray_tracing_instance_motion_srt", IntrinsicOp::RAY_TRACING_INSTANCE_MOTION_SRT},
        {"ray_tracing_set_instance_motion_matrix", IntrinsicOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX},
        {"ray_tracing_set_instance_motion_srt", IntrinsicOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT},
        {"ray_tracing_trace_closest_motion_blur", IntrinsicOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR},
        {"ray_tracing_trace_any_motion_blur", IntrinsicOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR},
        {"ray_tracing_query_all_motion_blur", IntrinsicOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR},
        {"ray_tracing_query_any_motion_blur", IntrinsicOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR},
        {"ray_query_world_space_ray", IntrinsicOp::RAY_QUERY_WORLD_SPACE_RAY},
        {"ray_query_procedural_candidate_hit", IntrinsicOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT},
        {"ray_query_triangle_candidate_hit", IntrinsicOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT},
        {"ray_query_committed_hit", IntrinsicOp::RAY_QUERY_COMMITTED_HIT},
        {"ray_query_commit_triangle", IntrinsicOp::RAY_QUERY_COMMIT_TRIANGLE},
        {"ray_query_commit_procedural", IntrinsicOp::RAY_QUERY_COMMIT_PROCEDURAL},
        {"ray_query_terminate", IntrinsicOp::RAY_QUERY_TERMINATE},
        {"ray_query_proceed", IntrinsicOp::RAY_QUERY_PROCEED},
        {"ray_query_is_triangle_candidate", IntrinsicOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE},
        {"ray_query_is_procedural_candidate", IntrinsicOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE},
        {"raster_discard", IntrinsicOp::RASTER_DISCARD},
        {"ddx", IntrinsicOp::DDX},
        {"ddy", IntrinsicOp::DDY},
        {"warp_is_first_active_lane", IntrinsicOp::WARP_IS_FIRST_ACTIVE_LANE},
        {"warp_first_active_lane", IntrinsicOp::WARP_FIRST_ACTIVE_LANE},
        {"warp_active_all_equal", IntrinsicOp::WARP_ACTIVE_ALL_EQUAL},
        {"warp_active_bit_and", IntrinsicOp::WARP_ACTIVE_BIT_AND},
        {"warp_active_bit_or", IntrinsicOp::WARP_ACTIVE_BIT_OR},
        {"warp_active_bit_xor", IntrinsicOp::WARP_ACTIVE_BIT_XOR},
        {"warp_active_count_bits", IntrinsicOp::WARP_ACTIVE_COUNT_BITS},
        {"warp_active_max", IntrinsicOp::WARP_ACTIVE_MAX},
        {"warp_active_min", IntrinsicOp::WARP_ACTIVE_MIN},
        {"warp_active_product", IntrinsicOp::WARP_ACTIVE_PRODUCT},
        {"warp_active_sum", IntrinsicOp::WARP_ACTIVE_SUM},
        {"warp_active_all", IntrinsicOp::WARP_ACTIVE_ALL},
        {"warp_active_any", IntrinsicOp::WARP_ACTIVE_ANY},
        {"warp_active_bit_mask", IntrinsicOp::WARP_ACTIVE_BIT_MASK},
        {"warp_prefix_count_bits", IntrinsicOp::WARP_PREFIX_COUNT_BITS},
        {"warp_prefix_sum", IntrinsicOp::WARP_PREFIX_SUM},
        {"warp_prefix_product", IntrinsicOp::WARP_PREFIX_PRODUCT},
        {"warp_read_lane", IntrinsicOp::WARP_READ_LANE},
        {"warp_read_first_active_lane", IntrinsicOp::WARP_READ_FIRST_ACTIVE_LANE},
        {"indirect_set_dispatch_kernel", IntrinsicOp::INDIRECT_SET_DISPATCH_KERNEL},
        {"indirect_set_dispatch_count", IntrinsicOp::INDIRECT_SET_DISPATCH_COUNT},
        {"texture2d_sample", IntrinsicOp::TEXTURE2D_SAMPLE},
        {"texture2d_sample_level", IntrinsicOp::TEXTURE2D_SAMPLE_LEVEL},
        {"texture2d_sample_grad", IntrinsicOp::TEXTURE2D_SAMPLE_GRAD},
        {"texture2d_sample_grad_level", IntrinsicOp::TEXTURE2D_SAMPLE_GRAD_LEVEL},
        {"texture3d_sample", IntrinsicOp::TEXTURE3D_SAMPLE},
        {"texture3d_sample_level", IntrinsicOp::TEXTURE3D_SAMPLE_LEVEL},
        {"texture3d_sample_grad", IntrinsicOp::TEXTURE3D_SAMPLE_GRAD},
        {"texture3d_sample_grad_level", IntrinsicOp::TEXTURE3D_SAMPLE_GRAD_LEVEL},
        {"shader_execution_reorder", IntrinsicOp::SHADER_EXECUTION_REORDER},
    };
    auto iter = m.find(name);
    LUISA_ASSERT(iter != m.end(), "Unknown intrinsic operation: {}.", name);
    return iter->second;
}

}// namespace luisa::compute::xir
