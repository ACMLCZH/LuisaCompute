#include <luisa/xir/metadata/comment.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/external_function.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/translators/ast2xir.h>

namespace luisa::compute::xir {

class AST2XIRContext {

public:
    struct BreakContinueTarget {
        BasicBlock *break_target{nullptr};
        BasicBlock *continue_target{nullptr};
    };

    struct Current {
        FunctionDefinition *f{nullptr};
        const ASTFunction *ast{nullptr};
        BreakContinueTarget break_continue_target;
        luisa::unordered_map<Variable, Value *> variables;
        luisa::vector<const CommentStmt *> comments;
    };

    struct TypedLiteral {
        const Type *type;
        LiteralExpr::Value value;

        [[nodiscard]] bool operator==(const TypedLiteral &rhs) const noexcept {
            return this->type == rhs.type &&
                   this->value.index() == rhs.value.index() &&
                   luisa::visit(
                       [&rhs_variant = rhs.value]<typename T>(T lhs) noexcept -> bool {
                           auto rhs = luisa::get<T>(rhs_variant);
                           if constexpr (luisa::is_scalar_v<T>) {
                               return lhs == rhs;
                           } else if constexpr (luisa::is_vector_v<T>) {
                               return luisa::all(lhs == rhs);
                           } else if constexpr (luisa::is_matrix_v<T>) {
                               for (auto i = 0u; i < luisa::matrix_dimension_v<T>; i++) {
                                   if (!all(lhs[i] == rhs[i])) { return false; }
                               }
                               return true;
                           }
                           LUISA_ERROR_WITH_LOCATION("Unexpected literal type.");
                       },
                       this->value);
        }

        [[nodiscard]] uint64_t hash() const noexcept {
            auto hv = luisa::visit(
                []<typename T>(T v) noexcept {
                    return hash_value(v);
                },
                value);
            return hash_combine({type->hash(), hv});
        }
    };

private:
    AST2XIRConfig _config;
    Module *_module;
    luisa::unordered_map<uint64_t, Function *> _generated_functions;
    luisa::unordered_map<ConstantData, Constant *> _generated_constants;
    luisa::unordered_map<TypedLiteral, Constant *> _generated_literals;
    Current _current;

private:
    [[nodiscard]] Value *_translate_unary_expr(Builder &b, const UnaryExpr *expr) noexcept {
        auto operand = _translate_expression(b, expr->operand(), true);
        auto op = [unary_op = expr->op()] {
            switch (unary_op) {
                case UnaryOp::PLUS: return IntrinsicOp::UNARY_PLUS;
                case UnaryOp::MINUS: return IntrinsicOp::UNARY_MINUS;
                case UnaryOp::NOT: return IntrinsicOp::UNARY_NOT;
                case UnaryOp::BIT_NOT: return IntrinsicOp::UNARY_BIT_NOT;
            }
            LUISA_ERROR_WITH_LOCATION("Unexpected unary operation.");
        }();
        return b.call(expr->type(), op, {operand});
    }

    [[nodiscard]] Value *_translate_binary_expr(Builder &b, const BinaryExpr *expr) noexcept {
        auto op = [binary_op = expr->op()] {
            switch (binary_op) {
                case BinaryOp::ADD: return IntrinsicOp::BINARY_ADD;
                case BinaryOp::SUB: return IntrinsicOp::BINARY_SUB;
                case BinaryOp::MUL: return IntrinsicOp::BINARY_MUL;
                case BinaryOp::DIV: return IntrinsicOp::BINARY_DIV;
                case BinaryOp::MOD: return IntrinsicOp::BINARY_MOD;
                case BinaryOp::BIT_AND: return IntrinsicOp::BINARY_BIT_AND;
                case BinaryOp::BIT_OR: return IntrinsicOp::BINARY_BIT_OR;
                case BinaryOp::BIT_XOR: return IntrinsicOp::BINARY_BIT_XOR;
                case BinaryOp::SHL: return IntrinsicOp::BINARY_SHIFT_LEFT;
                case BinaryOp::SHR: return IntrinsicOp::BINARY_SHIFT_RIGHT;
                case BinaryOp::AND: return IntrinsicOp::BINARY_AND;
                case BinaryOp::OR: return IntrinsicOp::BINARY_OR;
                case BinaryOp::LESS: return IntrinsicOp::BINARY_LESS;
                case BinaryOp::GREATER: return IntrinsicOp::BINARY_GREATER;
                case BinaryOp::LESS_EQUAL: return IntrinsicOp::BINARY_LESS_EQUAL;
                case BinaryOp::GREATER_EQUAL: return IntrinsicOp::BINARY_GREATER_EQUAL;
                case BinaryOp::EQUAL: return IntrinsicOp::BINARY_EQUAL;
                case BinaryOp::NOT_EQUAL: return IntrinsicOp::BINARY_NOT_EQUAL;
            }
            LUISA_ERROR_WITH_LOCATION("Unexpected binary operation.");
        }();
        auto type_promotion = promote_types(expr->op(), expr->lhs()->type(), expr->rhs()->type());
        auto lhs = _translate_expression(b, expr->lhs(), true);
        auto rhs = _translate_expression(b, expr->rhs(), true);
        lhs = b.static_cast_if_necessary(type_promotion.lhs, lhs);
        rhs = b.static_cast_if_necessary(type_promotion.rhs, rhs);
        auto result = b.call(expr->type(), op, {lhs, rhs});
        return b.static_cast_if_necessary(type_promotion.result, result);
    }

    [[nodiscard]] Value *_translate_constant_access_index(uint i) noexcept {
        auto key = TypedLiteral{Type::of<uint>(), LiteralExpr::Value{i}};
        return _translate_typed_literal(key);
    }

    [[nodiscard]] Value *_collect_access_indices(Builder &b, const Expression *expr, luisa::fixed_vector<Value *, 16u> &rev_indices) noexcept {
        switch (expr->tag()) {
            case Expression::Tag::MEMBER: {
                auto member_expr = static_cast<const MemberExpr *>(expr);
                auto member_index = _translate_constant_access_index(member_expr->member_index());
                rev_indices.emplace_back(member_index);
                return _collect_access_indices(b, member_expr->self(), rev_indices);
            }
            case Expression::Tag::ACCESS: {
                auto access_expr = static_cast<const AccessExpr *>(expr);
                auto index = _translate_expression(b, access_expr->index(), true);
                rev_indices.emplace_back(index);
                return _collect_access_indices(b, access_expr->range(), rev_indices);
            }
            default: break;
        }
        return _translate_expression(b, expr, false);
    }

    [[nodiscard]] Value *_translate_member_or_access_expr(Builder &b, const Expression *expr, bool load_lval) noexcept {
        luisa::fixed_vector<Value *, 16u> args;
        auto base = _collect_access_indices(b, expr, args);
        if (base->is_lvalue()) {
            std::reverse(args.begin(), args.end());
            auto elem = b.gep(expr->type(), base, args);
            return load_lval ? static_cast<Value *>(b.load(expr->type(), elem)) :
                               static_cast<Value *>(elem);
        }
        args.emplace_back(base);
        std::reverse(args.begin(), args.end());
        return b.call(expr->type(), IntrinsicOp::EXTRACT, args);
    }

    [[nodiscard]] Value *_translate_member_expr(Builder &b, const MemberExpr *expr, bool load_lval) noexcept {
        if (expr->is_swizzle()) {
            if (expr->swizzle_size() == 1u) {
                auto v = _translate_expression(b, expr->self(), load_lval);
                auto index = _translate_constant_access_index(expr->swizzle_index(0));
                if (v->is_lvalue()) {
                    LUISA_ASSERT(!load_lval, "Unexpected lvalue swizzle.");
                    return b.gep(expr->type(), v, {index});
                }
                return b.call(expr->type(), IntrinsicOp::EXTRACT, {v, index});
            }
            luisa::fixed_vector<Value *, 5u> args;
            auto v = _translate_expression(b, expr->self(), true);
            args.emplace_back(v);
            for (auto i = 0u; i < expr->swizzle_size(); i++) {
                auto index = expr->swizzle_index(i);
                args.emplace_back(_translate_constant_access_index(index));
            }
            return b.call(expr->type(), IntrinsicOp::SHUFFLE, args);
        }
        return _translate_member_or_access_expr(b, expr, load_lval);
    }

    [[nodiscard]] Value *_translate_typed_literal(TypedLiteral key) noexcept {
        auto [iter, just_inserted] = _generated_literals.try_emplace(key, nullptr);
        if (just_inserted) {
            iter->second = luisa::visit(
                [this, t = key.type]<typename T>(T v) noexcept {
                    LUISA_ASSERT(t == Type::of<T>(), "Literal type mismatch.");
                    return _module->create_constant(v);
                },
                key.value);
        }
        return iter->second;
    }

    [[nodiscard]] Value *_translate_literal_expr(const LiteralExpr *expr) noexcept {
        auto key = TypedLiteral{expr->type(), expr->value()};
        return _translate_typed_literal(key);
    }

    [[nodiscard]] Value *_translate_builtin_variable(Builder &b, Variable ast_var) noexcept {
        LUISA_ASSERT(ast_var.is_builtin(), "Unresolved variable reference.");
        auto op = [tag = ast_var.tag(), t = ast_var.type()] {
            switch (tag) {
                case Variable::Tag::THREAD_ID:
                    LUISA_ASSERT(t == Type::of<uint3>(), "Invalid thread_id type: {}", t->description());
                    return IntrinsicOp::THREAD_ID;
                case Variable::Tag::BLOCK_ID:
                    LUISA_ASSERT(t == Type::of<uint3>(), "Invalid block_id type: {}.", t->description());
                    return IntrinsicOp::BLOCK_ID;
                case Variable::Tag::DISPATCH_ID:
                    LUISA_ASSERT(t == Type::of<uint3>(), "Invalid dispatch_id type: {}", t->description());
                    return IntrinsicOp::DISPATCH_ID;
                case Variable::Tag::DISPATCH_SIZE:
                    LUISA_ASSERT(t == Type::of<uint3>(), "Invalid dispatch_size type: {}", t->description());
                    return IntrinsicOp::DISPATCH_SIZE;
                case Variable::Tag::KERNEL_ID:
                    LUISA_ASSERT(t == Type::of<uint>(), "Invalid kernel_id type: {}", t->description());
                    return IntrinsicOp::KERNEL_ID;
                case Variable::Tag::WARP_LANE_COUNT:
                    LUISA_ASSERT(t == Type::of<uint>(), "Invalid warp_size type: {}", t->description());
                    return IntrinsicOp::WARP_SIZE;
                case Variable::Tag::WARP_LANE_ID:
                    LUISA_ASSERT(t == Type::of<uint>(), "Invalid warp_lane_id type: {}", t->description());
                    return IntrinsicOp::WARP_LANE_ID;
                case Variable::Tag::OBJECT_ID:
                    LUISA_ASSERT(t == Type::of<uint>(), "Invalid object_id type: {}", t->description());
                    return IntrinsicOp::OBJECT_ID;
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Unexpected variable type.");
        }();
        return b.call(ast_var.type(), op, {});
    }

    [[nodiscard]] Value *_translate_ref_expr(Builder &b, const RefExpr *expr, bool load_lval) noexcept {
        auto ast_var = expr->variable();
        LUISA_ASSERT(ast_var.type() == expr->type(), "Variable type mismatch.");
        if (auto iter = _current.variables.find(ast_var); iter != _current.variables.end()) {
            auto var = iter->second;
            return load_lval && var->is_lvalue() ? b.load(expr->type(), var) : var;
        }
        return _translate_builtin_variable(b, ast_var);
    }

    [[nodiscard]] Value *_translate_constant_expr(const ConstantExpr *expr) noexcept {
        auto c = expr->data();
        auto [iter, just_inserted] = _generated_constants.try_emplace(c, nullptr);
        if (just_inserted) {
            iter->second = _module->create_constant(c.type(), c.raw());
        }
        return iter->second;
    }

    [[nodiscard]] Value *_translate_call_expr(Builder &b, const CallExpr *expr) noexcept {
        if (expr->is_external()) {
            auto ast = expr->external();
            auto f = add_external_function(*ast);
            LUISA_NOT_IMPLEMENTED();
        }
        if (expr->is_custom()) {
            auto ast = expr->custom();
            auto f = add_function(ast);
            LUISA_ASSERT(f->type() == expr->type(), "Function return type mismatch.");
            auto ast_args = expr->arguments();
            luisa::fixed_vector<Value *, 16u> args;
            args.reserve(expr->arguments().size());
            for (auto i = 0u; i < ast_args.size(); i++) {
                auto by_ref = ast.arguments()[i].is_reference();
                auto arg = _translate_expression(b, ast_args[i], !by_ref);
                args.emplace_back(arg);
            }
            return b.call(f->type(), f, args);
        }
        // builtin function
        switch (expr->op()) {
            case CallOp::CUSTOM: LUISA_ERROR_WITH_LOCATION("Unexpected custom call operation.");
            case CallOp::EXTERNAL: LUISA_ERROR_WITH_LOCATION("Unexpected external call operation.");
            case CallOp::ALL: break;
            case CallOp::ANY: break;
            case CallOp::SELECT: break;
            case CallOp::CLAMP: break;
            case CallOp::SATURATE: break;
            case CallOp::LERP: break;
            case CallOp::SMOOTHSTEP: break;
            case CallOp::STEP: break;
            case CallOp::ABS: break;
            case CallOp::MIN: break;
            case CallOp::MAX: break;
            case CallOp::CLZ: break;
            case CallOp::CTZ: break;
            case CallOp::POPCOUNT: break;
            case CallOp::REVERSE: break;
            case CallOp::ISINF: break;
            case CallOp::ISNAN: break;
            case CallOp::ACOS: break;
            case CallOp::ACOSH: break;
            case CallOp::ASIN: break;
            case CallOp::ASINH: break;
            case CallOp::ATAN: break;
            case CallOp::ATAN2: break;
            case CallOp::ATANH: break;
            case CallOp::COS: break;
            case CallOp::COSH: break;
            case CallOp::SIN: break;
            case CallOp::SINH: break;
            case CallOp::TAN: break;
            case CallOp::TANH: break;
            case CallOp::EXP: break;
            case CallOp::EXP2: break;
            case CallOp::EXP10: break;
            case CallOp::LOG: break;
            case CallOp::LOG2: break;
            case CallOp::LOG10: break;
            case CallOp::POW: break;
            case CallOp::SQRT: break;
            case CallOp::RSQRT: break;
            case CallOp::CEIL: break;
            case CallOp::FLOOR: break;
            case CallOp::FRACT: break;
            case CallOp::TRUNC: break;
            case CallOp::ROUND: break;
            case CallOp::FMA: break;
            case CallOp::COPYSIGN: break;
            case CallOp::CROSS: break;
            case CallOp::DOT: break;
            case CallOp::LENGTH: break;
            case CallOp::LENGTH_SQUARED: break;
            case CallOp::NORMALIZE: break;
            case CallOp::FACEFORWARD: break;
            case CallOp::REFLECT: break;
            case CallOp::REDUCE_SUM: break;
            case CallOp::REDUCE_PRODUCT: break;
            case CallOp::REDUCE_MIN: break;
            case CallOp::REDUCE_MAX: break;
            case CallOp::OUTER_PRODUCT: break;
            case CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION: break;
            case CallOp::DETERMINANT: break;
            case CallOp::TRANSPOSE: break;
            case CallOp::INVERSE: break;
            case CallOp::SYNCHRONIZE_BLOCK: break;
            case CallOp::ATOMIC_EXCHANGE: break;
            case CallOp::ATOMIC_COMPARE_EXCHANGE: break;
            case CallOp::ATOMIC_FETCH_ADD: break;
            case CallOp::ATOMIC_FETCH_SUB: break;
            case CallOp::ATOMIC_FETCH_AND: break;
            case CallOp::ATOMIC_FETCH_OR: break;
            case CallOp::ATOMIC_FETCH_XOR: break;
            case CallOp::ATOMIC_FETCH_MIN: break;
            case CallOp::ATOMIC_FETCH_MAX: break;
            case CallOp::ADDRESS_OF: break;
            case CallOp::BUFFER_READ: {
                auto buffer = _translate_expression(b, expr->arguments()[0], true);
                auto index = _translate_expression(b, expr->arguments()[1], true);
                return b.call(expr->type(), IntrinsicOp::BUFFER_READ, {buffer, index});
            }
            case CallOp::BUFFER_WRITE: {
                auto buffer = _translate_expression(b, expr->arguments()[0], true);
                auto index = _translate_expression(b, expr->arguments()[1], true);
                auto value = _translate_expression(b, expr->arguments()[2], true);
                return b.call(expr->type(), IntrinsicOp::BUFFER_WRITE, {buffer, index, value});
            }
            case CallOp::BUFFER_SIZE: break;
            case CallOp::BUFFER_ADDRESS: break;
            case CallOp::BYTE_BUFFER_READ: break;
            case CallOp::BYTE_BUFFER_WRITE: break;
            case CallOp::BYTE_BUFFER_SIZE: break;
            case CallOp::TEXTURE_READ: break;
            case CallOp::TEXTURE_WRITE: break;
            case CallOp::TEXTURE_SIZE: break;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE: break;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: break;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: break;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE: break;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: break;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: break;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: break;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: break;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: break;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: break;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: break;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: break;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
            case CallOp::BINDLESS_TEXTURE2D_READ: break;
            case CallOp::BINDLESS_TEXTURE3D_READ: break;
            case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: break;
            case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: break;
            case CallOp::BINDLESS_TEXTURE2D_SIZE: break;
            case CallOp::BINDLESS_TEXTURE3D_SIZE: break;
            case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: break;
            case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: break;
            case CallOp::BINDLESS_BUFFER_READ: break;
            case CallOp::BINDLESS_BUFFER_WRITE: break;
            case CallOp::BINDLESS_BYTE_BUFFER_READ: break;
            case CallOp::BINDLESS_BUFFER_SIZE: break;
            case CallOp::BINDLESS_BUFFER_TYPE: break;
            case CallOp::BINDLESS_BUFFER_ADDRESS: break;
            case CallOp::MAKE_BOOL2: break;
            case CallOp::MAKE_BOOL3: break;
            case CallOp::MAKE_BOOL4: break;
            case CallOp::MAKE_INT2: break;
            case CallOp::MAKE_INT3: break;
            case CallOp::MAKE_INT4: break;
            case CallOp::MAKE_UINT2: break;
            case CallOp::MAKE_UINT3: break;
            case CallOp::MAKE_UINT4: break;
            case CallOp::MAKE_FLOAT2: break;
            case CallOp::MAKE_FLOAT3: break;
            case CallOp::MAKE_FLOAT4: break;
            case CallOp::MAKE_SHORT2: break;
            case CallOp::MAKE_SHORT3: break;
            case CallOp::MAKE_SHORT4: break;
            case CallOp::MAKE_USHORT2: break;
            case CallOp::MAKE_USHORT3: break;
            case CallOp::MAKE_USHORT4: break;
            case CallOp::MAKE_LONG2: break;
            case CallOp::MAKE_LONG3: break;
            case CallOp::MAKE_LONG4: break;
            case CallOp::MAKE_ULONG2: break;
            case CallOp::MAKE_ULONG3: break;
            case CallOp::MAKE_ULONG4: break;
            case CallOp::MAKE_HALF2: break;
            case CallOp::MAKE_HALF3: break;
            case CallOp::MAKE_HALF4: break;
            case CallOp::MAKE_DOUBLE2: break;
            case CallOp::MAKE_DOUBLE3: break;
            case CallOp::MAKE_DOUBLE4: break;
            case CallOp::MAKE_BYTE2: break;
            case CallOp::MAKE_BYTE3: break;
            case CallOp::MAKE_BYTE4: break;
            case CallOp::MAKE_UBYTE2: break;
            case CallOp::MAKE_UBYTE3: break;
            case CallOp::MAKE_UBYTE4: break;
            case CallOp::MAKE_FLOAT2X2: break;
            case CallOp::MAKE_FLOAT3X3: break;
            case CallOp::MAKE_FLOAT4X4: break;
            case CallOp::ASSERT: break;
            case CallOp::ASSUME: break;
            case CallOp::UNREACHABLE: break;
            case CallOp::ZERO: break;
            case CallOp::ONE: break;
            case CallOp::PACK: break;
            case CallOp::UNPACK: break;
            case CallOp::REQUIRES_GRADIENT: break;
            case CallOp::GRADIENT: break;
            case CallOp::GRADIENT_MARKER: break;
            case CallOp::ACCUMULATE_GRADIENT: break;
            case CallOp::BACKWARD: break;
            case CallOp::DETACH: break;
            case CallOp::RAY_TRACING_INSTANCE_TRANSFORM: break;
            case CallOp::RAY_TRACING_INSTANCE_USER_ID: break;
            case CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: break;
            case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: break;
            case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY: break;
            case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY: break;
            case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID: break;
            case CallOp::RAY_TRACING_TRACE_CLOSEST: break;
            case CallOp::RAY_TRACING_TRACE_ANY: break;
            case CallOp::RAY_TRACING_QUERY_ALL: break;
            case CallOp::RAY_TRACING_QUERY_ANY: break;
            case CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: break;
            case CallOp::RAY_TRACING_INSTANCE_MOTION_SRT: break;
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: break;
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: break;
            case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: break;
            case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: break;
            case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: break;
            case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: break;
            case CallOp::RAY_QUERY_WORLD_SPACE_RAY: break;
            case CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT: break;
            case CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT: break;
            case CallOp::RAY_QUERY_COMMITTED_HIT: break;
            case CallOp::RAY_QUERY_COMMIT_TRIANGLE: break;
            case CallOp::RAY_QUERY_COMMIT_PROCEDURAL: break;
            case CallOp::RAY_QUERY_TERMINATE: break;
            case CallOp::RAY_QUERY_PROCEED: break;
            case CallOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE: break;
            case CallOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE: break;
            case CallOp::RASTER_DISCARD: break;
            case CallOp::DDX: break;
            case CallOp::DDY: break;
            case CallOp::WARP_IS_FIRST_ACTIVE_LANE: break;
            case CallOp::WARP_FIRST_ACTIVE_LANE: break;
            case CallOp::WARP_ACTIVE_ALL_EQUAL: break;
            case CallOp::WARP_ACTIVE_BIT_AND: break;
            case CallOp::WARP_ACTIVE_BIT_OR: break;
            case CallOp::WARP_ACTIVE_BIT_XOR: break;
            case CallOp::WARP_ACTIVE_COUNT_BITS: break;
            case CallOp::WARP_ACTIVE_MAX: break;
            case CallOp::WARP_ACTIVE_MIN: break;
            case CallOp::WARP_ACTIVE_PRODUCT: break;
            case CallOp::WARP_ACTIVE_SUM: break;
            case CallOp::WARP_ACTIVE_ALL: break;
            case CallOp::WARP_ACTIVE_ANY: break;
            case CallOp::WARP_ACTIVE_BIT_MASK: break;
            case CallOp::WARP_PREFIX_COUNT_BITS: break;
            case CallOp::WARP_PREFIX_SUM: break;
            case CallOp::WARP_PREFIX_PRODUCT: break;
            case CallOp::WARP_READ_LANE: break;
            case CallOp::WARP_READ_FIRST_ACTIVE_LANE: break;
            case CallOp::INDIRECT_SET_DISPATCH_KERNEL: break;
            case CallOp::INDIRECT_SET_DISPATCH_COUNT: break;
            case CallOp::TEXTURE2D_SAMPLE: break;
            case CallOp::TEXTURE2D_SAMPLE_LEVEL: break;
            case CallOp::TEXTURE2D_SAMPLE_GRAD: break;
            case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
            case CallOp::TEXTURE3D_SAMPLE: break;
            case CallOp::TEXTURE3D_SAMPLE_LEVEL: break;
            case CallOp::TEXTURE3D_SAMPLE_GRAD: break;
            case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
            case CallOp::SHADER_EXECUTION_REORDER: break;
        }
        LUISA_NOT_IMPLEMENTED();
    }

    [[nodiscard]] Value *_translate_cast_expr(Builder &b, const CastExpr *expr) noexcept {
        auto value = _translate_expression(b, expr->expression(), true);
        switch (expr->op()) {
            case compute::CastOp::STATIC: return b.static_cast_if_necessary(expr->type(), value);
            case compute::CastOp::BITWISE: return b.bit_cast_if_necessary(expr->type(), value);
        }
        LUISA_ERROR_WITH_LOCATION("Unexpected cast operation.");
    }

    [[nodiscard]] Value *_translate_expression(Builder &b, const Expression *expr, bool load_lval) noexcept {
        LUISA_ASSERT(expr != nullptr, "Expression must not be null.");
        switch (expr->tag()) {
            case Expression::Tag::UNARY: return _translate_unary_expr(b, static_cast<const UnaryExpr *>(expr));
            case Expression::Tag::BINARY: return _translate_binary_expr(b, static_cast<const BinaryExpr *>(expr));
            case Expression::Tag::MEMBER: return _translate_member_expr(b, static_cast<const MemberExpr *>(expr), load_lval);
            case Expression::Tag::ACCESS: return _translate_member_or_access_expr(b, expr, load_lval);
            case Expression::Tag::LITERAL: return _translate_literal_expr(static_cast<const LiteralExpr *>(expr));
            case Expression::Tag::REF: return _translate_ref_expr(b, static_cast<const RefExpr *>(expr), load_lval);
            case Expression::Tag::CONSTANT: return _translate_constant_expr(static_cast<const ConstantExpr *>(expr));
            case Expression::Tag::CALL: return _translate_call_expr(b, static_cast<const CallExpr *>(expr));
            case Expression::Tag::CAST: return _translate_cast_expr(b, static_cast<const CastExpr *>(expr));
            case Expression::Tag::TYPE_ID: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::STRING_ID: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::FUNC_REF: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::CPUCUSTOM: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::GPUCUSTOM: LUISA_NOT_IMPLEMENTED();
        }
        return nullptr;
    }

    template<typename T>
    auto _commented(T inst) noexcept {
        for (auto comment : _current.comments) {
            inst->add_comment(comment->comment());
        }
        _current.comments.clear();
        return inst;
    }

    void _collect_comment(const Statement *stmt) noexcept {
        LUISA_ASSERT(stmt->tag() == Statement::Tag::COMMENT, "Unexpected statement type.");
        _current.comments.emplace_back(static_cast<const CommentStmt *>(stmt));
    }

    void _translate_switch_stmt(Builder &b, const SwitchStmt *ast_switch, luisa::span<const Statement *const> cdr) noexcept {
        // we do not support break/continue in switch statement
        auto old_break_continue_target = std::exchange(_current.break_continue_target, {});
        auto value = _translate_expression(b, ast_switch->expression(), true);
        auto inst = _commented(b.switch_(value));
        auto merge_block = inst->create_merge_block();
        auto case_break_removed = [](auto stmt_span) noexcept {
            while (!stmt_span.empty() &&
                   (stmt_span.back()->tag() == Statement::Tag::BREAK ||
                    stmt_span.back()->tag() == Statement::Tag::COMMENT)) {
                stmt_span = stmt_span.subspan(0, stmt_span.size() - 1);
            }
            return stmt_span;
        };
        for (auto s : ast_switch->body()->statements()) {
            switch (s->tag()) {
                case Statement::Tag::SWITCH_CASE: {
                    auto ast_case = static_cast<const SwitchCaseStmt *>(s);
                    LUISA_ASSERT(ast_case->expression()->tag() == Expression::Tag::LITERAL,
                                 "Unexpected switch case expression.");
                    auto ast_literal = static_cast<const LiteralExpr *>(ast_case->expression());
                    auto case_value = luisa::visit(
                        []<typename T>(T x) noexcept -> SwitchInst::case_value_type {
                            if constexpr (std::is_integral_v<T>) {
                                return static_cast<SwitchInst::case_value_type>(x);
                            } else {
                                LUISA_ERROR_WITH_LOCATION("Unexpected literal integer in switch case.");
                            }
                        },
                        ast_literal->value());
                    auto case_block = _commented(inst->create_case_block(case_value));
                    b.set_insertion_point(case_block);
                    auto case_stmts = case_break_removed(ast_case->body()->statements());
                    _translate_statements(b, case_stmts);
                    if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
                    break;
                }
                case Statement::Tag::SWITCH_DEFAULT: {
                    LUISA_ASSERT(inst->default_block() == nullptr,
                                 "Multiple default blocks in a switch statement.");
                    auto default_block = inst->create_default_block();
                    b.set_insertion_point(default_block);
                    auto ast_default = static_cast<const SwitchDefaultStmt *>(s);
                    auto case_stmts = case_break_removed(ast_default->body()->statements());
                    _translate_statements(b, case_stmts);
                    if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
                    break;
                }
                case Statement::Tag::COMMENT: {
                    _collect_comment(s);
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION("Unexpected statement in switch body.");
            }
        }
        if (inst->default_block() == nullptr) {
            b.set_insertion_point(inst->create_default_block());
            b.br(merge_block);
        }
        _current.break_continue_target = old_break_continue_target;
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_if_stmt(Builder &b, const IfStmt *ast_if, luisa::span<const Statement *const> cdr) noexcept {
        auto cond = _translate_expression(b, ast_if->condition(), true);
        auto inst = _commented(b.if_(cond));
        auto merge_block = inst->create_merge_block();
        // true branch
        {
            b.set_insertion_point(inst->create_true_block());
            _translate_statements(b, ast_if->true_branch()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        // false branch
        {
            b.set_insertion_point(inst->create_false_block());
            _translate_statements(b, ast_if->false_branch()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        // merge block
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_loop_stmt(Builder &b, const LoopStmt *ast_loop, luisa::span<const Statement *const> cdr) noexcept {
        auto inst = _commented(b.simple_loop());
        auto merge_block = inst->create_merge_block();
        auto body_block = inst->create_body_block();
        auto old_break_continue_target = std::exchange(_current.break_continue_target, {});
        // body block
        {
            _current.break_continue_target = {.break_target = merge_block,
                                              .continue_target = body_block};
            b.set_insertion_point(body_block);
            _translate_statements(b, ast_loop->body()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(inst->body_block()); }
            _current.break_continue_target = {};
        }
        // merge block
        _current.break_continue_target = old_break_continue_target;
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_for_stmt(Builder &b, const ForStmt *ast_for, luisa::span<const Statement *const> cdr) noexcept {
        auto var = _translate_expression(b, ast_for->variable(), false);
        auto inst = _commented(b.loop());
        auto merge_block = inst->create_merge_block();
        auto prepare_block = inst->create_prepare_block();
        auto body_block = inst->create_body_block();
        auto update_block = inst->create_update_block();
        auto old_break_continue_target = std::exchange(_current.break_continue_target, {});
        // prepare block
        {
            b.set_insertion_point(prepare_block);
            auto cond = _translate_expression(b, ast_for->condition(), true);
            b.cond_br(cond, body_block, merge_block);
        }
        // body block
        {
            _current.break_continue_target = {.break_target = merge_block,
                                              .continue_target = update_block};
            b.set_insertion_point(body_block);
            _translate_statements(b, ast_for->body()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(update_block); }
            _current.break_continue_target = {};
        }
        // update block
        {
            b.set_insertion_point(update_block);
            auto t = ast_for->variable()->type();
            // var += step
            auto step = _translate_expression(b, ast_for->step(), true);
            auto cast_step = b.static_cast_if_necessary(t, step);
            auto prev = b.load(t, var);
            auto next = b.call(t, IntrinsicOp::BINARY_ADD, {prev, cast_step});
            b.store(var, next);
            // jump back to prepare block
            b.br(prepare_block);
        }
        // merge block
        _current.break_continue_target = old_break_continue_target;
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_ray_query_stmt(Builder &b, const RayQueryStmt *ast_ray_query, luisa::span<const Statement *const> cdr) noexcept {
        // we do not support break/continue in ray query statement
        auto old_break_continue_target = std::exchange(_current.break_continue_target, {});
        auto query_object = _translate_expression(b, ast_ray_query->query(), false);
        auto inst = _commented(b.ray_query(query_object));
        auto merge_block = inst->create_merge_block();
        // on surface candidate block
        {
            b.set_insertion_point(inst->create_on_surface_candidate_block());
            _translate_statements(b, ast_ray_query->on_triangle_candidate()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        // on procedural candidate block
        {
            b.set_insertion_point(inst->create_on_procedural_candidate_block());
            _translate_statements(b, ast_ray_query->on_procedural_candidate()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        // merge block
        _current.break_continue_target = old_break_continue_target;
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_statements(Builder &b, luisa::span<const Statement *const> stmts) noexcept {
        if (stmts.empty()) { return; }
        auto car = stmts.front();
        auto cdr = stmts.subspan(1);
        switch (car->tag()) {
            case Statement::Tag::BREAK: {
                auto break_target = _current.break_continue_target.break_target;
                LUISA_ASSERT(break_target != nullptr, "Invalid break statement.");
                _commented(b.break_(break_target));
                break;
            }
            case Statement::Tag::CONTINUE: {
                auto continue_target = _current.break_continue_target.continue_target;
                LUISA_ASSERT(continue_target != nullptr, "Invalid continue statement.");
                _commented(b.continue_(continue_target));
                break;
            }
            case Statement::Tag::RETURN: {
                if (auto ast_expr = static_cast<const ReturnStmt *>(car)->expression()) {
                    auto value = _translate_expression(b, ast_expr, true);
                    _commented(b.return_(value));
                } else {
                    _commented(b.return_void());
                }
                break;
            }
            case Statement::Tag::SCOPE: LUISA_ERROR_WITH_LOCATION("Unexpected scope statement.");
            case Statement::Tag::IF: {
                auto ast_if = static_cast<const IfStmt *>(car);
                _translate_if_stmt(b, ast_if, cdr);
                break;
            }
            case Statement::Tag::LOOP: {
                auto ast_loop = static_cast<const LoopStmt *>(car);
                _translate_loop_stmt(b, ast_loop, cdr);
                break;
            }
            case Statement::Tag::EXPR: {
                auto ast_expr = static_cast<const ExprStmt *>(car)->expression();
                _commented(_translate_expression(b, ast_expr, false));
                _translate_statements(b, cdr);
                break;
            }
            case Statement::Tag::SWITCH: {
                auto ast_switch = static_cast<const SwitchStmt *>(car);
                _translate_switch_stmt(b, ast_switch, cdr);
                break;
            }
            case Statement::Tag::SWITCH_CASE: LUISA_ERROR_WITH_LOCATION("Unexpected switch case statement.");
            case Statement::Tag::SWITCH_DEFAULT: LUISA_ERROR_WITH_LOCATION("Unexpected switch default statement.");
            case Statement::Tag::ASSIGN: {
                auto assign = static_cast<const AssignStmt *>(car);
                auto variable = _translate_expression(b, assign->lhs(), false);
                auto value = _translate_expression(b, assign->rhs(), true);
                _commented(b.store(variable, value));
                _translate_statements(b, cdr);
                break;
            }
            case Statement::Tag::FOR: {
                auto ast_for = static_cast<const ForStmt *>(car);
                _translate_for_stmt(b, ast_for, cdr);
                break;
            }
            case Statement::Tag::COMMENT: {
                _collect_comment(car);
                _translate_statements(b, cdr);
                break;
            }
            case Statement::Tag::RAY_QUERY: {
                auto ast_ray_query = static_cast<const RayQueryStmt *>(car);
                _translate_ray_query_stmt(b, ast_ray_query, cdr);
                break;
            }
            case Statement::Tag::AUTO_DIFF: LUISA_NOT_IMPLEMENTED();
            case Statement::Tag::PRINT: {
                auto ast_print = static_cast<const PrintStmt *>(car);
                luisa::fixed_vector<Value *, 16u> args;
                for (auto ast_arg : ast_print->arguments()) {
                    args.emplace_back(_translate_expression(b, ast_arg, true));
                }
                _commented(b.print(luisa::string{ast_print->format()}, args));
                _translate_statements(b, cdr);
                break;
            }
        }
    }

    void _translate_current_function() noexcept {
        // convert the arguments
        for (auto ast_arg : _current.ast->arguments()) {
            auto arg = _current.f->create_argument(ast_arg.type(), ast_arg.is_reference());
            _current.variables.emplace(ast_arg, arg);
        }
        // create the body block
        Builder b;
        b.set_insertion_point(_current.f->create_body_block());
        for (auto ast_local : _current.ast->local_variables()) {
            LUISA_DEBUG_ASSERT(_current.variables.find(ast_local) == _current.variables.end(),
                               "Local variable already exists.");
            auto v = _current.variables.emplace(ast_local, b.alloca_local(ast_local.type())).first->second;
            if (ast_local.is_builtin()) {
                auto builtin_init = _translate_builtin_variable(b, ast_local);
                LUISA_ASSERT(v->type() == builtin_init->type(), "Variable type mismatch.");
                b.store(v, builtin_init);
            }
        }
        for (auto ast_shared : _current.ast->shared_variables()) {
            LUISA_DEBUG_ASSERT(_current.variables.find(ast_shared) == _current.variables.end(),
                               "Shared variable already exists.");
            _current.variables.emplace(ast_shared, b.alloca_shared(ast_shared.type()));
        }
        _translate_statements(b, _current.ast->body()->statements());
        if (!b.is_insertion_point_terminator()) {
            LUISA_ASSERT(_current.f->type() == nullptr,
                         "Non-void function must have a return statement at the end.");
            b.return_void();
        }
    }

public:
    explicit AST2XIRContext(const AST2XIRConfig &config) noexcept
        : _config{config}, _module{Pool::current()->create<Module>()} {}

    Function *add_function(const ASTFunction &f) noexcept {
        LUISA_ASSERT(_module != nullptr, "Module has been finalized.");
        // try emplace the function
        auto [iter, just_inserted] = _generated_functions.try_emplace(f.hash(), nullptr);
        // return the function if it has been translated
        if (!just_inserted) { return iter->second; }
        // create a new function
        FunctionDefinition *def = nullptr;
        switch (f.tag()) {
            case ASTFunction::Tag::KERNEL:
                def = _module->create_kernel();
                break;
            case ASTFunction::Tag::CALLABLE:
                def = _module->create_callable(f.return_type());
                break;
            case ASTFunction::Tag::RASTER_STAGE:
                LUISA_NOT_IMPLEMENTED();
        }
        iter->second = def;
        // translate the function
        auto old = std::exchange(_current, {.f = def, .ast = &f});
        _translate_current_function();
        _current = old;
        return def;
    }

    Function *add_external_function(const ASTExternalFunction &f) noexcept {
        LUISA_ASSERT(_module != nullptr, "Module has been finalized.");
        LUISA_NOT_IMPLEMENTED();
    }

    [[nodiscard]] Module *finalize() noexcept {
        auto module = std::exchange(_module, nullptr);
        return module;
    }
};

AST2XIRContext *ast_to_xir_translate_begin(const AST2XIRConfig &config) noexcept {
    return luisa::new_with_allocator<AST2XIRContext>(config);
}

void ast_to_xir_translate_add_function(AST2XIRContext *ctx, const ASTFunction &f) noexcept {
    ctx->add_function(f);
}

void ast_to_xir_translate_add_external_function(AST2XIRContext *ctx, const ASTExternalFunction &f) noexcept {
    ctx->add_external_function(f);
}

Module *ast_to_xir_translate_finalize(AST2XIRContext *ctx) noexcept {
    auto m = ctx->finalize();
    luisa::delete_with_allocator(ctx);
    return m;
}

Module *ast_to_xir_translate(const ASTFunction &kernel, const AST2XIRConfig &config) noexcept {
    AST2XIRContext ctx{config};
    ctx.add_function(kernel);
    return ctx.finalize();
}

}// namespace luisa::compute::xir
