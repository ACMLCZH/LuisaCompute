//
// Created by Mike on 10/12/2023.
//

#include <luisa/ast/function_builder.h>
#include <luisa/core/logging.h>
namespace luisa::compute::detail {

class FunctionDuplicator {

private:
    struct DupCtx {
        const FunctionBuilder &original;
        luisa::unordered_map<
            uint32_t /* original uid */,
            const RefExpr * /* copy */>
            var_map;
        luisa::unordered_map<
            const Expression * /* original */,
            const Expression * /* copy */>
            expr_map;
    };

private:
    luisa::unordered_map<
        const FunctionBuilder * /* original */,
        luisa::shared_ptr<const FunctionBuilder> /* copy */>
        _duplicated;
    luisa::vector<DupCtx *> _contexts;

private:
    void _dup_function(const FunctionBuilder &f) noexcept {
        auto fb = FunctionBuilder::current();
        if (f.tag() == Function::Tag::KERNEL) { fb->set_block_size(f.block_size()); }
        auto dup_arg = [this, fb](Variable original) noexcept {
            auto dup = [&] {
                switch (original.tag()) {
                    case Variable::Tag::REFERENCE: return fb->reference(original.type());
                    case Variable::Tag::BUFFER: return fb->buffer(original.type());
                    case Variable::Tag::TEXTURE: return fb->texture(original.type());
                    case Variable::Tag::BINDLESS_ARRAY: return fb->bindless_array();
                    case Variable::Tag::ACCEL: return fb->accel();
                    default: return fb->argument(original.type());
                }
            }();
            _contexts.back()->var_map.emplace(original.uid(), dup);
        };
        if (f.tag() == Function::Tag::CALLABLE) {// captures are already lowered to arguments
            for (auto &arg : f.arguments()) { dup_arg(arg); }
        } else {
            for (auto i = 0u; i < f.bound_arguments().size(); i++) {
                auto &&a = f.arguments()[i];
                auto &&b = f.bound_arguments()[i];
                auto copy = luisa::visit(
                    [&](auto &&bb) noexcept -> const RefExpr * {
                        using T = std::remove_cvref_t<decltype(bb)>;
                        if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                            return fb->buffer_binding(a.type(), bb.handle, bb.offset, bb.size);
                        } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                            return fb->texture_binding(a.type(), bb.handle, bb.level);
                        } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                            return fb->bindless_array_binding(bb.handle);
                        } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                            return fb->accel_binding(bb.handle);
                        } else {
                            LUISA_ERROR_WITH_LOCATION("Unbound captured argument.");
                        }
                    },
                    b);
                _contexts.back()->var_map.emplace(a.uid(), copy);
            }
            for (auto &arg : f.unbound_arguments()) { dup_arg(arg); }
            for (auto &b : f.builtin_variables()) {
                auto copy = fb->_builtin(b.type(), b.tag());
                _contexts.back()->var_map.emplace(b.uid(), copy);
            }
        }
        for (auto &shared : f.shared_variables()) {
            auto s = fb->shared(shared.type());
            _contexts.back()->var_map.emplace(shared.uid(), s);
        }
        for (auto &local : f.local_variables()) {
            auto l = fb->local(local.type());
            _contexts.back()->var_map.emplace(local.uid(), l);
        }
        _dup_scope(f.body(), fb->body());
    }

    [[nodiscard]] const Expression *_dup_expr(const Expression *original) noexcept {
        if (original == nullptr) { return nullptr; }
        if (auto iter = _contexts.back()->expr_map.find(original);
            iter != _contexts.back()->expr_map.end()) {
            return iter->second;
        }
        auto copy = static_cast<const Expression *>(nullptr);
        auto fb = FunctionBuilder::current();
        switch (original->tag()) {
            case Expression::Tag::UNARY: {
                auto e = static_cast<const UnaryExpr *>(original);
                auto operand = _dup_expr(e->operand());
                copy = fb->unary(e->type(), e->op(), operand);
                break;
            }
            case Expression::Tag::BINARY: {
                auto e = static_cast<const BinaryExpr *>(original);
                auto lhs = _dup_expr(e->lhs());
                auto rhs = _dup_expr(e->rhs());
                copy = fb->binary(e->type(), e->op(), lhs, rhs);
                break;
            }
            case Expression::Tag::MEMBER: {
                auto e = static_cast<const MemberExpr *>(original);
                auto self = _dup_expr(e->self());
                copy = e->is_swizzle() ?
                           fb->swizzle(e->type(), self,
                                       e->swizzle_size(),
                                       e->swizzle_code()) :
                           fb->member(e->type(), self,
                                      e->member_index());
                break;
            }
            case Expression::Tag::ACCESS: {
                auto e = static_cast<const AccessExpr *>(original);
                auto range = _dup_expr(e->range());
                auto index = _dup_expr(e->index());
                copy = fb->access(e->type(), range, index);
                break;
            }
            case Expression::Tag::LITERAL: {
                auto e = static_cast<const LiteralExpr *>(original);
                copy = fb->literal(e->type(), e->value());
                break;
            }
            case Expression::Tag::REF: {
                auto e = static_cast<const RefExpr *>(original);
                auto iter = _contexts.back()->var_map.find(e->variable().uid());
                LUISA_ASSERT(iter != _contexts.back()->var_map.end(),
                             "Variable not found in context.");
                copy = iter->second;
                break;
            }
            case Expression::Tag::CONSTANT: {
                auto e = static_cast<const ConstantExpr *>(original);
                copy = fb->constant(e->data());
                break;
            }
            case Expression::Tag::CALL: {
                auto e = static_cast<const CallExpr *>(original);
                luisa::vector<const Expression *> args;
                args.reserve(e->arguments().size());
                for (auto arg : e->arguments()) { args.emplace_back(_dup_expr(arg)); }
                if (e->is_builtin()) {
                    copy = fb->call(e->type(), e->op(), args);
                } else if (e->is_custom()) {
                    auto callee = _duplicate(*e->custom().builder());
                    copy = fb->call(e->type(), callee->function(), args);
                } else if (e->is_external()) {
                    auto &o = _contexts.back()->original;
                    auto iter = std::find_if(
                        o.external_callables().cbegin(),
                        o.external_callables().cend(),
                        [ext = e->external()](auto &&f) noexcept { return *f == *ext; });
                    LUISA_ASSERT(iter != o.external_callables().cend(),
                                 "External function not found in context.");
                    copy = fb->call(e->type(), *iter, args);
                } else {
                    LUISA_ASSERT(e->is_external(), "Unknown call type.");
                }
                break;
            }
            case Expression::Tag::CAST: {
                auto e = static_cast<const CastExpr *>(original);
                auto src = _dup_expr(e->expression());
                copy = fb->cast(e->type(), e->op(), src);
                break;
            }
            case Expression::Tag::TYPE_ID: {
                auto e = static_cast<const TypeIDExpr *>(original);
                copy = fb->type_id(e->type());
                break;
            }
            case Expression::Tag::STRING_ID: {
                auto e = static_cast<const StringIDExpr *>(original);
                copy = fb->string_id(luisa::string{e->data()});
                break;
            }
            case Expression::Tag::CPUCUSTOM: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::GPUCUSTOM: LUISA_NOT_IMPLEMENTED();
        }
        _contexts.back()->expr_map.emplace(original, copy);
        return copy;
    }

    void _dup_scope(const ScopeStmt *original, ScopeStmt *copy) noexcept {
        FunctionBuilder::current()->with(copy, [original, this] {
            for (auto s : original->statements()) {
                _dup_stmt(s);
                if (auto tag = s->tag();
                    tag == Statement::Tag::BREAK ||
                    tag == Statement::Tag::CONTINUE ||
                    tag == Statement::Tag::RETURN) {
                    break;
                }
            }
        });
    }

    void _dup_stmt(const Statement *stmt) noexcept {
        auto fb = FunctionBuilder::current();
        switch (stmt->tag()) {
            case Statement::Tag::BREAK: {
                fb->break_();
                break;
            }
            case Statement::Tag::CONTINUE: {
                fb->continue_();
                break;
            }
            case Statement::Tag::RETURN: {
                auto s = static_cast<const ReturnStmt *>(stmt);
                auto e = _dup_expr(s->expression());
                fb->return_(e);
                break;
            }
            case Statement::Tag::SCOPE: {
                LUISA_ERROR_WITH_LOCATION(
                    "ScopeStmt should have been "
                    "handled in parent statements.");
            }
            case Statement::Tag::IF: {
                auto s = static_cast<const IfStmt *>(stmt);
                auto cond = _dup_expr(s->condition());
                auto if_ = fb->if_(cond);
                _dup_scope(s->true_branch(), if_->true_branch());
                _dup_scope(s->false_branch(), if_->false_branch());
                break;
            }
            case Statement::Tag::LOOP: {
                auto s = static_cast<const LoopStmt *>(stmt);
                auto loop = fb->loop_();
                _dup_scope(s->body(), loop->body());
                break;
            }
            case Statement::Tag::EXPR: {
                auto s = static_cast<const ExprStmt *>(stmt);
                auto e = _dup_expr(s->expression());
                auto is_void_call = s->expression()->tag() == Expression::Tag::CALL &&
                                    s->expression()->type() == nullptr;
                // call to void function will be handled by FunctionBuilder,
                // otherwise we need to explicitly call void_expr()
                if (!is_void_call) { fb->_void_expr(e); }
                break;
            }
            case Statement::Tag::SWITCH: {
                auto s = static_cast<const SwitchStmt *>(stmt);
                auto e = _dup_expr(s->expression());
                auto sw = fb->switch_(e);
                _dup_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::SWITCH_CASE: {
                auto s = static_cast<const SwitchCaseStmt *>(stmt);
                auto e = _dup_expr(s->expression());
                auto sw = fb->case_(e);
                _dup_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::SWITCH_DEFAULT: {
                auto s = static_cast<const SwitchDefaultStmt *>(stmt);
                auto sw = fb->default_();
                _dup_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::ASSIGN: {
                auto s = static_cast<const AssignStmt *>(stmt);
                auto lhs = _dup_expr(s->lhs());
                auto rhs = _dup_expr(s->rhs());
                fb->assign(lhs, rhs);
                break;
            }
            case Statement::Tag::FOR: {
                auto s = static_cast<const ForStmt *>(stmt);
                auto var = _dup_expr(s->variable());
                auto cond = _dup_expr(s->condition());
                auto step = _dup_expr(s->step());
                auto for_ = fb->for_(var, cond, step);
                _dup_scope(s->body(), for_->body());
                break;
            }
            case Statement::Tag::COMMENT: {
                auto s = static_cast<const CommentStmt *>(stmt);
                fb->comment_(luisa::string{s->comment()});
                break;
            }
            case Statement::Tag::RAY_QUERY: {
                auto s = static_cast<const RayQueryStmt *>(stmt);
                auto q = _dup_expr(s->query());
                LUISA_ASSERT(q->tag() == Expression::Tag::REF,
                             "RayQueryExpr should be a reference.");
                auto rq = fb->ray_query_(static_cast<const RefExpr *>(q));
                _dup_scope(s->on_triangle_candidate(),
                           rq->on_triangle_candidate());
                _dup_scope(s->on_procedural_candidate(),
                           rq->on_procedural_candidate());
                break;
            }
            case Statement::Tag::AUTO_DIFF: {
                auto s = static_cast<const AutoDiffStmt *>(stmt);
                auto ad = fb->autodiff_();
                _dup_scope(s->body(), ad->body());
                break;
            }
        }
    }

private:
    [[nodiscard]] luisa::shared_ptr<const FunctionBuilder>
    _duplicate(const FunctionBuilder &f) noexcept {
        if (auto iter = _duplicated.find(&f);
            iter != _duplicated.end()) {
            return iter->second;
        }
        auto dup = FunctionBuilder::_define(f.tag(), [&f, this] {
            DupCtx ctx{.original = f};
            _contexts.emplace_back(&ctx);
            _dup_function(f);
            LUISA_ASSERT(!_contexts.empty() && _contexts.back() == &ctx,
                         "Corrupted context stack.");
            _contexts.pop_back();
        });
        auto not_predefined = _duplicated.emplace(&f, dup).second;
        LUISA_ASSERT(not_predefined, "FunctionBuilder::duplicate() called recursively.");
        return dup;
    }

public:
    [[nodiscard]] static luisa::shared_ptr<const FunctionBuilder>
    duplicate(const FunctionBuilder &f) noexcept {
        FunctionDuplicator d;
        return d._duplicate(f);
    }
};

luisa::shared_ptr<const FunctionBuilder> FunctionBuilder::duplicate() const noexcept {
    return FunctionDuplicator::duplicate(*this);
}

luisa::shared_ptr<const FunctionBuilder> FunctionBuilder::_duplicate_if_necessary() const noexcept {
    return shared_from_this();// TODO
}

}// namespace luisa::compute::detail
