#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ir_v2/ir_v2_fwd.h>
#include <luisa/ir_v2/ir_v2_defs.h>
#include <luisa/core/logging.h>

namespace luisa::compute::ir_v2 {
class Pool;
struct LC_IR_API Node {
    Node *prev = nullptr;
    Node *next = nullptr;
    BasicBlock *scope = nullptr;
    Instruction *inst = nullptr;
    const Type *ty = Type::of<void>();
    Node() noexcept = default;
    Node(Instruction *inst, const Type *ty) : inst{inst}, ty(ty) {}

    void insert_after_this(Node *n) noexcept {
        LUISA_ASSERT(n != nullptr, "bad node");
        
        LUISA_ASSERT(next != nullptr, "bad node");
        n->prev = this;
        n->next = next;
        next->prev = n;
        next = n;
    }
    void insert_before_this(Node *n) noexcept {
        LUISA_ASSERT(n != nullptr, "bad node");
        
        LUISA_ASSERT(prev != nullptr, "bad node");
        n->prev = prev;
        n->next = this;
        prev->next = n;
        prev = n;
    }
    void remove_this() noexcept {
        
        LUISA_ASSERT(prev != nullptr, "bad node");
        LUISA_ASSERT(next != nullptr, "bad node");
        prev->next = next;
        next->prev = prev;
        prev = nullptr;
        next = nullptr;
    }
    luisa::optional<int32_t> get_index() const noexcept {
        _check();
        if (auto cst = inst->as<Const>()) {
            auto ty = cst->ty;
            if (ty == Type::of<int32_t>()) {
                return *reinterpret_cast<int32_t *>(cst->value.data());
            }
        }
        return luisa::nullopt;
    }
    bool is_const() const noexcept {
        _check();
        return inst->isa<Const>();
    }
    bool is_call() const noexcept {
        _check();
        return inst->isa<Call>();
    }
    bool is_local() const noexcept {
        _check();
        return inst->isa<Local>();
    }
    bool is_gep() const noexcept {
        _check();
        if (auto call = inst->as<Call>()) {
            return call->func->isa<GetElementPtr>();
        }
        return false;
    }
private:
    void _check() const noexcept {
        LUISA_ASSERT(scope != nullptr, "bad node");
        
    }
};

class LC_IR_API BasicBlock {
    Node *_first;
    Node *_last;
public:
    BasicBlock(Pool &pool) noexcept;
    template<class F>
    void for_each(F &&f) const noexcept {
        auto n = _first.next;
        while (n != _last) {
            f(n);
            n = n->next;
        }
    }
    Node *first() const noexcept {
        return _first;
    }
    Node *last() const noexcept {
        return _last;
    }
};

class LC_IR_API Pool {
    using Deleter = void (*)(void *) noexcept;
public:
    template<typename T, typename... Args>
    [[nodiscard]] T *alloc(Args &&...args) noexcept {
        auto ptr = luisa::new_with_allocator<T>(std::forward<Args>(args)...);
        _deleters.emplace_back([](void *p) noexcept {
            luisa::delete_with_allocator<T>(static_cast<T *>(p));
        });
        return ptr;
    }
    ~Pool() noexcept {
        for (auto d : _deleters) { d(nullptr); }
    }
private:
    luisa::vector<Deleter> _deleters;
};

class LC_IR_API IrBuilder {
    luisa::shared_ptr<Pool> _pool;
    Node *_insert_point = nullptr;
    BasicBlock *_current_bb = nullptr;
    [[nodiscard]] Node *append(Node *n) {
        LUISA_ASSERT(n != nullptr, "bad node");
        LUISA_ASSERT(_current_bb != nullptr, "bad node");
        LUISA_ASSERT(n->scope == nullptr, "bad node");
        n->scope = _current_bb;
        _insert_point->insert_after_this(n);
        _insert_point = n;
        return n;
    }
public:
    static IrBuilder create_without_bb(luisa::shared_ptr<Pool> pool) noexcept {
        auto builder = IrBuilder{pool};
        return builder;
    }
    IrBuilder(luisa::shared_ptr<Pool> pool) noexcept : _pool{pool} {
        _current_bb = _pool->alloc<BasicBlock>(*pool);
        _insert_point = _current_bb->first();
    }
    [[nodiscard]] Pool &pool() const noexcept {
        return *_pool;
    }
    void set_insert_point(Node *n) noexcept {
        LUISA_ASSERT(n != nullptr, "bad node");
        if (_current_bb != nullptr) {
            LUISA_ASSERT(n->scope == _current_bb, "bad node");
        }
        _insert_point = n;
    }
    [[nodiscard]] Node *insert_point() const noexcept {
        return _insert_point;
    }
    auto &pool() noexcept {
        return *_pool;
    }
    template<class F>
    [[nodiscard]] Node *call(luisa::span<Node *> args, const Type *ty) noexcept {
        auto f = _pool->alloc<F>();
        return call(f, args, ty);
    }
    [[nodiscard]] Node *call(const Func *f, luisa::span<Node *> args, const Type *ty) noexcept;
    template<class T>
    [[nodiscard]] Node *const_(T v) noexcept {
        luisa::vector<uint8_t> data(sizeof(T));
        std::memcpy(data.data(), &v, sizeof(T));
        auto cst = _pool->alloc<Const>();
        cst->ty = Type::of<T>();
        cst->value = std::move(data);
        return append(_pool->alloc<Node>(cst, Type::of<T>()));
    }
    [[nodiscard]] Node *extract_element(Node *value, luisa::span<uint32_t> indices, const Type *ty) noexcept {
        luisa::vector<Node *> args;
        args.push_back(value);
        for (auto i : indices) {
            args.push_back(const_(i));
        }
        return call(_pool->alloc<ExtractElement>(), args, ty);
    }
    [[nodiscard]] Node *insert_element(Node *agg, Node *el, luisa::span<uint32_t> indices, const Type *ty) noexcept {
        luisa::vector<Node *> args;
        args.push_back(agg);
        args.push_back(el);
        for (auto i : indices) {
            args.push_back(const_(i));
        }
        return call(_pool->alloc<InsertElement>(), args, ty);
    }
    [[nodiscard]] Node *gep(Node *agg, luisa::span<uint32_t> indices, const Type *ty) noexcept {
        luisa::vector<Node *> args;
        if (agg->is_gep()) {
            auto call = agg->inst->as<Call>();
            LUISA_ASSERT(!call->args.empty(), "bad gep");
            LUISA_ASSERT(!call->args[0]->is_gep(), "bad gep");
            for (auto a : call->args) {
                args.push_back(a);
            }
        } else {
            args.push_back(agg);
        }
        for (auto i : indices) {
            args.push_back(const_(i));
        }
        return call(_pool->alloc<GetElementPtr>(), args, ty);
    }
    Node *if_(Node *cond, BasicBlock *true_branch, BasicBlock *false_branch) noexcept;
    Node *generic_loop(BasicBlock *perpare, Node *cond, BasicBlock *body, BasicBlock *after) noexcept;
    Node *switch_(Node *value, luisa::span<SwitchCase> cases, BasicBlock *default_branch) noexcept;
    const BasicBlock *finish() && noexcept {
        LUISA_ASSERT(_current_bb != nullptr, "IrBuilder is not configured to produce a basic block");
        return _current_bb;
    }
    Node *return_(Node *value) noexcept {
        auto ret = _pool->alloc<Return>(value);
        return append(_pool->alloc<Node>(ret, Type::of<void>()));
    }
    Node *break_() noexcept {
        auto br = _pool->alloc<Break>();
        return append(_pool->alloc<Node>(br, Type::of<void>()));
    }
    Node *continue_() noexcept {
        auto cont = _pool->alloc<Continue>();
        return append(_pool->alloc<Node>(cont, Type::of<void>()));
    }
};
struct Module {
    BasicBlock *entry = nullptr;
};

struct Capture {
    Node *node = nullptr;
    Binding *binding = nullptr;
};

struct CallableModule {
    luisa::vector<Node *> args;
    Module body;
    luisa::shared_ptr<Pool> pool;
};

struct KernelModule {
    Module body;
    luisa::vector<Node *> args;
    luisa::vector<Capture> captures;
    luisa::shared_ptr<Pool> pool;
    std::array<uint32_t, 3> block_size;
};

class Transform {
    virtual void run(Module &module) noexcept = 0;
};

}// namespace luisa::compute::ir_v2