import os
from typing import List, Dict, Tuple, Optional, Union, Set

cpp_def = open('ir_v2_defs.h', 'w')
fwd_file = open('ir_v2_fwd.h', 'w')
c_def = open('ir_v2_api.h', 'w')
c_api_impl = open('../../../src/ir_v2/ir_v2_api.cpp', 'w')

MAP_FFI_TYPE = {
    'bool': 'bool',
    'uint8_t': 'uint8_t',
    'uint16_t': 'uint16_t',
    'uint32_t': 'uint32_t',
    'uint64_t': 'uint64_t',

    'const Type*': 'const Type *',
    'luisa::string': 'Slice<const char>',
    'luisa::vector<uint8_t>': 'Slice<uint8_t>',
    'luisa::vector<Node*>': 'Slice<Node*>',
    'luisa::vector<PhiIncoming>': 'Slice<PhiIncoming>',
    'luisa::vector<SwitchCase>': 'Slice<SwitchCase>',
    'luisa::vector<Binding>': 'Slice<Binding>',
    'luisa::vector<Callable*>': 'Slice<Callable*>',
    'luisa::vector<CallableModule*>': 'Slice<CallableModule*>',
    'luisa::vector<Module*>': 'Slice<Module*>',
    'luisa::vector<KernelModule*>': 'Slice<KernelModule*>',
    'luisa::shared_ptr<CallableModule>': 'CallableModule*',
    'CpuExternFn': 'CpuExternFn',
    'const Func*': 'const Func*',
    'Node*': 'Node*',
    'BasicBlock*': 'BasicBlock*',
}


def to_screaming_snake_case(name: str):
    out = ''
    for c in name:
        if c.isupper() and out != '':
            out += '_'
        out += c.upper()
    return out


print('#pragma once', file=cpp_def)
print('''#pragma once
// if msvc
#ifdef _MSC_VER
#pragma warning( disable : 4190)
#endif
''', file=fwd_file)
print('#include <cstdint>', file=fwd_file)
print('#pragma once', file=c_def)
print('/// This file is generated by gen_ir_def.py', file=cpp_def)
print('#include <type_traits>', file=cpp_def)
print('#include <luisa/ir_v2/ir_v2_fwd.h>', file=cpp_def)
print('namespace luisa::compute::ir_v2 {', file=cpp_def)
print(
    'namespace luisa::compute { class Type;} namespace luisa::compute::ir_v2 {', file=fwd_file)
print('#include <luisa/ir_v2/ir_v2_fwd.h>', file=c_def)
print('#include <luisa/ir_v2/ir_v2_api.h>', file=c_api_impl)
print('''
#ifndef LC_IR_API
#ifdef LC_IR_EXPORT_DLL
#define LC_IR_API __declspec(dllexport)
#else
#define LC_IR_API __declspec(dllimport)
#endif
#endif
namespace luisa::compute::ir_v2 {
template<class T>
struct Slice {
    T *data = nullptr;
    size_t len = 0;
    constexpr Slice() noexcept = default;
    constexpr Slice(T *data, size_t len) noexcept : data(data), len(len) {}
#ifndef BINDGEN
    // construct from array
    template<size_t N>
    constexpr Slice(T (&arr)[N]) noexcept : data(arr), len(N) {}
    // construct from std::array
    template<size_t N>
    constexpr Slice(std::array<T, N> &arr) noexcept : data(arr.data()), len(N) {}
    // construct from luisa::vector
    constexpr Slice(luisa::vector<T> &vec) noexcept : data(vec.data()), len(vec.size()) {}
    // construct from luisa::span
    constexpr Slice(luisa::span<T> &span) noexcept : data(span.data()), len(span.size()) {}
    // construct from luisa::string
    constexpr Slice(luisa::string &str) noexcept : data(str.data()), len(str.size()) {
        static_assert(std::is_same_v<T, char> || std::is_same_v<T, const char>);
    }
#endif 
};    
}
''', file=c_def)
print('namespace luisa::compute::ir_v2 {', file=c_def)
print('namespace luisa::compute::ir_v2 {', file=c_api_impl)
print('''
struct Node;
class BasicBlock;
struct CallableModule;
struct Module;
struct KernelModule; 
''', file=fwd_file)


class Item:
    def __init__(self, name, fields: List[Tuple[str, str]], comment=None) -> None:
        self.cpp_src = ''
        self.name = name
        self.fields = fields
        self.comment = comment

    def gen(self):
        out = 'public:\n'
        for field in self.fields:
            out += '    {} {};\n'.format(field[0], field[1])
        out += self.cpp_src
        return out

    def gen_c_api(self):
        # get xx_field() -> xx
        for f in self.fields:
            print('extern "C" LC_IR_API void lc_ir_v2_{}_{}({} *self, {} *out);'.format(
                self.name, f[1], self.name, MAP_FFI_TYPE[f[0]]), file=c_def)
            print('extern "C" LC_IR_API void lc_ir_v2_{}_{}({} *self, {} *out) {{'.format(
                self.name, f[1], self.name, MAP_FFI_TYPE[f[0]]), file=c_api_impl)
            if 'shared_ptr' in f[0]:
                print(
                    '    *out = self->{}.get();'.format(f[1]), file=c_api_impl)
            else:
                print('    *out = self->{};'.format(f[1]), file=c_api_impl)
            print('}', file=c_api_impl)


class Instruction(Item):
    def __init__(self, name, fields=None, cpp_src=None, **kwargs) -> None:
        if fields is None:
            fields = []
        super().__init__(name, fields, **kwargs)
        if cpp_src is not None:
            self.cpp_src = cpp_src


class Func(Item):
    def __init__(self, name, fields=None, side_effects=False, **kwargs) -> None:
        if fields is None:
            fields = []
        super().__init__(name, fields, **kwargs)
        self.side_effects = side_effects

    def gen(self):
        out = super().gen()
        out += '    [[nodiscard]] constexpr bool has_side_effects() const noexcept override{\n'
        out += '        return {};\n'.format(
            'true' if self.side_effects else 'false')
        out += '    }\n'
        return out


def gen_adt(adt: str, cpp_src: str, variants: List[Item]):
    # gen cpp
    print('struct {};'.format(adt), file=fwd_file)
    for variant in variants:
        print('struct {};'.format(variant.name), file=fwd_file)
    print('struct {} {{'.format(adt), file=cpp_def)
    print('public:', file=cpp_def)
    print('    typedef {}Tag Tag;'.format(adt), file=cpp_def)
    print('    [[nodiscard]] virtual Tag tag() const noexcept = 0;', file=cpp_def)
    print('    template<class T> requires std::is_base_of_v<{}, T> [[nodiscard]] bool isa()const noexcept {{'.format(
        adt), file=cpp_def)
    print('        return tag() == T::static_tag();', file=cpp_def)
    print('    }', file=cpp_def)
    print('    template<class T> requires std::is_base_of_v<{}, T> [[nodiscard]]  T* as() {{'.format(
        adt), file=cpp_def)
    print('        return isa<T>() ? static_cast<T*>(this) : nullptr;', file=cpp_def)
    print('    }', file=cpp_def)
    print('    template<class T> requires std::is_base_of_v<{}, T> [[nodiscard]]  const T* as() const {{'.format(
        adt), file=cpp_def)
    print('        return isa<T>() ? static_cast<const T*>(this) : nullptr;', file=cpp_def)
    print('    }', file=cpp_def)
    print('    ', cpp_src, file=cpp_def)
    print('};', file=cpp_def)

    for variant in variants:
        print('struct LC_IR_API {} : public {} {{'.format(
            variant.name, adt), file=cpp_def)
        print('public:', file=cpp_def)
        print('    using {}::Tag;'.format(adt), file=cpp_def)
        print('    static constexpr Tag static_tag() noexcept {', file=cpp_def)
        print('        return Tag::{};'.format(
            to_screaming_snake_case(variant.name)), file=cpp_def)
        print('    }', file=cpp_def)
        print(
            '    [[nodiscard]] Tag tag() const noexcept override {', file=cpp_def)
        print('        return static_tag();', file=cpp_def)
        print('    }', file=cpp_def)
        print('    ', variant.gen(), file=cpp_def)
        print('};', file=cpp_def)

    # gen c api
    print(f'    enum class {adt}Tag : unsigned int {{', file=fwd_file)
    for variant in variants:
        print('        {},'.format(
            to_screaming_snake_case(variant.name)), file=fwd_file)
    print('    };', file=fwd_file)
    for variant in variants:
        print('extern "C" LC_IR_API {1} * lc_ir_v2_{0}_as_{1}({0} *self);'.format(
            adt, variant.name), file=c_def)
        print('extern "C" LC_IR_API {1} * lc_ir_v2_{0}_as_{1}({0} *self) {{'.format(
            adt, variant.name), file=c_api_impl)
        print('    return self->as<{0}>();'.format(variant.name),
              file=c_api_impl)
        print('}', file=c_api_impl)

    print('extern "C" LC_IR_API {0}Tag lc_ir_v2_{0}_tag({0} *self);'.format(
        adt), file=c_def)
    print('extern "C" LC_IR_API {0}Tag lc_ir_v2_{0}_tag({0} *self) {{'.format(
        adt), file=c_api_impl)
    print('    return self->tag();', file=c_api_impl)
    print('}', file=c_api_impl)
    for variant in variants:
        variant.gen_c_api()


instructions = [
    Instruction('Buffer'),
    Instruction('Texture2d'),
    Instruction('Texture3d'),
    Instruction('BindlessArray'),
    Instruction('Accel'),
    Instruction('Shared'),
    Instruction('Uniform'),
    Instruction('Argument', [
        ('bool', 'by_value'),
    ]),
    Instruction('Const', [
        ('const Type*', 'ty'),
        ('luisa::vector<uint8_t>', 'value')
    ]),
    Instruction('Call', [
        ('const Func*', 'func'),
        ('luisa::vector<Node*>', 'args'),
    ], cpp_src='''Call(const Func* func, luisa::vector<Node*> args) noexcept {
        this->func = std::move(func);
        this->args = std::move(args);
}
'''),
    Instruction('Phi', [('luisa::vector<PhiIncoming>', 'incomings')]),
    Instruction("BasicBlockSentinel", []),
    Instruction('If', [
        ('Node*', 'cond'),
        ('BasicBlock*', 'true_branch'),
        ('BasicBlock*', 'false_branch')
    ], cpp_src='''If(Node* cond, BasicBlock* true_branch, BasicBlock* false_branch) noexcept {
    this->cond = cond;
    this->true_branch = true_branch;
    this->false_branch = false_branch;
}
'''),
    Instruction('GenericLoop', [
        ('BasicBlock*', 'prepare'),
        ('Node*', 'cond'),
        ('BasicBlock*', 'body'),
        ('BasicBlock*', 'update')
    ], cpp_src='''GenericLoop(BasicBlock* prepare, Node* cond, BasicBlock* body, BasicBlock* update) noexcept {
    this->prepare = prepare;
    this->cond = cond;
    this->body = body;
    this->update = update;
}
'''),
    Instruction('Switch', [
        ('Node*', 'value'),
        ('luisa::vector<SwitchCase>', 'cases'),
        ('BasicBlock*', 'default_')
    ], cpp_src='''Switch(Node* value, luisa::vector<SwitchCase> cases, BasicBlock* default_) noexcept {
    this->value = value;
    this->cases = std::move(cases);
    this->default_ = default_;
}'''),
    Instruction('Local', [
        ('Node*', 'init')
    ], cpp_src='''Local(Node* init) noexcept {
    this->init = init;
}'''),
    Instruction('Break', []),
    Instruction('Continue', []),
    Instruction('Return', [
        ('Node*', 'value')
    ], cpp_src='''Return(Node* value) noexcept {
    this->value = value;
}'''),
    Instruction('Print', [
        ('luisa::string', 'fmt'),
        ('luisa::vector<Node*>', 'args')
    ], cpp_src='''Print(luisa::string fmt, luisa::vector<Node*> args) noexcept {
    this->fmt = std::move(fmt);
    this->args = std::move(args);
}'''),
]

funcs = [
    Func('Zero', []),
    Func('One', []),

    Func('Assume', [
        ('luisa::string', 'msg')
    ]),
    Func('Unreachable', [
         ('luisa::string', 'msg')
         ]),

    Func('ThreadId', []),
    Func('BlockId', []),
    Func('WarpSize', []),
    Func('WarpLaneId', []),
    Func('DispatchId', []),
    Func('DispatchSize', []),

    Func('PropagateGradient', [], side_effects=True),
    Func('OutputGradient', []),

    Func('RequiresGradient', [], side_effects=True),
    Func('Backward', [], comment='//Backward(out, out_grad)', side_effects=True),
    Func('Gradient', []),
    Func('AccGrad', [], side_effects=True),
    Func('Detach', []),

    Func('RayTracingInstanceTransform'),
    Func('RayTracingInstanceVisibilityMask'),
    Func('RayTracingInstanceUserId'),
    Func('RayTracingSetInstanceTransform', side_effects=True),
    Func('RayTracingSetInstanceOpacity', side_effects=True),
    Func('RayTracingSetInstanceVisibility', side_effects=True),
    Func('RayTracingSetInstanceUserId', side_effects=True),

    Func('RayTracingTraceClosest', []),
    Func('RayTracingTraceAny', []),
    Func('RayTracingQueryAll', []),
    Func('RayTracingQueryAny', []),
    Func('RayQueryWorldSpaceRay', []),
    Func('RayQueryProceduralCandidateHit', []),
    Func('RayQueryTriangleCandidateHit', []),
    Func('RayQueryCommittedHit', []),
    Func('RayQueryCommitTriangle', [], side_effects=True),
    Func('RayQueryCommitdProcedural', [], side_effects=True),
    Func('RayQueryTerminate', []),

    Func('Load', []),
    Func('Store', [], side_effects=True),

    Func('Cast', []),
    Func('BitCast', []),

    Func('Add'),
    Func('Sub'),
    Func('Mul'),
    Func('Div'),
    Func('Rem'),
    Func('BitAnd'),
    Func('BitOr'),
    Func('BitXor'),
    Func('Shl'),
    Func('Shr'),
    Func('RotRight'),
    Func('RotLeft'),
    Func('Eq'),
    Func('Ne'),
    Func('Lt'),
    Func('Le'),
    Func('Gt'),
    Func('Ge'),
    Func('MatCompMul'),

    Func('Neg'),
    Func('Not'),
    Func('BitNot'),

    Func('All', []),
    Func('Any', []),

    Func('Select'),
    Func('Clamp'),
    Func('Lerp'),
    Func('Step'),
    Func('Saturate'),
    Func('SmoothStep'),

    Func('Abs'),
    Func('Min'),
    Func('Max'),

    Func('ReduceSum'),
    Func('ReduceProd'),
    Func('ReduceMin'),
    Func('ReduceMax'),
    Func('Clz'),
    Func('Ctz'),
    Func('PopCount'),
    Func('Reverse'),
    Func('IsInf'),
    Func('IsNan'),
    Func('Acos'),
    Func('Acosh'),
    Func('Asin'),
    Func('Asinh'),
    Func('Atan'),
    Func('Atan2'),
    Func('Atanh'),
    Func('Cos'),
    Func('Cosh'),
    Func('Sin'),
    Func('Sinh'),
    Func('Tan'),
    Func('Tanh'),
    Func('Exp'),
    Func('Exp2'),
    Func('Exp10'),
    Func('Log'),
    Func('Log2'),
    Func('Log10'),
    Func('Powi'),
    Func('Powf'),
    Func('Sqrt'),
    Func('Rsqrt'),
    Func('Ceil'),
    Func('Floor'),
    Func('Fract'),
    Func('Trunc'),
    Func('Round'),
    Func('Fma'),
    Func('Copysign'),
    Func('Cross'),
    Func('Dot'),
    Func('OuterProduct'),
    Func('Length'),
    Func('LengthSquared'),
    Func('Normalize'),
    Func('Faceforward'),
    Func('Distance'),
    Func('Reflect'),
    Func('Determinant'),
    Func('Transpose'),
    Func('Inverse'),

    Func('WarpIsFirstActiveLane'),
    Func('WarpFirstActiveLane'),
    Func('WarpActiveAllEqual'),
    Func('WarpActiveBitAnd'),
    Func('WarpActiveBitOr'),
    Func('WarpActiveBitXor'),
    Func('WarpActiveCountBits'),
    Func('WarpActiveMax'),
    Func('WarpActiveMin'),
    Func('WarpActiveProduct'),
    Func('WarpActiveSum'),
    Func('WarpActiveAll'),
    Func('WarpActiveAny'),
    Func('WarpActiveBitMask'),
    Func('WarpPrefixCountBits'),
    Func('WarpPrefixSum'),
    Func('WarpPrefixProduct'),
    Func('WarpReadLaneAt'),
    Func('WarpReadFirstLane'),
    Func('SynchronizeBlock'),

    Func('AtomicExchange', [], side_effects=True),
    Func('AtomicCompareExchange', [], side_effects=True),
    Func('AtomicFetchAdd', [], side_effects=True),
    Func('AtomicFetchSub', [], side_effects=True),
    Func('AtomicFetchAnd', [], side_effects=True),
    Func('AtomicFetchOr', [], side_effects=True),
    Func('AtomicFetchXor', [], side_effects=True),
    Func('AtomicFetchMin', [], side_effects=True),
    Func('AtomicFetchMax', [], side_effects=True),

    Func('BufferWrite', [], side_effects=True),
    Func('BufferRead', []),
    Func('BufferSize', []),

    Func('ByteBufferWrite', [], side_effects=True),
    Func('ByteBufferRead', []),
    Func('ByteBufferSize', []),

    Func('Texture2dRead'),
    Func('Texture2dWrite', [], side_effects=True),
    Func('Texture2dSize'),

    Func('Texture3dRead'),
    Func('Texture3dWrite', [], side_effects=True),
    Func('Texture3dSize'),

    Func('BindlessTexture2dSample', []),
    Func('BindlessTexture2dSampleLevel', []),
    Func('BindlessTexture2dSampleGrad', []),
    Func('BindlessTexture2dSampleGradLevel', []),
    Func('BindlessTexture2dRead', []),
    Func('BindlessTexture2dSize', []),
    Func('BindlessTexture2dSizeLevel', []),

    Func('BindlessTexture3dSample', []),
    Func('BindlessTexture3dSampleLevel', []),
    Func('BindlessTexture3dSampleGrad', []),
    Func('BindlessTexture3dSampleGradLevel', []),
    Func('BindlessTexture3dRead', []),
    Func('BindlessTexture3dSize', []),
    Func('BindlessTexture3dSizeLevel', []),

    Func('BindlessBufferWrite', [], side_effects=True),
    Func('BindlessBufferRead', []),
    Func('BindlessBufferSize', []),

    Func('BindlessByteBufferWrite', [], side_effects=True),
    Func('BindlessByteBufferRead', []),
    Func('BindlessByteBufferSize', []),

    Func('Vec', []),
    Func('Vec2', []),
    Func('Vec3', []),
    Func('Vec4', []),

    Func('Permute'),

    Func('GetElementPtr', []),
    Func('ExtractElement', []),
    Func('InsertElement', []),

    Func('Array'),
    Func('Struct'),

    Func('MatFull', []),
    Func('Mat2', []),
    Func('Mat3', []),
    Func('Mat4', []),

    Func('BindlessAtomicFetchAdd', [
         ('const Type*', 'ty')], side_effects=True),
    Func('BindlessAtomicFetchSub', [
         ('const Type*', 'ty')], side_effects=True),
    Func('BindlessAtomicFetchAnd', [
         ('const Type*', 'ty')], side_effects=True),
    Func('BindlessAtomicFetchOr', [('const Type*', 'ty')], side_effects=True),
    Func('BindlessAtomicFetchXor', [
         ('const Type*', 'ty')], side_effects=True),
    Func('BindlessAtomicFetchMin', [
         ('const Type*', 'ty')], side_effects=True),
    Func('BindlessAtomicFetchMax', [
         ('const Type*', 'ty')], side_effects=True),

    Func('Callable', [
        ('luisa::shared_ptr<CallableModule>', 'module'),
    ]),
    Func('CpuExt', [
        ('CpuExternFn', 'f'),
    ]),
    Func('ShaderExecutionReorder')
]

FUNC_CPP_SRC = '''
[[nodiscard]] virtual bool has_side_effects() const noexcept = 0;
'''

print('''
struct PhiIncoming {
    const BasicBlock *block = nullptr;
    const Node *value = nullptr;
};
struct SwitchCase {
    int32_t value = 0;
    const BasicBlock *block = nullptr;    
};
struct CpuExternFn {
    void *data = nullptr;
    void (*func)(void *data, void *args) = nullptr;
    void (*dtor)(void *data) = nullptr;
    const Type* arg_ty = nullptr;
};
''', file=fwd_file)
gen_adt("Func", FUNC_CPP_SRC, funcs)
gen_adt("Instruction", "", instructions)


bindings = [
    Item('BufferBinding', [
        ('uint64_t', 'handle'),
        ('uint64_t', 'offset'),
        ('uint64_t', 'size')
    ]),
    Item('TextureBinding', [
        ('uint64_t', 'handle'),
        ('uint64_t', 'level'),
    ]),
    Item('BindlessArrayBinding', [
        ('uint64_t', 'handle'),
    ]),
    Item('AccelBinding', [
        ('uint64_t', 'handle'),
    ]),
]
gen_adt('Binding', '', bindings)
print('}', file=cpp_def)
print('}', file=fwd_file)
print('}', file=c_def)
print('}', file=c_api_impl)
cpp_def.close()
fwd_file.close()
c_def.close()
c_api_impl.close()

# run clang-format
os.system('clang-format -i ir_v2_defs.h')
os.system('clang-format -i ir_v2_fwd.h')
os.system('clang-format -i ir_v2_api.h')
os.system('clang-format -i ../../../src/ir_v2/ir_v2_api.cpp')

# run bindgen
os.system('bindgen ir_v2_api.h -o ../../../src/rust/luisa_compute_ir_v2/src/binding.rs --rustified-enum .*Tag --disable-name-namespacing '
          '-- -I../../ -x c++ -std=c++17 -DLC_IR_EXPORT_DLL=1 -DBINDGEN -Wno-pragma-once-outside-header -Wno-return-type-c-linkage')
