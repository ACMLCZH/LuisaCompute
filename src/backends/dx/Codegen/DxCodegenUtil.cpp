#pragma vengine_package vengine_directx

#include <Codegen/DxCodegen.h>
#include <vstl/StringUtility.h>
#include <d3dx12.h>
#include <vstl/variant_util.h>
#include <ast/constant_data.h>
#include <Codegen/StructGenerator.h>
#include <Codegen/ShaderHeader.h>
namespace toolhub::directx {
struct CodegenGlobal {
    vstd::HashMap<Type const *, uint64> structTypes;
    vstd::HashMap<uint64, uint64> constTypes;
    vstd::HashMap<uint64, uint64> funcTypes;
    vstd::HashMap<Type const *, vstd::unique_ptr<StructGenerator>> customStruct;
    vstd::vector<StructGenerator *> customStructVector;
    uint64 count = 0;
    uint64 constCount = 0;
    uint64 funcCount = 0;
    vstd::function<StructGenerator *(Type const *)> generateStruct;
    CodegenGlobal()
        : generateStruct(
              [this](Type const *t) {
                  return CreateStruct(t);
              }) {
    }
    void Clear() {
        structTypes.Clear();
        constTypes.Clear();
        funcTypes.Clear();
        customStruct.Clear();
        customStructVector.clear();
        constCount = 0;
        count = 0;
        funcCount = 0;
    }
    StructGenerator *CreateStruct(Type const *t) {
        auto ite = customStruct.Find(t);
        if (ite) {
            return ite.Value().get();
        }
        auto newPtr = new StructGenerator(
            t,
            customStructVector.size(),
            generateStruct);
        customStruct.ForceEmplace(
            t,
            vstd::create_unique(newPtr));
        customStructVector.emplace_back(newPtr);
        return newPtr;
    }
    uint64 GetConstCount(uint64 data) {
        auto ite = constTypes.Emplace(
            data,
            vstd::MakeLazyEval(
                [&] {
                    return constCount++;
                }));
        return ite.Value();
    }
    uint64 GetFuncCount(uint64 data) {
        auto ite = funcTypes.Emplace(
            data,
            vstd::MakeLazyEval(
                [&] {
                    return funcCount++;
                }));
        return ite.Value();
    }
    uint64 GetTypeCount(Type const *t) {
        auto ite = structTypes.Emplace(
            t,
            vstd::MakeLazyEval(
                [&] {
                    return count++;
                }));
        return ite.Value();
    }
};
static thread_local vstd::optional<CodegenGlobal> opt;
void CodegenUtility::ClearStructType() {
    opt.New();
    opt->Clear();
}
void CodegenUtility::RegistStructType(Type const *type) {
    if (type->is_structure() || type->is_array())
        opt->structTypes.Emplace(type, opt->count++);
    else if (type->is_buffer()) {
        RegistStructType(type->element());
    }
}

static bool IsVarWritable(Function func, Variable i) {
    return ((uint)func.variable_usage(i.uid()) & (uint)Usage::WRITE) != 0;
}
void CodegenUtility::GetVariableName(Variable::Tag type, uint id, vstd::string &str) {
    switch (type) {
        case Variable::Tag::BLOCK_ID:
            str << "thdId"sv;
            break;
        case Variable::Tag::DISPATCH_ID:
            str << "dspId"sv;
            break;
        case Variable::Tag::THREAD_ID:
            str << "grpId"sv;
            break;
        case Variable::Tag::DISPATCH_SIZE:
            str << "dsp_c"sv;
            break;
        case Variable::Tag::LOCAL:
            str << 'l';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::SHARED:
            str << 's';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::REFERENCE:
            str << 'r';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::BUFFER:
            str << 'b';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::TEXTURE:
            str << 't';
            vstd::to_string(id, str);
            break;
        case Variable::Tag::BINDLESS_ARRAY:
            str << "ba"sv;
            vstd::to_string(id, str);
            break;
        case Variable::Tag::ACCEL:
            str << "ac"sv;
            vstd::to_string(id, str);
            break;
        default:
            str << 'v';
            vstd::to_string(id, str);
            break;
    }
}

void CodegenUtility::GetVariableName(Variable const &type, vstd::string &str) {
    GetVariableName(type.tag(), type.uid(), str);
}
void CodegenUtility::GetConstName(ConstantData const &data, vstd::string &str) {
    uint64 constCount = opt->GetConstCount(data.hash());
    str << "c";
    vstd::to_string((constCount), str);
}
void CodegenUtility::GetConstantStruct(ConstantData const &data, vstd::string &str) {
    uint64 constCount = opt->GetConstCount(data.hash());
    //auto typeName = CodegenUtility::GetBasicTypeName(view.index());
    str << "struct tc";
    vstd::to_string((constCount), str);
    uint64 varCount = 1;
    eastl::visit(
        [&](auto &&arr) {
            varCount = arr.size();
        },
        data.view());
    str << "{\n";
    str << CodegenUtility::GetBasicTypeName(data.view().index()) << " v[";
    vstd::to_string((varCount), str);
    str << "];\n";
    str << "};\n";
}
void CodegenUtility::GetConstantData(ConstantData const &data, vstd::string &str) {
    auto &&view = data.view();
    uint64 constCount = opt->GetConstCount(data.hash());

    vstd::string name = vstd::to_string((constCount));
    str << "uniform const tc" << name << " c" << name;
    str << "={{";
    eastl::visit(
        [&](auto &&arr) {
            for (auto const &ele : arr) {
                PrintValue<std::remove_cvref_t<typename std::remove_cvref_t<decltype(arr)>::element_type>> prt;
                prt(ele, str);
                str << ',';
            }
        },
        view);
    auto last = str.end() - 1;
    if (*last == ',')
        *last = '}';
    else
        str << '}';
    str << "};\n";
}

void CodegenUtility::GetTypeName(Type const &type, vstd::string &str, Usage usage) {
    switch (type.tag()) {
        case Type::Tag::BOOL:
            str << "bool"sv;
            return;
        case Type::Tag::FLOAT:
            str << "float"sv;
            return;
        case Type::Tag::INT:
            str << "int"sv;
            return;
        case Type::Tag::UINT:
            str << "uint"sv;
            return;
        case Type::Tag::MATRIX: {
            CodegenUtility::GetTypeName(*type.element(), str, usage);
            vstd::to_string((type.dimension() == 3) ? 4 : type.dimension(), str);
            str << 'x';
            vstd::to_string(type.dimension(), str);
        }
            return;
        case Type::Tag::VECTOR: {
            CodegenUtility::GetTypeName(*type.element(), str, usage);
            vstd::to_string((type.dimension()), str);
        }
            return;
        case Type::Tag::ARRAY:
        case Type::Tag::STRUCTURE: {
            auto customType = opt->CreateStruct(&type);
            str << customType->GetStructName();
        }
            return;
        case Type::Tag::BUFFER:
            if ((static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0)
                str << "RW"sv;
            str << "StructuredBuffer<"sv;
            GetTypeName(*type.element(), str, usage);
            str << '>';
            break;
        case Type::Tag::TEXTURE: {
            if ((static_cast<uint>(usage) & static_cast<uint>(Usage::WRITE)) != 0)
                str << "RW"sv;
            str << "Texture"sv;
            vstd::to_string((type.dimension()), str);
            str << "D<"sv;
            GetTypeName(*type.element(), str, usage);
            if (type.tag() != Type::Tag::VECTOR) {
                str << '4';
            }
            str << '>';
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: {
            str << "StructuredBuffer<BdlsStruct>"sv;
        } break;
        case Type::Tag::ACCEL: {
            str << "RaytracingAccelerationStructure"sv;
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("Bad.");
            break;
    }
}

void CodegenUtility::GetFunctionDecl(Function func, vstd::string &data) {
    if (func.return_type()) {
        //TODO: return type
        CodegenUtility::GetTypeName(*func.return_type(), data, Usage::READ);
    } else {
        data += "void"sv;
    }
    switch (func.tag()) {
        case Function::Tag::CALLABLE: {
            data += " custom_"sv;
            vstd::to_string((opt->GetFuncCount(func.hash())), data);
            if (func.arguments().empty()) {
                data += "()"sv;
            } else {
                data += '(';
                for (auto &&i : func.arguments()) {
                    if (i.tag() == Variable::Tag::REFERENCE) {
                        data += "inout ";
                    }
                    RegistStructType(i.type());
                    CodegenUtility::GetTypeName(*i.type(), data, func.variable_usage(i.uid()));
                    data << ' ';
                    CodegenUtility::GetVariableName(i, data);
                    data += ',';
                }
                data[data.size() - 1] = ')';
            }
        } break;
        default:
            //TODO
            break;
    }
}
void CodegenUtility::GetFunctionName(CallExpr const *expr, vstd::string &str, StringStateVisitor &vis) {
    auto getPointer = [&]() {
        str << '(';
        uint64 sz = 1;
        auto args = expr->arguments();
        if (args.size() >= 1) {
            str << "&(";
            args[0]->accept(vis);
            str << "),";
        }
        for (auto i : vstd::range(1, args.size())) {
            ++sz;
            args[i]->accept(vis);
            if (sz != args.size()) {
                str << ',';
            }
        }
        str << ')';
    };

    auto IsType = [](Type const *const type, Type::Tag const tag, uint const vecEle) {
        if (type->tag() == Type::Tag::VECTOR) {
            if (vecEle > 1) {
                return type->element()->tag() == tag && type->dimension() == vecEle;
            } else {
                return type->tag() == tag;
            }
        } else {
            return vecEle == 1;
        }
    };
    switch (expr->op()) {
        case CallOp::CUSTOM:
            str << "custom_"sv << vstd::to_string((opt->GetFuncCount(expr->custom().hash())));
            break;

        case CallOp::ALL:
            str << "all"sv;
            break;
        case CallOp::ANY:
            str << "any"sv;
            break;
        case CallOp::SELECT: {
            if (expr->arguments()[2]->type()->tag() == Type::Tag::BOOL)
                str << "select_scale"sv;
            else
                str << "select"sv;
        } break;
        case CallOp::CLAMP:
            str << "clamp"sv;
            break;
        case CallOp::LERP:
            str << "lerp"sv;
            break;
        case CallOp::STEP:
            str << "step"sv;
            break;
        case CallOp::ABS:
            str << "abs"sv;
            break;
        case CallOp::MAX:
            str << "max"sv;
            break;
        case CallOp::MIN:
            str << "min"sv;
            break;
        case CallOp::POW:
            str << "pow"sv;
            break;
        case CallOp::CLZ:
            str << "clz"sv;
            break;
        case CallOp::CTZ:
            str << "ctz"sv;
            break;
        case CallOp::POPCOUNT:
            str << "popcount"sv;
            break;
        case CallOp::REVERSE:
            str << "reverse"sv;
            break;
        case CallOp::ISINF:
            str << "isinf"sv;
            break;
        case CallOp::ISNAN:
            str << "isnan"sv;
            break;
        case CallOp::ACOS:
            str << "acos"sv;
            break;
        case CallOp::ACOSH:
            str << "acosh"sv;
            break;
        case CallOp::ASIN:
            str << "asin"sv;
            break;
        case CallOp::ASINH:
            str << "asinh"sv;
            break;
        case CallOp::ATAN:
            str << "atan"sv;
            break;
        case CallOp::ATAN2:
            str << "atan2"sv;
            break;
        case CallOp::ATANH:
            str << "atanh"sv;
            break;
        case CallOp::COS:
            str << "cos"sv;
            break;
        case CallOp::COSH:
            str << "cosh"sv;
            break;
        case CallOp::SIN:
            str << "sin"sv;
            break;
        case CallOp::SINH:
            str << "sinh"sv;
            break;
        case CallOp::TAN:
            str << "tan"sv;
            break;
        case CallOp::TANH:
            str << "tanh"sv;
            break;
        case CallOp::EXP:
            str << "exp"sv;
            break;
        case CallOp::EXP2:
            str << "exp2"sv;
            break;
        case CallOp::EXP10:
            str << "exp10"sv;
            break;
        case CallOp::LOG:
            str << "log"sv;
            break;
        case CallOp::LOG2:
            str << "log2"sv;
            break;
        case CallOp::LOG10:
            str << "log10"sv;
            break;
        case CallOp::SQRT:
            str << "sqrt"sv;
            break;
        case CallOp::RSQRT:
            str << "rsqrt"sv;
            break;
        case CallOp::CEIL:
            str << "ceil"sv;
            break;
        case CallOp::FLOOR:
            str << "floor"sv;
            break;
        case CallOp::FRACT:
            str << "fract"sv;
            break;
        case CallOp::TRUNC:
            str << "trunc"sv;
            break;
        case CallOp::ROUND:
            str << "round"sv;
            break;
        case CallOp::FMA:
            str << "fma"sv;
            break;
        case CallOp::COPYSIGN:
            str << "copysign"sv;
            break;
        case CallOp::CROSS:
            str << "cross"sv;
            break;
        case CallOp::DOT:
            str << "dot"sv;
            break;
        case CallOp::LENGTH:
            str << "length"sv;
            break;
        case CallOp::LENGTH_SQUARED:
            str << "length_sqr"sv;
            break;
        case CallOp::NORMALIZE:
            str << "normalize"sv;
            break;
        case CallOp::FACEFORWARD:
            str << "faceforward"sv;
            break;
        case CallOp::DETERMINANT:
            str << "determinant"sv;
            break;
        case CallOp::TRANSPOSE:
            str << "transpose"sv;
            break;
        case CallOp::INVERSE:
            str << "inverse"sv;
            break;
        case CallOp::ATOMIC_EXCHANGE:
            str << "_atomic_exchange"sv;
            getPointer();
            return;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            str << "_atomic_compare_exchange"sv;
            getPointer();
            return;
        case CallOp::ATOMIC_FETCH_ADD:
            str << "_atomic_add"sv;
            getPointer();
            return;
        case CallOp::ATOMIC_FETCH_SUB:
            str << "_atomic_sub"sv;
            getPointer();
            return;
        case CallOp::ATOMIC_FETCH_AND:
            str << "_atomic_and"sv;
            getPointer();
            return;
        case CallOp::ATOMIC_FETCH_OR:
            str << "_atomic_or"sv;
            getPointer();
            return;
        case CallOp::ATOMIC_FETCH_XOR:
            str << "_atomic_xor"sv;
            getPointer();
            return;
        case CallOp::ATOMIC_FETCH_MIN:
            str << "_atomic_min"sv;
            getPointer();
            return;
        case CallOp::ATOMIC_FETCH_MAX:
            str << "_atomic_max"sv;
            getPointer();
            return;
        case CallOp::TEXTURE_READ:
            str << "Smptx";
            break;
        case CallOp::TEXTURE_WRITE:
            str << "Writetx";
            break;
        case CallOp::MAKE_BOOL2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 2))
                str << "make_bool2"sv;

            break;
        case CallOp::MAKE_BOOL3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 3))
                str << "make_bool3"sv;

            break;
        case CallOp::MAKE_BOOL4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::BOOL, 4))
                str << "make_bool4"sv;

            break;
        case CallOp::MAKE_UINT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 2))
                str << "make_uint2"sv;

            break;
        case CallOp::MAKE_UINT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 3))
                str << "make_uint3"sv;
            break;
        case CallOp::MAKE_UINT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::UINT, 4))
                str << "make_uint4"sv;

            break;
        case CallOp::MAKE_INT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 2))
                str << "make_int2"sv;

            break;
        case CallOp::MAKE_INT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 3))
                str << "make_int3"sv;

            break;
        case CallOp::MAKE_INT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::INT, 4))
                str << "make_int4"sv;

            break;
        case CallOp::MAKE_FLOAT2:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 2))
                str << "make_float2"sv;

            break;
        case CallOp::MAKE_FLOAT3:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 3))
                str << "make_float3"sv;

            break;
        case CallOp::MAKE_FLOAT4:
            if (!IsType(expr->arguments()[0]->type(), Type::Tag::FLOAT, 4))
                str << "make_float4"sv;

            break;
        case CallOp::MAKE_FLOAT2X2:
            str << "make_float2x2"sv;
            break;
        case CallOp::MAKE_FLOAT3X3:
            str << "make_float3x3"sv;
            break;
        case CallOp::MAKE_FLOAT4X4:
            str << "make_float4x4"sv;
            break;
        case CallOp::BUFFER_READ:
            str << "bfread";
            //TODO
            break;
        case CallOp::BUFFER_WRITE:
            str << "bfwrite";
            break;
        default: {
            auto errorType = expr->op();
            VEngine_Log("Function Not Implemented"sv);
            VSTL_ABORT();
        }
    }
    str << '(';
    uint64 sz = 0;
    auto args = expr->arguments();
    for (auto &&i : args) {
        ++sz;
        i->accept(vis);
        if (sz != args.size()) {
            str << ',';
        }
    }
    str << ')';
}
size_t CodegenUtility::GetTypeAlign(Type const &t) {// TODO: use t.alignment()
    switch (t.tag()) {
        case Type::Tag::BOOL:
        case Type::Tag::FLOAT:
        case Type::Tag::INT:
        case Type::Tag::UINT:
            return 4;
            // TODO: incorrect
        case Type::Tag::VECTOR:
        case Type::Tag::MATRIX:
            switch (t.dimension()) {
                case 1:
                    return 4;
                case 2:
                    return 8;
                default:
                    return 16;
            }
        case Type::Tag::ARRAY:
            return GetTypeAlign(*t.element());
        case Type::Tag::STRUCTURE: {
            size_t maxAlign = 1;
            for (auto &&i : t.members()) {
                maxAlign = std::max(maxAlign, GetTypeAlign(*i));
            }
            return maxAlign;
        }
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
        case Type::Tag::ACCEL:
        case Type::Tag::BINDLESS_ARRAY:
            return 8;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Invalid type: {}.", t.description());
    }
}

size_t CodegenUtility::GetTypeSize(Type const &t) {// TODO: use t.size()
    switch (t.tag()) {
        case Type::Tag::BOOL:
            return 1;
        case Type::Tag::FLOAT:
        case Type::Tag::INT:
        case Type::Tag::UINT:
            return 4;
        case Type::Tag::VECTOR:
        case Type::Tag::ARRAY:
            return GetTypeSize(*t.element()) * t.dimension();
        case Type::Tag::MATRIX:
            return GetTypeSize(*t.element()) * t.dimension() * t.dimension();
        case Type::Tag::STRUCTURE: {
            size_t sz = 0;
            size_t maxAlign = 1;
            for (auto &&i : t.members()) {
                auto a = GetTypeAlign(*i);
                maxAlign = std::max(maxAlign, a);
                sz = CalcAlign(sz, a);
                sz += GetTypeSize(*i);
            }
            return CalcAlign(sz, maxAlign);
        }
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
        case Type::Tag::ACCEL:
        case Type::Tag::BINDLESS_ARRAY:
            return 8;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid type: {}.",
        t.description());
}

template<typename T>
struct TypeNameStruct {
    void operator()(vstd::string &str) {
        using BasicTypeUtil = vstd::VariantVisitor_t<basic_types>;
        if constexpr (std::is_same_v<bool, T>) {
            str << "bool";
        } else if constexpr (std::is_same_v<int, T>) {
            str << "int";
        } else if constexpr (std::is_same_v<uint, T>) {
            str << "uint";
        } else if constexpr (std::is_same_v<float, T>) {
            str << "float";
        } else {
            static_assert(vstd::AlwaysFalse<T>, "illegal type");
        }
    }
};
template<typename T, size_t t>
struct TypeNameStruct<luisa::Vector<T, t>> {
    void operator()(vstd::string &str) {
        TypeNameStruct<T>()(str);
        size_t n = (t == 3) ? 4 : t;
        str += ('0' + n);
    }
};
template<size_t t>
struct TypeNameStruct<luisa::Matrix<t>> {
    void operator()(vstd::string &str) {
        TypeNameStruct<float>()(str);
        if constexpr (t == 2) {
            str << "2x2";
        } else if constexpr (t == 3) {
            str << "4x3";
        } else if constexpr (t == 4) {
            str << "4x4";
        } else {
            static_assert(vstd::AlwaysFalse<luisa::Matrix<t>>, "illegal type");
        }
    }
};
void CodegenUtility::GetBasicTypeName(uint64 typeIndex, vstd::string &str) {
    vstd::VariantVisitor_t<basic_types>()(
        [&]<typename T>() {
            TypeNameStruct<T>()(str);
        },
        typeIndex);
}
void CodegenUtility::CodegenFunction(Function func, vstd::string &result) {
    if (func.tag() == Function::Tag::KERNEL) {
        result << "[numthreads("
               << vstd::to_string(func.block_size().x)
               << ','
               << vstd::to_string(func.block_size().y)
               << ','
               << vstd::to_string(func.block_size().z)
               << ")]\n"
               << "void main(uint3 thdId : SV_GroupThreadId, uint3 dspId : SV_DispatchThreadID, uint3 grpId : SV_GroupId)";
    } else {
        GetFunctionDecl(func, result);
    }
    result << "{\n"sv;
    auto constants = func.constants();
    for (auto &&i : constants) {
        GetTypeName(*i.type, result, Usage::READ);
        result << ' ';
        vstd::string constName;
        GetConstName(
            i.data,
            constName);

        result << constName << ";\nconst "sv;
        vstd::string constValueName(constName + "_v");
        GetTypeName(*i.type->element(), result, Usage::READ);
        result << ' ' << constValueName << '[';
        vstd::to_string(i.type->dimension(), result);
        result << "]={"sv;
        auto &&dataView = i.data.view();
        eastl::visit(
            [&]<typename T>(eastl::span<T> const &sp) {
                for (auto i : vstd::range(sp.size())) {
                    auto &&value = sp[i];
                    PrintValue<std::remove_cvref_t<T>>()(value, result);
                    if (i != (sp.size() - 1)) {
                        result << ',';
                    }
                }
            },
            dataView);
        //TODO: constants
        result << "};\n"sv
               << constName
               << ".v="sv
               << constValueName
               << ";\n"sv;
    }
    StringStateVisitor vis(func, result);
    func.body()->accept(vis);
    result << "}\n"sv;
}
void CodegenUtility::GenerateCBuffer(
    Function f,
    std::span<const Variable> vars,
    vstd::string &result) {
    result << R"(cbuffer _Global:register(b0){
uint3 dsp_c;
)"sv;
    size_t structSize = 12;
    size_t alignCount = 0;
    auto isCBuffer = [&](Variable::Tag t) {
        switch (t) {
            case Variable::Tag::BUFFER:
            case Variable::Tag::TEXTURE:
            case Variable::Tag::BINDLESS_ARRAY:
            case Variable::Tag::ACCEL:
            case Variable::Tag::THREAD_ID:
            case Variable::Tag::BLOCK_ID:
            case Variable::Tag::DISPATCH_ID:
            case Variable::Tag::DISPATCH_SIZE:
                return false;
        }
        return true;
    };
    for (auto &&i : vars) {
        if (!isCBuffer(i.tag())) continue;
        StructGenerator::ProvideAlignVariable(
            GetTypeAlign(*i.type()),
            structSize,
            alignCount,
            result);
        GetTypeName(*i.type(), result, f.variable_usage(i.uid()));
        result << ' ';
        GetVariableName(i, result);
        result << ";\n"sv;
    }
    result << "}\n";
}
vstd::optional<CodegenResult> CodegenUtility::Codegen(
    Function kernel) {
    if (kernel.tag() != Function::Tag::KERNEL) return {};
    ClearStructType();
    vstd::string codegenData;
    auto callable = [&](auto &&callable, Function func) -> void {
        for (auto &&i : func.custom_callables()) {
            Function f(i.get());
            callable(callable, f);
        }
        CodegenFunction(func, codegenData);
    };
    callable(callable, kernel);
    vstd::string finalResult;
    finalResult.reserve(65536);
    if (!opt->customStructVector.empty()) {
        for (auto ite = opt->customStructVector.end() - 1; ite != opt->customStructVector.begin() - 1; --ite) {
            auto &&v = *ite;
            finalResult << "struct " << v->GetStructName() << "{\n"
                        << v->GetStructDesc() << "};\n";
        }
    }
    //TODO: print custom struct
    GenerateCBuffer(kernel, kernel.arguments(), finalResult);

    finalResult << "Texture2D<float4> _BindlessTex:register(t0,space1);\n"sv;
    CodegenResult::Properties properties;
    properties.reserve(kernel.arguments().size() + 2);
    properties.emplace_back(
        "_Global"sv,
        Shader::Property{
            ShaderVariableType::ConstantBuffer,
            0,
            0,
            0});
    properties.emplace_back(
        "_BindlessTex"sv,
        Shader::Property{
            ShaderVariableType::SRVDescriptorHeap,
            1,
            0,
            0});
    enum class RegisterType : vbyte {
        CBV,
        UAV,
        SRV
    };
    uint registerCount[3] = {0, 0, 0};
    auto Writable = [&](Variable const &v) {
        return (static_cast<uint>(kernel.variable_usage(v.uid())) & static_cast<uint>(Usage::WRITE)) != 0;
    };
    for (auto &&i : kernel.arguments()) {
        auto print = [&] {
            GetTypeName(*i.type(), finalResult, kernel.variable_usage(i.uid()));
            finalResult << ' ';
            vstd::string varName;
            GetVariableName(i, varName);
            finalResult << varName;
            return varName;
        };
        auto genArg = [&](RegisterType regisT, ShaderVariableType sT, char v) {
            auto &&r = registerCount[(vbyte)regisT];
            Shader::Property prop = {
                .type = sT,
                .spaceIndex = 0,
                .registerIndex = r,
                .arrSize = 0};
            properties.emplace_back(print(), prop);
            finalResult << ":register("sv << v;
            vstd::to_string(r, finalResult);
            finalResult << ");\n"sv;
            r++;
        };

        switch (i.type()->tag()) {
            case Type::Tag::TEXTURE:
                if (Writable(i)) {
                    genArg(RegisterType::UAV, ShaderVariableType::UAVDescriptorHeap, 'u');
                } else {
                    genArg(RegisterType::SRV, ShaderVariableType::SRVDescriptorHeap, 't');
                }
            case Type::Tag::BUFFER: {
                if (Writable(i)) {
                    genArg(RegisterType::UAV, ShaderVariableType::RWStructuredBuffer, 'u');
                } else {
                    genArg(RegisterType::SRV, ShaderVariableType::StructuredBuffer, 't');
                }
            } break;
            case Type::Tag::BINDLESS_ARRAY:
            case Type::Tag::ACCEL: {
                //TODO
            } break;
        }
    }

    finalResult << codegenData;
    return {std::move(finalResult), std::move(properties)};
}
}// namespace toolhub::directx