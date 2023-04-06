#include "codegen_stack_data.h"
#include <runtime/rtx/ray.h>
#include <runtime/rtx/hit.h>
#include <ast/type_registry.h>
namespace lc::dx {
CodegenStackData::CodegenStackData()
    : generateStruct(
          [this](Type const *t) {
              CreateStruct(t);
          }) {
    structReplaceName.try_emplace(
        "float3"sv, "float4"sv);
    structReplaceName.try_emplace(
        "int3"sv, "int4"sv);
    structReplaceName.try_emplace(
        "uint3"sv, "uint4"sv);
    internalStruct.emplace(Type::of<CommittedHit>(), "Hit0");
    internalStruct.emplace(Type::of<TriangleHit>(), "Hit1");
    internalStruct.emplace(Type::of<ProceduralHit>(), "Hit2");
}
void CodegenStackData::Clear() {
    tempSwitchExpr = nullptr;
    arguments.clear();
    scopeCount = -1;
    tempSwitchCounter = 0;
    structTypes.clear();
    constTypes.clear();
    funcTypes.clear();
    customStruct.clear();
    sharedVariable.clear();
    constCount = 0;
    argOffset = 0;
    appdataId = -1;
    count = 0;
    structCount = 0;
    funcCount = 0;
    tempCount = 0;
    bindlessBufferCount = 0;
}
void CodegenStackData::AddBindlessType(Type const *type) {
    bindlessBufferCount = 1;
}
/*
static thread_local bool gIsCodegenSpirv = false;
bool &CodegenStackData::ThreadLocalSpirv() {
    return gIsCodegenSpirv;
}*/

vstd::string_view CodegenStackData::CreateStruct(Type const *t) {
    auto iter = internalStruct.find(t);
    if (iter != internalStruct.end())
        return iter->second;
    auto ite = customStruct.try_emplace(
        t,
        vstd::lazy_eval([&] {
            auto newPtr = new StructGenerator(
                t,
                structCount++,
                util);
            return vstd::create_unique(newPtr);
        }));
    if (ite.second) {
        auto newPtr = ite.first.value().get();
        newPtr->Init(generateStruct);
    }
    return ite.first.value()->GetStructName();
}
std::pair<uint64, bool> CodegenStackData::GetConstCount(uint64 data) {
    auto ite = constTypes.try_emplace(
        data,
        vstd::lazy_eval(
            [&] {
                return constCount++;
            }));
    return {ite.first->second, ite.second};
}

uint64 CodegenStackData::GetFuncCount(void const *data) {
    auto ite = funcTypes.try_emplace(
        data,
        vstd::lazy_eval(
            [&] {
                return funcCount++;
            }));
    return ite.first->second;
}
uint64 CodegenStackData::GetTypeCount(Type const *t) {
    auto ite = structTypes.try_emplace(
        t,
        vstd::lazy_eval(
            [&] {
                return count++;
            }));
    return ite.first->second;
}
namespace detail {

struct CodegenGlobalPool {
    std::mutex mtx;
    vstd::vector<vstd::unique_ptr<CodegenStackData>> allCodegen;
    vstd::unique_ptr<CodegenStackData> Allocate() {
        std::lock_guard lck(mtx);
        if (!allCodegen.empty()) {
            auto ite = std::move(allCodegen.back());
            allCodegen.pop_back();
            ite->Clear();
            return ite;
        }
        return vstd::unique_ptr<CodegenStackData>(new CodegenStackData());
    }
    void DeAllocate(vstd::unique_ptr<CodegenStackData> &&v) {
        std::lock_guard lck(mtx);
        allCodegen.emplace_back(std::move(v));
    }
};
static CodegenGlobalPool codegenGlobalPool;
}// namespace detail
CodegenStackData::~CodegenStackData() {}
vstd::unique_ptr<CodegenStackData> CodegenStackData::Allocate(CodegenUtility* util) {
    auto ptr = detail::codegenGlobalPool.Allocate();
    ptr->util = util;
    return ptr;
}
void CodegenStackData::DeAllocate(vstd::unique_ptr<CodegenStackData> &&v) {
    detail::codegenGlobalPool.DeAllocate(std::move(v));
}
}// namespace lc::dx