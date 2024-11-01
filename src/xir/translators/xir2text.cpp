#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/string_scratch.h>
#include <luisa/ast/type.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/intrinsic.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/outline.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/print.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/translators/xir2text.h>

namespace luisa::compute::xir {

class XIR2TextTranslator final {

private:
    StringScratch _prelude;
    StringScratch _main;
    luisa::unordered_map<const Value *, uint> _value_uid_map;
    luisa::unordered_map<const Type *, uint> _struct_uid_map;

private:
    void _emit_constant(const Constant *c) noexcept {
        auto t = _type_ident(c->type());
        auto v = _value_ident(c);
        if (!c->metadata_list().empty()) {
            _emit_metadata_list(_prelude, c->metadata_list());
            _prelude << "\n";
        }
        _prelude << "const %" << v << ": " << t << " = ";
        auto size = c->type()->size();
        for (auto i = 0u; i < size; i++) {
            auto x = static_cast<const uint8_t *>(c->data())[i];
            _prelude << luisa::format("{:02x}", static_cast<uint>(x));
        }
        _prelude << ";\n\n";
    }

    [[nodiscard]] auto _value_uid(const Value *value) noexcept {
        LUISA_ASSERT(value != nullptr, "Value must not be null.");
        auto next_uid = static_cast<uint>(_value_uid_map.size());
        return _value_uid_map.try_emplace(value, next_uid).first->second;
    }

    [[nodiscard]] auto _value_ident(const Value *value) noexcept {
        auto uid = _value_uid(value);
        return luisa::format("%{}", uid);
    }

    [[nodiscard]] auto _struct_uid(const Type *type) noexcept {
        LUISA_ASSERT(type != nullptr, "Type must not be null.");
        // custom
        if (type->is_custom()) {
            auto next_uid = static_cast<uint>(_struct_uid_map.size());
            _prelude << "type T" << next_uid << " = opaque \"" << type->description() << "\";\n\n";
            _struct_uid_map.emplace(type, next_uid);
            return next_uid;
        }
        // structure
        LUISA_ASSERT(type->is_structure(), "Type must be a structure.");
        if (auto iter = _struct_uid_map.find(type); iter != _struct_uid_map.end()) {
            return iter->second;
        }
        luisa::string desc;
        for (auto elem : type->members()) {
            desc.append(_type_ident(elem)).append(", ");
        }
        if (!type->members().empty()) {
            desc.pop_back();
            desc.pop_back();
        }
        auto next_uid = static_cast<uint>(_struct_uid_map.size());
        _prelude << "type T" << next_uid << " = struct { " << desc << " };\n\n";
        _struct_uid_map.emplace(type, next_uid);
        return next_uid;
    }

    [[nodiscard]] luisa::string _type_ident(const Type *type) noexcept {
        if (type == nullptr) { return "void"; }
        switch (type->tag()) {
            case Type::Tag::BOOL: return "bool";
            case Type::Tag::INT8: return "i8";
            case Type::Tag::UINT8: return "u8";
            case Type::Tag::INT16: return "i16";
            case Type::Tag::UINT16: return "u16";
            case Type::Tag::INT32: return "i32";
            case Type::Tag::UINT32: return "u32";
            case Type::Tag::INT64: return "i64";
            case Type::Tag::UINT64: return "u64";
            case Type::Tag::FLOAT16: return "f16";
            case Type::Tag::FLOAT32: return "f32";
            case Type::Tag::FLOAT64: return "f64";
            case Type::Tag::VECTOR: return luisa::format("vector<{}, {}>", _type_ident(type->element()), type->dimension());
            case Type::Tag::MATRIX: return luisa::format("matrix<{}, {}>", _type_ident(type->element()), type->dimension());
            case Type::Tag::ARRAY: return luisa::format("array<{}, {}>", _type_ident(type->element()), type->dimension());
            case Type::Tag::STRUCTURE: return luisa::format("T{}", _struct_uid(type));
            case Type::Tag::BUFFER: return luisa::format("buffer<{}>", _type_ident(type->element()));
            case Type::Tag::TEXTURE: return luisa::format("texture{}d<{}>", type->dimension(), _type_ident(type->element()));
            case Type::Tag::BINDLESS_ARRAY: return "bindless_array";
            case Type::Tag::ACCEL: return "accel";
            case Type::Tag::CUSTOM: return luisa::format("T{}", _struct_uid(type));
        }
        LUISA_ERROR_WITH_LOCATION("Unknown type tag.");
    }

    void _emit_indent(int indent) noexcept {
        for (int i = 0; i < indent; i++) { _main << "    "; }
    }

    static void _emit_string_escaped(StringScratch &ss, luisa::string_view s) noexcept {
        ss << "\"";
        for (auto c : s) {
            if (isprint(c)) {
                switch (c) {
                    case '\n': ss << "\\n"; break;
                    case '\r': ss << "\\r"; break;
                    case '\t': ss << "\\t"; break;
                    case '\"': ss << "\\\""; break;
                    case '\\': ss << "\\\\"; break;
                    default: ss << luisa::string_view{&c, 1}; break;
                }
            } else {
                ss << "\\x" << (c >> 4u) << (c & 0x0fu);
            }
        }
        ss << "\"";
    }

    void _emit_operands(const Instruction *inst) noexcept {
        for (auto &&o : inst->operand_uses()) {
            _main << _value_ident(o->value()) << ", ";
        }
        if (!inst->operand_uses().empty()) {
            _main.pop_back();
            _main.pop_back();
        }
    }

    void _emit_if_inst(const IfInst *inst, int indent) noexcept {
        _main << "if " << _value_ident(inst->condition()) << ", then ";
        _emit_basic_block(inst->true_block(), indent);
        _main << ", else ";
        _emit_basic_block(inst->false_block(), indent);
        _main << ", merge ";
        _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_switch_inst(const SwitchInst *inst, int indent) noexcept {
        _main << "switch " << _value_ident(inst->value()) << ", ";
        for (auto i = 0u; i < inst->case_count(); i++) {
            _main << "case " << inst->case_value(i) << " ";
            _emit_basic_block(inst->case_block(i), indent);
            _main << ", ";
        }
        _main << "default ";
        _emit_basic_block(inst->default_block(), indent);
        _main << ", merge ";
        _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_loop_inst(const LoopInst *inst, int indent) noexcept {
        _main << "loop prepare ";
        _emit_basic_block(inst->prepare_block(), indent);
        _main << ", body ";
        _emit_basic_block(inst->body_block(), indent);
        _main << ", update ";
        _emit_basic_block(inst->update_block(), indent);
        _main << ", merge ";
        _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_outline_inst(const OutlineInst *inst, int indent) noexcept {
        _main << "outline body ";
        _emit_basic_block(inst->target_block(), indent);
        _main << ", merge ";
        _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_ray_query_inst(const RayQueryInst *inst, int indent) noexcept {
        _main << "ray_query " << _value_ident(inst->query_object()) << ", on_surface_candidate ";
        _emit_basic_block(inst->on_surface_candidate_block(), indent);
        _main << ", on_procedural_candidate ";
        _emit_basic_block(inst->on_procedural_candidate_block(), indent);
        _main << ", merge ";
        _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_return_inst(const ReturnInst *inst) noexcept {
        if (auto ret = inst->return_value()) {
            _main << "return " << _value_ident(ret);
        } else {
            _main << "return";
        }
    }

    void _emit_phi_inst(const PhiInst *inst) noexcept {
        _main << "phi";
        for (auto i = 0u; i < inst->incoming_count(); i++) {
            auto incoming = inst->incoming(i);
            _main << " (" << _value_ident(incoming.value) << ", "
                  << _value_ident(incoming.block) << "),";
        }
        if (inst->incoming_count() != 0u) { _main.pop_back(); }
    }

    void _emit_alloca_inst(const AllocaInst *inst) noexcept {
        _main << "alloca ";
        switch (inst->space()) {
            case AllocSpace::LOCAL: _main << "local"; break;
            case AllocSpace::SHARED: _main << "shared"; break;
        }
    }

    void _emit_load_inst(const LoadInst *inst) noexcept {
        _main << "load ";
        _emit_operands(inst);
    }

    void _emit_store_inst(const StoreInst *inst) noexcept {
        _main << "store ";
        _emit_operands(inst);
    }

    void _emit_gep_inst(const GEPInst *inst) noexcept {
        _main << "getelementptr ";
        _emit_operands(inst);
    }

    void _emit_call_inst(const CallInst *inst) noexcept {
        _main << "call ";
        _emit_operands(inst);
    }

    void _emit_intrinsic_inst(const IntrinsicInst *inst) noexcept {
        _main << "@" << to_string(inst->op());
        if (!inst->operand_uses().empty()) {
            _main << " ";
            _emit_operands(inst);
        }
    }

    void _emit_cast_inst(const CastInst *inst) noexcept {
        _main << "cast ";
        switch (inst->op()) {
            case CastOp::STATIC_CAST: _main << "static"; break;
            case CastOp::BITWISE_CAST: _main << "bitwise"; break;
        }
    }

    void _emit_print_inst(const PrintInst *inst) noexcept {
        _main << "print ";
        _emit_string_escaped(_main, inst->format());
        _main << " ";
        _emit_operands(inst);
    }

    void _emit_branch_inst(const BranchInst *inst) noexcept {
        LUISA_DEBUG_ASSERT(inst->target_block() != nullptr,
                           "Branch target block must not be null.");
        _main << "br " << _value_ident(inst->target_block());
    }

    void _emit_conditional_branch_inst(const ConditionalBranchInst *inst) noexcept {
        LUISA_DEBUG_ASSERT(inst->true_block() != nullptr && inst->false_block() != nullptr,
                           "Conditional branch target blocks must not be null.");
        _main << "cond_br "
              << _value_ident(inst->condition()) << ", "
              << _value_ident(inst->true_block()) << ", "
              << _value_ident(inst->false_block());
    }

    void _emit_instruction(const Instruction *inst, int indent) noexcept {
        if (!inst->metadata_list().empty()) {
            _emit_indent(indent);
            _emit_metadata_list(_main, inst->metadata_list());
            _main << "\n";
        }
        _emit_indent(indent);
        if (auto t = inst->type()) {
            _main << _value_ident(inst) << ": " << _type_ident(t) << " = ";
        }
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::SENTINEL:
                LUISA_ERROR_WITH_LOCATION("Unexpected sentinel instruction.");
            case DerivedInstructionTag::UNREACHABLE:
                _main << "unreachable";
                break;
            case DerivedInstructionTag::IF:
                _emit_if_inst(static_cast<const IfInst *>(inst), indent);
                break;
            case DerivedInstructionTag::SWITCH:
                _emit_switch_inst(static_cast<const SwitchInst *>(inst), indent);
                break;
            case DerivedInstructionTag::LOOP:
                _emit_loop_inst(static_cast<const LoopInst *>(inst), indent);
                break;
            case DerivedInstructionTag::BREAK:
                _main << "break";
                break;
            case DerivedInstructionTag::CONTINUE:
                _main << "continue";
                break;
            case DerivedInstructionTag::RETURN:
                _emit_return_inst(static_cast<const ReturnInst *>(inst));
                break;
            case DerivedInstructionTag::PHI:
                _emit_phi_inst(static_cast<const PhiInst *>(inst));
                break;
            case DerivedInstructionTag::ALLOCA:
                _emit_alloca_inst(static_cast<const AllocaInst *>(inst));
                break;
            case DerivedInstructionTag::LOAD:
                _emit_load_inst(static_cast<const LoadInst *>(inst));
                break;
            case DerivedInstructionTag::STORE:
                _emit_store_inst(static_cast<const StoreInst *>(inst));
                break;
            case DerivedInstructionTag::GEP:
                _emit_gep_inst(static_cast<const GEPInst *>(inst));
                break;
            case DerivedInstructionTag::CALL:
                _emit_call_inst(static_cast<const CallInst *>(inst));
                break;
            case DerivedInstructionTag::INTRINSIC:
                _emit_intrinsic_inst(static_cast<const IntrinsicInst *>(inst));
                break;
            case DerivedInstructionTag::CAST:
                _emit_cast_inst(static_cast<const CastInst *>(inst));
                break;
            case DerivedInstructionTag::PRINT:
                _emit_print_inst(static_cast<const PrintInst *>(inst));
                break;
            case DerivedInstructionTag::OUTLINE:
                _emit_outline_inst(static_cast<const OutlineInst *>(inst), indent);
                break;
            case DerivedInstructionTag::AUTO_DIFF: LUISA_NOT_IMPLEMENTED();
            case DerivedInstructionTag::RAY_QUERY:
                _emit_ray_query_inst(static_cast<const RayQueryInst *>(inst), indent);
                break;
            case DerivedInstructionTag::BRANCH:
                _emit_branch_inst(static_cast<const BranchInst *>(inst));
                break;
            case DerivedInstructionTag::CONDITIONAL_BRANCH:
                _emit_conditional_branch_inst(static_cast<const ConditionalBranchInst *>(inst));
                break;
        }
        _main << ";\n";
    }

    void _emit_basic_block(const BasicBlock *b, int indent) noexcept {
        if (b == nullptr) {
            _main << "null";
        } else {
            if (!b->metadata_list().empty()) {
                _emit_metadata_list(_main, b->metadata_list());
                _main << " ";
            }
            _main << _value_ident(b) << ": {\n";
            for (auto &&inst : b->instructions()) {
                _emit_instruction(&inst, indent + 1);
            }
            _emit_indent(indent);
            _main << "}";
        }
    }

    void _emit_function(const Function *f) noexcept {
        if (!f->metadata_list().empty()) {
            _emit_metadata_list(_main, f->metadata_list());
            _main << "\n";
        }
        switch (f->function_tag()) {
            case FunctionTag::KERNEL: _main << "kernel " << _value_ident(f); break;
            case FunctionTag::CALLABLE: _main << "callable " << _value_ident(f) << ": " << _type_ident(f->type()); break;
        }
        _main << " = (";
        // TODO: metadata
        if (!f->arguments().empty()) { _main << "\n"; }
        for (auto arg : f->arguments()) {
            if (!arg->metadata_list().empty()) {
                _emit_indent(1);
                _emit_metadata_list(_main, arg->metadata_list());
                _main << "\n";
            }
            _emit_indent(1);
            _main << _value_ident(arg) << ": ";
            if (arg->derived_value_tag() == DerivedValueTag::REFERENCE_ARGUMENT) {
                _main << "&";
            }
            _main << _type_ident(arg->type()) << ";\n";
        }
        _main << "), body ";
        _emit_basic_block(f->body_block(), 0);
        _main << ";\n\n";
    }

    void _emit_module(const Module *module) noexcept {
        if (!module->metadata_list().empty()) {
            _emit_metadata_list(_prelude, module->metadata_list());
            _prelude << "\n";
        }
        _prelude << "module;\n\n";// TODO: metadata
        for (auto &c : module->constants()) { _emit_constant(&c); }
        for (auto &f : module->functions()) { _emit_function(&f); }
    }

    static void _emit_name_metadata(StringScratch &s, const NameMD &m) noexcept {
        s << "name = " << m.name();
    }

    static void _emit_location_metadata(StringScratch &s, const LocationMD &m) noexcept {
        s << "location = (";
        _emit_string_escaped(s, m.file().string());
        s << ", " << m.line() << ")";
    }

    static void _emit_comment_metadata(StringScratch &s, const CommentMD &m) noexcept {
        s << "comment = ";
        _emit_string_escaped(s, m.comment());
    }

    static void _emit_metadata_list(StringScratch &s, const MetadataList &m) noexcept {
        s << "[";
        for (auto &item : m) {
            switch (item.derived_metadata_tag()) {
                case DerivedMetadataTag::NAME:
                    _emit_name_metadata(s, static_cast<const NameMD &>(item));
                    break;
                case DerivedMetadataTag::LOCATION:
                    _emit_location_metadata(s, static_cast<const LocationMD &>(item));
                    break;
                case DerivedMetadataTag::COMMENT:
                    _emit_comment_metadata(s, static_cast<const CommentMD &>(item));
                    break;
                default: LUISA_NOT_IMPLEMENTED();
            }
            s << ", ";
        }
        if (!m.empty()) {
            s.pop_back();
            s.pop_back();
        }
        s << "]";
    }

public:
    XIR2TextTranslator() noexcept : _prelude{1_k}, _main{4_k} {}
    [[nodiscard]] luisa::string emit(const Module &module) noexcept {
        _prelude.clear();
        _main.clear();
        _value_uid_map.clear();
        _emit_module(&module);
        return _prelude.string() + _main.string();
    }
};

luisa::string translate_to_text(const Module &module) noexcept {
    return XIR2TextTranslator{}.emit(module);
}

}// namespace luisa::compute::xir
