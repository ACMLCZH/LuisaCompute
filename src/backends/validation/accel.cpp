#include "mesh.h"
#include "procedural_primitives.h"
#include "buffer.h"
#include "accel.h"
#include <core/logging.h>
namespace lc::validation {
void Mesh::set(Stream *stream, Usage usage) {
    set_usage(stream, this, usage);
    LUISA_ASSERT(vert, "{}'s vertex-buffer must be set before use.", get_name());
    set_usage(stream, vert, Usage::READ);
    LUISA_ASSERT(index, "{}'s index-buffer must be set before use.", get_name());
    set_usage(stream, index, Usage::READ);
}
void ProceduralPrimitives::set(Stream *stream, Usage usage) {
    set_usage(stream, this, usage);
    LUISA_ASSERT(bbox, "{}'s bounding-boxes must be set before use.", get_name());
    set_usage(stream, bbox, Usage::READ);
}
void Accel::set(Stream *stream, Usage usage) {
    set_usage(stream, this, usage);
    for (auto &&i : _ref_count) {
        set_usage(stream, reinterpret_cast<RWResource *>(i.first), Usage::READ);
    }
}
void Accel::modify(size_t size, Stream *stream, luisa::span<AccelBuildCommand::Modification const> modifies) {
    auto last_size = _meshes.size();
    _meshes.resize(size);
    for (auto &&i : modifies) {
        auto &mesh = _meshes[i.index];
        if (mesh) {
            auto iter = _ref_count.find(mesh);
            if (iter != _ref_count.end()) {
                if (--iter->second == 0) {
                    _ref_count.erase(iter);
                }
            }
        }
        mesh = i.primitive;
        if (mesh && mesh != invalid_resource_handle) {
            auto iter = _ref_count.try_emplace(mesh, 0);
            iter.first->second++;
        } else {
            LUISA_ERROR("Accel modification must have primitive.");
        }
    }
    if (last_size < _meshes.size()) {
        for (auto i : vstd::range(last_size, _meshes.size())) {
            if (_meshes[i] && _meshes[i] != invalid_resource_handle) continue;
            LUISA_ERROR("Accel instance {} catch invalid primitive handle", i);
        }
    }
}
}// namespace lc::validation