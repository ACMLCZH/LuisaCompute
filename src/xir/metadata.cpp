#include <luisa/core/logging.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute::xir::detail {

Metadata *metadata_find(DerivedMetadataTag tag, MetadataList &list) noexcept {
    for (auto &m : list) {
        if (m.derived_metadata_tag() == tag) {
            return &m;
        }
    }
    return nullptr;
}

Metadata *metadata_find_or_create(Pool *pool, DerivedMetadataTag tag, MetadataList &list) noexcept {
    if (auto m = metadata_find(tag, list); m != nullptr) {
        return m;
    }
    switch (tag) {
        case DerivedMetadataTag::NAME: return pool->create<NameMD>();
        case DerivedMetadataTag::LOCATION: return pool->create<LocationMD>();
        case DerivedMetadataTag::COMMENT: return pool->create<CommentMD>();
    }
    LUISA_ERROR_WITH_LOCATION("Unknown derived metadata tag 0x{:x}.",
                              static_cast<uint32_t>(tag));
}

void metadata_set_or_create_name(Pool *pool, MetadataList &list,
                                 luisa::string_view name) noexcept {
    auto m = static_cast<NameMD *>(metadata_find_or_create(pool, DerivedMetadataTag::NAME, list));
    m->set_name(name);
}

void metadata_set_or_create_location(Pool *pool, MetadataList &list,
                                     const luisa::filesystem::path &file,
                                     int line, int column) noexcept {
    auto m = static_cast<LocationMD *>(metadata_find_or_create(pool, DerivedMetadataTag::LOCATION, list));
    m->set_file(file);
    m->set_line(line);
    m->set_column(column);
}

void metadata_add_comment(Pool *pool, MetadataList &list,
                          luisa::string_view comment) noexcept {
    auto m = pool->create<CommentMD>(luisa::string{comment});
    m->add_to_list(list);
}

}// namespace luisa::compute::xir::detail
