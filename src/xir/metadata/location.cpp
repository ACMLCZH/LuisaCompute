#include <luisa/xir/metadata/location.h>

namespace luisa::compute::xir {

LocationMD::LocationMD(luisa::filesystem::path file, int line) noexcept
    : _file{std::move(file)}, _line{line} {}

void LocationMD::set_location(luisa::filesystem::path file, int line) noexcept {
    set_file(std::move(file));
    set_line(line);
}

}// namespace luisa::compute::xir
