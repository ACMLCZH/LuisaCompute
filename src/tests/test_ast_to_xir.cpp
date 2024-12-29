#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main() {

    Callable f = [](UInt a, UInt b) noexcept {
        $if (a == 0u) {
            a = 1u;
        };
        $for (i, 3) {
            b += i;
        };
        auto delta = thread_x();
        return a * b + delta;
    };

    Kernel1D k = [&](BufferUInt buffer, UInt y) noexcept {
        $comment("i");
        auto i = dispatch_x();
        $comment("x");
        auto x = buffer.read(i);
        $comment("buffer");
        buffer.write(i, f(x, y));
    };

    xir::Pool pool;
    xir::PoolGuard guard{&pool};

    auto module = xir::ast_to_xir_translate(k.function()->function(), {});
    auto text = xir::xir_to_text_translate(module, true);

    LUISA_INFO("AST2IR:\n{}", text);
}
