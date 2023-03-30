#pragma once

#if LC_NVRTC_VERSION < 110200
#define LC_CONSTANT const
#else
#define LC_CONSTANT constexpr
#endif

#if LC_NVRTC_VERSION < 110200
inline __device__ void lc_assume(bool) noexcept {}
#else
#define lc_assume(...) __builtin_assume(__VA_ARGS__)
#endif

template<typename T>
[[noreturn]] inline __device__ T lc_unreachable() noexcept {
#if LC_NVRTC_VERSION < 110300
    asm("trap;");
#else
    __builtin_unreachable();
#endif
}

#ifdef LC_DEBUG
#define lc_assert(...)                                \
    do {                                              \
        if (!(__VA_ARGS__)) {                         \
            printf("Assertion failed: %s, %s:%d\n",   \
                   #__VA_ARGS__, __FILE__, __LINE__); \
            asm("trap;");                             \
        }                                             \
    } while (false)
#else
inline __device__ void lc_assert(bool) noexcept {}
#endif

struct lc_half {
    unsigned short bits;
};

struct alignas(4) lc_half2 {
    lc_half x, y;
};

struct alignas(8) lc_half4 {
    lc_half x, y, z, w;
};

[[nodiscard]] __device__ inline auto lc_half_to_float(lc_half x) noexcept {
    lc_float val;
    asm("{  cvt.f32.f16 %0, %1;}\n"
        : "=f"(val)
        : "h"(x.bits));
    return val;
}

[[nodiscard]] __device__ inline auto lc_float_to_half(lc_float x) noexcept {
    lc_half val;
    asm("{  cvt.rn.f16.f32 %0, %1;}\n"
        : "=h"(val.bits)
        : "f"(x));
    return val;
}

template<typename T>
struct LCBuffer {
    T *__restrict__ ptr;
    size_t size_bytes;
};

template<typename T>
struct LCBuffer<const T> {
    const T *__restrict__ ptr;
    size_t size_bytes;
    LCBuffer(LCBuffer<T> buffer) noexcept
        : ptr{buffer.ptr}, size_bytes{buffer.size_bytes} {}
    LCBuffer() noexcept = default;
};

template<typename T, typename Index>
[[nodiscard]] __device__ inline auto lc_buffer_read(LCBuffer<T> buffer, Index index) noexcept {
    return buffer.ptr[index];
}

template<typename T, typename Index>
__device__ inline void lc_buffer_write(LCBuffer<T> buffer, Index index, T value) noexcept {
    buffer.ptr[index] = value;
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_buffer_size(LCBuffer<T> buffer) noexcept {
    return buffer.size_bytes / sizeof(T);
}

enum struct LCPixelStorage {

    BYTE1,
    BYTE2,
    BYTE4,

    SHORT1,
    SHORT2,
    SHORT4,

    INT1,
    INT2,
    INT4,

    HALF1,
    HALF2,
    HALF4,

    FLOAT1,
    FLOAT2,
    FLOAT4
};

struct alignas(16) LCSurface {
    cudaSurfaceObject_t handle;
    unsigned long long storage;
};

static_assert(sizeof(LCSurface) == 16);

template<typename A, typename B>
struct lc_is_same {
    [[nodiscard]] static constexpr auto value() noexcept { return false; };
};

template<typename A>
struct lc_is_same<A, A> {
    [[nodiscard]] static constexpr auto value() noexcept { return true; };
};

template<typename...>
struct lc_always_false {
    [[nodiscard]] static constexpr auto value() noexcept { return false; };
};

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_float(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<unsigned char>(x) * (1.0f / 255.0f);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<unsigned short>(x) * (1.0f / 65535.0f);
    } else if constexpr (lc_is_same<P, lc_half>::value()) {
        return lc_half_to_float(x);
    } else if constexpr (lc_is_same<P, lc_float>::value()) {
        return x;
    }
    return 0.0f;
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_int(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<lc_int>(x);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<lc_int>(x);
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return x;
    }
    return 0;
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_uint(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<lc_uint>(static_cast<unsigned char>(x));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<lc_uint>(static_cast<unsigned short>(x));
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return static_cast<lc_uint>(x);
    }
    return 0u;
}

template<typename T, typename P>
[[nodiscard]] __device__ inline auto lc_texel_read_convert(P p) noexcept {
    if constexpr (lc_is_same<T, lc_float>::value()) {
        return lc_texel_to_float<P>(p);
    } else if constexpr (lc_is_same<T, lc_int>::value()) {
        return lc_texel_to_int<P>(p);
    } else if constexpr (lc_is_same<T, lc_uint>::value()) {
        return lc_texel_to_uint<P>(p);
    } else {
        static_assert(lc_always_false<T, P>::value());
    }
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_float_to_texel(lc_float x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(static_cast<unsigned char>(lc_round(lc_saturate(x) * 255.0f)));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(static_cast<unsigned short>(lc_round(lc_saturate(x) * 65535.0f)));
    } else if constexpr (lc_is_same<P, lc_half>::value()) {
        return lc_float_to_half(x);
    } else if constexpr (lc_is_same<P, lc_float>::value()) {
        return x;
    }
    return P{};
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_int_to_texel(lc_int x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(x);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(x);
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return x;
    }
    return P{};
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_uint_to_texel(lc_uint x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(static_cast<unsigned char>(x));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(static_cast<unsigned short>(x));
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return static_cast<lc_int>(x);
    }
    return P{};
}

template<typename P, typename T>
[[nodiscard]] __device__ inline auto lc_texel_write_convert(T t) noexcept {
    if constexpr (lc_is_same<T, lc_float>::value()) {
        return lc_float_to_texel<P>(t);
    } else if constexpr (lc_is_same<T, lc_int>::value()) {
        return lc_int_to_texel<P>(t);
    } else if constexpr (lc_is_same<T, lc_uint>::value()) {
        return lc_uint_to_texel<P>(t);
    } else {
        static_assert(lc_always_false<T, P>::value());
    }
}

template<typename T>
struct lc_vec4 {};

template<>
struct lc_vec4<lc_int> {
    using type = lc_int4;
};

template<>
struct lc_vec4<lc_uint> {
    using type = lc_uint4;
};

template<>
struct lc_vec4<lc_float> {
    using type = lc_float4;
};

template<typename T>
using lc_vec4_t = typename lc_vec4<T>::type;

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf2d_read(LCSurface surf, lc_uint2 p) noexcept {
    lc_vec4_t<T> result{0, 0, 0, 0};
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int x;
            asm("suld.b.2d.b8.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            break;
        }
        case LCPixelStorage::BYTE2: {
            int x, y;
            asm("suld.b.2d.v2.b8.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            break;
        }
        case LCPixelStorage::BYTE4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b8.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            result.z = lc_texel_read_convert<T, char>(z);
            result.w = lc_texel_read_convert<T, char>(w);
            break;
        }
        case LCPixelStorage::SHORT1: {
            int x;
            asm("suld.b.2d.b16.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            break;
        }
        case LCPixelStorage::SHORT2: {
            int x, y;
            asm("suld.b.2d.v2.b16.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            break;
        }
        case LCPixelStorage::SHORT4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            result.z = lc_texel_read_convert<T, short>(z);
            result.w = lc_texel_read_convert<T, short>(w);
            break;
        }
        case LCPixelStorage::INT1: {
            int x;
            asm("suld.b.2d.b32.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            break;
        }
        case LCPixelStorage::INT2: {
            int x, y;
            asm("suld.b.2d.v2.b32.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            break;
        }
        case LCPixelStorage::INT4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            result.z = lc_texel_read_convert<T, int>(z);
            result.w = lc_texel_read_convert<T, int>(w);
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint x;
            asm("suld.b.2d.b16.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint x, y;
            asm("suld.b.2d.v2.b16.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            result.y = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(y)});
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint x, y, z, w;
            asm("suld.b.2d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            result.y = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(y)});
            result.z = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(z)});
            result.w = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(w)});
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float x;
            asm("suld.b.2d.b32.zero %0, [%1, {%2, %3}];"
                : "=f"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float x, y;
            asm("suld.b.2d.v2.b32.zero {%0, %1}, [%2, {%3, %4}];"
                : "=f"(x), "=f"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float x, y, z, w;
            asm("suld.b.2d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            result.z = lc_texel_read_convert<T, float>(z);
            result.w = lc_texel_read_convert<T, float>(w);
            break;
        }
        default: __builtin_unreachable();
    }
    return result;
}

template<typename T, typename V>
__device__ inline void lc_surf2d_write(LCSurface surf, lc_uint2 p, V value) noexcept {
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int v = lc_texel_write_convert<char>(value.x);
            asm volatile("sust.b.2d.b8.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE2: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            asm volatile("sust.b.2d.v2.b8.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE4: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            int vz = lc_texel_write_convert<char>(value.z);
            int vw = lc_texel_write_convert<char>(value.w);
            asm volatile("sust.b.2d.v4.b8.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT1: {
            int v = lc_texel_write_convert<short>(value.x);
            asm volatile("sust.b.2d.b16.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT2: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            asm volatile("sust.b.2d.v2.b16.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT4: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            int vz = lc_texel_write_convert<short>(value.z);
            int vw = lc_texel_write_convert<short>(value.w);
            asm volatile("sust.b.2d.v4.b16.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT1: {
            int v = lc_texel_write_convert<int>(value.x);
            asm volatile("sust.b.2d.b32.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT2: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            asm volatile("sust.b.2d.v2.b32.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT4: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            int vz = lc_texel_write_convert<int>(value.z);
            int vw = lc_texel_write_convert<int>(value.w);
            asm volatile("sust.b.2d.v4.b32.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint v = lc_texel_write_convert<lc_half>(value.x).bits;
            asm volatile("sust.b.2d.b16.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            asm volatile("sust.b.2d.v2.b16.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            lc_uint vz = lc_texel_write_convert<lc_half>(value.z).bits;
            lc_uint vw = lc_texel_write_convert<lc_half>(value.w).bits;
            asm volatile("sust.b.2d.v4.b16.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float v = lc_texel_write_convert<float>(value.x);
            asm volatile("sust.b.2d.b32.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float))), "r"(p.y), "f"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            asm volatile("sust.b.2d.v2.b32.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float2))), "r"(p.y), "f"(vx), "f"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            float vz = lc_texel_write_convert<float>(value.z);
            float vw = lc_texel_write_convert<float>(value.w);
            asm volatile("sust.b.2d.v4.b32.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float4))), "r"(p.y), "f"(vx), "f"(vy), "f"(vz), "f"(vw)
                         : "memory");
            break;
        }
        default: __builtin_unreachable();
    }
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf3d_read(LCSurface surf, lc_uint3 p) noexcept {
    lc_vec4_t<T> result{0, 0, 0, 0};
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int x;
            asm("suld.b.3d.b8.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            break;
        }
        case LCPixelStorage::BYTE2: {
            int x, y;
            asm("suld.b.3d.v2.b8.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert < T, char(x);
            result.y = lc_texel_read_convert < T, char(y);
            break;
        }
        case LCPixelStorage::BYTE4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b8.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            result.z = lc_texel_read_convert<T, char>(z);
            result.w = lc_texel_read_convert<T, char>(w);
            break;
        }
        case LCPixelStorage::SHORT1: {
            int x;
            asm("suld.b.3d.b16.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            break;
        }
        case LCPixelStorage::SHORT2: {
            int x, y;
            asm("suld.b.3d.v2.b16.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            break;
        }
        case LCPixelStorage::SHORT4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            result.z = lc_texel_read_convert<T, short>(z);
            result.w = lc_texel_read_convert<T, short>(w);
            break;
        }
        case LCPixelStorage::INT1: {
            int x;
            asm("suld.b.3d.b32.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            break;
        }
        case LCPixelStorage::INT2: {
            int x, y;
            asm("suld.b.3d.v2.b32.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            break;
        }
        case LCPixelStorage::INT4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            result.z = lc_texel_read_convert<T, int>(z);
            result.w = lc_texel_read_convert<T, int>(w);
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint x;
            asm("suld.b.3d.b16.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint x, y;
            asm("suld.b.3d.v2.b16.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            result.y = lc_texel_read_convert<T, lc_half>(y);
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint x, y, z, w;
            asm("suld.b.3d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            result.y = lc_texel_read_convert<T, lc_half>(y);
            result.z = lc_texel_read_convert<T, lc_half>(z);
            result.w = lc_texel_read_convert<T, lc_half>(w);
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float x;
            asm("suld.b.3d.b32.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=f"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float x, y;
            asm("suld.b.3d.v2.b32.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=f"(x), "=f"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float x, y, z, w;
            asm("suld.b.3d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            result.z = lc_texel_read_convert<T, float>(z);
            result.w = lc_texel_read_convert<T, float>(w);
            break;
        }
        default: __builtin_unreachable();
    }
    return result;
}

template<typename T, typename V>
__device__ inline void lc_surf3d_write(LCSurface surf, lc_uint3 p, V value) noexcept {
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int v = lc_texel_write_convert<char>(value.x);
            asm volatile("sust.b.3d.b8.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE2: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            asm volatile("sust.b.3d.v2.b8.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE4: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            int vz = lc_texel_write_convert<char>(value.z);
            int vw = lc_texel_write_convert<char>(value.w);
            asm volatile("sust.b.3d.v4.b8.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT1: {
            int v = lc_texel_write_convert<short>(value.x);
            asm volatile("sust.b.3d.b16.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT2: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            asm volatile("sust.b.3d.v2.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT4: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            int vz = lc_texel_write_convert<short>(value.z);
            int vw = lc_texel_write_convert<short>(value.w);
            asm volatile("sust.b.3d.v4.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT1: {
            int v = lc_texel_write_convert<int>(value.x);
            asm volatile("sust.b.3d.b32.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT2: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            asm volatile("sust.b.3d.v2.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT4: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            int vz = lc_texel_write_convert<int>(value.z);
            int vw = lc_texel_write_convert<int>(value.w);
            asm volatile("sust.b.3d.v4.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint v = lc_texel_write_convert<lc_half>(value.x).bits;
            asm volatile("sust.b.3d.b16.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            asm volatile("sust.b.3d.v2.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            lc_uint vz = lc_texel_write_convert<lc_half>(value.z).bits;
            lc_uint vw = lc_texel_write_convert<lc_half>(value.w).bits;
            asm volatile("sust.b.3d.v4.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float v = lc_texel_write_convert<float>(value.x);
            asm volatile("sust.b.3d.b32.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float))), "r"(p.y), "r"(p.z), "r"(0), "f"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            asm volatile("sust.b.3d.v2.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float2))), "r"(p.y), "r"(p.z), "r"(0), "f"(vx), "f"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            float vz = lc_texel_write_convert<float>(value.z);
            float vw = lc_texel_write_convert<float>(value.w);
            asm volatile("sust.b.3d.v4.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float4))), "r"(p.y), "r"(p.z), "r"(0), "f"(vx), "f"(vy), "f"(vz), "f"(vw)
                         : "memory");
            break;
        }
        default: __builtin_unreachable();
    }
}

template<typename T>
struct LCTexture2D {
    LCSurface surface;
};

template<typename T>
struct LCTexture3D {
    LCSurface surface;
};

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture2D<T> tex, lc_uint2 p) noexcept {
    return lc_surf2d_read<T>(tex.surface, p);
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture3D<T> tex, lc_uint3 p) noexcept {
    return lc_surf3d_read<T>(tex.surface, p);
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture2D<T> tex, lc_uint2 p, V value) noexcept {
    lc_surf2d_write<T>(tex.surface, p, value);
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture3D<T> tex, lc_uint3 p, V value) noexcept {
    lc_surf3d_write<T>(tex.surface, p, value);
}

struct alignas(16) LCBindlessSlot {
    const void *__restrict__ buffer;
    size_t buffer_size;
    cudaTextureObject_t tex2d;
    cudaTextureObject_t tex3d;
};

struct alignas(16) LCBindlessArray {
    const LCBindlessSlot *__restrict__ slots;
};

template<typename T>
[[nodiscard]] inline __device__ auto lc_bindless_buffer_read(LCBindlessArray array, lc_uint index, lc_uint i) noexcept {
    auto buffer = static_cast<const T *>(array.slots[index].buffer);
    return buffer[i];
}

template<typename T = unsigned char>
[[nodiscard]] inline __device__ auto lc_bindless_buffer_size(LCBindlessArray array, lc_uint index) noexcept {
    return array.slots[index].buffer_size / sizeof(T);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d(LCBindlessArray array, lc_uint index, lc_float2 p) noexcept {
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d(LCBindlessArray array, lc_uint index, lc_float3 p) noexcept {
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d_level(LCBindlessArray array, lc_uint index, lc_float2 p, float level) noexcept {
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.level.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d_level(LCBindlessArray array, lc_uint index, lc_float3 p, float level) noexcept {
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f), "f"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d_grad(LCBindlessArray array, lc_uint index, lc_float2 p, lc_float2 dx, lc_float2 dy) noexcept {
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.grad.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, {%9, %10};"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(dx.x), "f"(dx.y), "f"(dy.x), "f"(dy.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d_grad(LCBindlessArray array, lc_uint index, lc_float3 p, lc_float3 dx, lc_float3 dy) noexcept {
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.grad.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], {%9, %10, %11, %12}, {%13, %14, %15, 16};"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f),
          "f"(dx.x), "f"(dx.y), "f"(dx.z), "f"(0.f),
          "f"(dy.x), "f"(dy.y), "f"(dy.z), "f"(0.f));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size2d(LCBindlessArray array, lc_uint index) noexcept {
    auto t = array.slots[index].tex2d;
    auto s = lc_make_uint2();
    asm("txq.width.b32 %0, [%1];"
        : "=r"(s.x)
        : "l"(t));
    asm("txq.height.b32 %0, [%1];"
        : "=r"(s.y)
        : "l"(t));
    return s;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size3d(LCBindlessArray array, lc_uint index) noexcept {
    auto t = array.slots[index].tex3d;
    auto s = lc_make_uint3();
    asm("txq.width.b32 %0, [%1];"
        : "=r"(s.x)
        : "l"(t));
    asm("txq.height.b32 %0, [%1];"
        : "=r"(s.y)
        : "l"(t));
    asm("txq.depth.b32 %0, [%1];"
        : "=r"(s.z)
        : "l"(t));
    return s;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size2d_level(LCBindlessArray array, lc_uint index, lc_uint level) noexcept {
    auto s = lc_bindless_texture_size2d(array, index);
    return lc_max(s >> level, lc_make_uint2(1u));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size3d_level(LCBindlessArray array, lc_uint index, lc_uint level) noexcept {
    auto s = lc_bindless_texture_size3d(array, index);
    return lc_max(s >> level, lc_make_uint3(1u));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read2d(LCBindlessArray array, lc_uint index, lc_uint2 p) noexcept {
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read3d(LCBindlessArray array, lc_uint index, lc_uint3 p) noexcept {
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(p.z), "r"(0u));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read2d_level(LCBindlessArray array, lc_uint index, lc_uint2 p, lc_uint level) noexcept {
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.level.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read3d_level(LCBindlessArray array, lc_uint index, lc_uint3 p, lc_uint level) noexcept {
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.level.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(p.z), "r"(0u), "r"(level));
    return v;
}

struct alignas(16) LCRay {
    lc_array<float, 3> m0;// origin
    float m1;             // t_min
    lc_array<float, 3> m2;// direction
    float m3;             // t_max
};

struct LCTriangleHit {
    lc_uint m0;  // instance index
    lc_uint m1;  // primitive index
    lc_float2 m2;// barycentric coordinates
    lc_float m3; // t_hit
};

struct LCProceduralHit {
    lc_uint m0;// instance index
    lc_uint m1;// primitive index
};

enum struct LCHitType {
    MISS = 0,
    TRIANGLE = 1,
    PROCEDURAL = 2,
};

struct LCCommittedHit {
    lc_uint m0;  // instance index
    lc_uint m1;  // primitive index
    lc_float2 m2;// baricentric coordinates
    lc_uint m3;  // hit type
    lc_float m4; // t_hit
};

struct LCRayQuery {
    // TODO: add support for ray query
};

enum LCInstanceFlags : unsigned int {
    LC_INSTANCE_FLAG_NONE = 0u,
    LC_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 0u,
    LC_INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1u << 1u,
    LC_INSTANCE_FLAG_DISABLE_ANYHIT = 1u << 2u,
    LC_INSTANCE_FLAG_ENFORCE_ANYHIT = 1u << 3u,
};

struct alignas(16) LCAccelInstance {
    lc_array<lc_float4, 3> m;
    lc_uint instance_id;
    lc_uint sbt_offset;
    lc_uint mask;
    lc_uint flags;
    lc_uint pad[4];
};

struct alignas(16u) LCAccel {
    unsigned long long handle;
    LCAccelInstance *instances;
};

[[nodiscard]] __device__ inline auto lc_accel_instance_transform(LCAccel accel, lc_uint instance_id) noexcept {
    auto m = accel.instances[instance_id].m;
    return lc_make_float4x4(
        m[0].x, m[1].x, m[2].x, 0.0f,
        m[0].y, m[1].y, m[2].y, 0.0f,
        m[0].z, m[1].z, m[2].z, 0.0f,
        m[0].w, m[1].w, m[2].w, 1.0f);
}

__device__ inline void lc_accel_set_instance_transform(LCAccel accel, lc_uint index, lc_float4x4 m) noexcept {
    lc_array<lc_float4, 3> p;
    p[0].x = m[0][0];
    p[0].y = m[1][0];
    p[0].z = m[2][0];
    p[0].w = m[3][0];
    p[1].x = m[0][1];
    p[1].y = m[1][1];
    p[1].z = m[2][1];
    p[1].w = m[3][1];
    p[2].x = m[0][2];
    p[2].y = m[1][2];
    p[2].z = m[2][2];
    p[2].w = m[3][2];
}

__device__ inline void lc_accel_set_instance_visibility(LCAccel accel, lc_uint index, lc_uint mask) noexcept {
    accel.instances[index].mask = mask & 0xffu;
}

__device__ inline void lc_accel_set_instance_opacity(LCAccel accel, lc_uint index, bool opaque) noexcept {
    accel.instances[index].flags = opaque ?
                                       LC_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING | LC_INSTANCE_FLAG_DISABLE_ANYHIT :
                                       LC_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
}

__device__ inline float atomicCAS(float *a, float cmp, float v) noexcept {
    return __uint_as_float(atomicCAS(reinterpret_cast<lc_uint *>(a), __float_as_uint(cmp), __float_as_uint(v)));
}

__device__ inline float atomicSub(float *a, float v) noexcept {
    return atomicAdd(a, -v);
}

// is this valid?
__device__ inline float atomicMin(float *a, float v) noexcept {
    return __int_as_float(atomicMin(reinterpret_cast<int *>(a), __float_as_int(v)));
}

// is this valid?
__device__ inline float atomicMax(float *a, float v) noexcept {
    return __int_as_float(atomicMax(reinterpret_cast<int *>(a), __float_as_int(v)));
}

#define lc_atomic_exchange(buffer, index, value) atomicExch(&((buffer).ptr[index]), value)
#define lc_atomic_compare_exchange(buffer, index, cmp, value) atomicCAS(&((buffer).ptr[index]), cmp, value)
#define lc_atomic_fetch_add(buffer, index, value) atomicAdd(&((buffer).ptr[index]), value)
#define lc_atomic_fetch_sub(buffer, index, value) atomicSub(&((buffer).ptr[index]), value)
#define lc_atomic_fetch_min(buffer, index, value) atomicMin(&((buffer).ptr[index]), value)
#define lc_atomic_fetch_max(buffer, index, value) atomicMax(&((buffer).ptr[index]), value)
#define lc_atomic_fetch_and(buffer, index, value) atomicAnd(&((buffer).ptr[index]), value)
#define lc_atomic_fetch_or(buffer, index, value) atomicOr(&((buffer).ptr[index]), value)
#define lc_atomic_fetch_xor(buffer, index, value) atomicXor(&((buffer).ptr[index]), value)

// static block size
[[nodiscard]] __device__ constexpr lc_uint3 lc_block_size() noexcept {
    return LC_BLOCK_SIZE;
}

#ifdef LUISA_ENABLE_OPTIX

enum LCPayloadTypeID : unsigned int {
    LC_PAYLOAD_TYPE_DEFAULT = 0u,
    LC_PAYLOAD_TYPE_ID_0 = 1u << 0u,
    LC_PAYLOAD_TYPE_ID_1 = 1u << 1u,
    LC_PAYLOAD_TYPE_ID_2 = 1u << 2u,
    LC_PAYLOAD_TYPE_ID_3 = 1u << 3u,
    LC_PAYLOAD_TYPE_ID_4 = 1u << 4u,
    LC_PAYLOAD_TYPE_ID_5 = 1u << 5u,
    LC_PAYLOAD_TYPE_ID_6 = 1u << 6u,
    LC_PAYLOAD_TYPE_ID_7 = 1u << 7u,
};

inline void lc_set_payload_types(LCPayloadTypeID type) noexcept {
    asm volatile("call _optix_set_payload_types, (%0);"
                 :
                 : "r"(type)
                 :);
}

template<lc_uint i>
inline void lc_set_payload(lc_uint x) noexcept {
    asm volatile("call _optix_set_payload, (%0, %1);"
                 :
                 : "r"(i), "r"(x)
                 :);
}

[[nodiscard]] inline auto lc_get_primitive_index() noexcept {
    lc_uint u0;
    asm("call (%0), _optix_read_primitive_idx, ();"
        : "=r"(u0)
        :);
    return u0;
}

[[nodiscard]] inline auto lc_get_instance_index() noexcept {
    lc_uint u0;
    asm("call (%0), _optix_read_instance_idx, ();"
        : "=r"(u0)
        :);
    return u0;
}

[[nodiscard]] inline auto lc_get_bary_coords() noexcept {
    float f0, f1;
    asm("call (%0, %1), _optix_get_triangle_barycentrics, ();"
        : "=f"(f0), "=f"(f1)
        :);
    return lc_make_float2(f0, f1);
}

[[nodiscard]] inline auto lc_get_hit_distance() noexcept {
    float f0;
    asm("call (%0), _optix_get_ray_tmax, ();"
        : "=f"(f0)
        :);
    return f0;
}

extern "C" __global__ void __closesthit__trace_closest() {
    lc_set_payload_types(LC_PAYLOAD_TYPE_ID_0);
    auto inst = lc_get_instance_index();
    auto prim = lc_get_primitive_index();
    auto bary = lc_get_bary_coords();
    auto t_hit = lc_get_hit_distance();
    lc_set_payload<0u>(inst);
    lc_set_payload<1u>(prim);
    lc_set_payload<2u>(__float_as_uint(bary.x));
    lc_set_payload<3u>(__float_as_uint(bary.y));
    lc_set_payload<4u>(__float_as_uint(t_hit));
}

extern "C" __global__ void __miss__trace_closest() {
    lc_set_payload_types(LC_PAYLOAD_TYPE_ID_0);
    lc_set_payload<0u>(~0u);
}

extern "C" __global__ void __miss__trace_any() {
    lc_set_payload_types(LC_PAYLOAD_TYPE_ID_1);
    lc_set_payload<0u>(~0u);
}

[[nodiscard]] inline auto lc_undef() noexcept {
    auto u0 = 0u;
    asm("call (%0), _optix_undef_value, ();"
        : "=r"(u0)
        :);
    return u0;
}

template<lc_uint ray_type, lc_uint reg_count, lc_uint flags>
[[nodiscard]] inline auto lc_trace_impl(
    lc_uint payload_type, LCAccel accel, LCRay ray, lc_uint mask,
    lc_uint &r0, lc_uint &r1, lc_uint &r2, lc_uint &r3, lc_uint &r4) noexcept {
    auto ox = ray.m0[0];
    auto oy = ray.m0[1];
    auto oz = ray.m0[2];
    auto dx = ray.m2[0];
    auto dy = ray.m2[1];
    auto dz = ray.m2[2];
    auto t_min = ray.m1;
    auto t_max = ray.m3;
    auto u = lc_undef();
    [[maybe_unused]] unsigned int
        p5,
        p6, p7, p8, p9, p10, p11, p12, p13,
        p14, p15, p16, p17, p18, p19, p20, p21, p22,
        p23, p24, p25, p26, p27, p28, p29, p30, p31;
    asm volatile(
        "call"
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31),"
        "_optix_trace_typed_32,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(p5), "=r"(p6), "=r"(p7), "=r"(p8),
          "=r"(p9), "=r"(p10), "=r"(p11), "=r"(p12), "=r"(p13), "=r"(p14), "=r"(p15), "=r"(p16),
          "=r"(p17), "=r"(p18), "=r"(p19), "=r"(p20), "=r"(p21), "=r"(p22), "=r"(p23), "=r"(p24),
          "=r"(p25), "=r"(p26), "=r"(p27), "=r"(p28), "=r"(p29), "=r"(p30), "=r"(p31)
        : "r"(payload_type), "l"(accel.handle), "f"(ox), "f"(oy), "f"(oz), "f"(dx), "f"(dy), "f"(dz), "f"(t_min),
          "f"(t_max), "f"(0.0f), "r"(mask & 0xffu), "r"(flags), "r"(0u), "r"(0u),
          "r"(ray_type), "r"(reg_count), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u)
        :);
}

enum LCRayFlags : unsigned int {
    LC_RAY_FLAG_NONE = 0u,
    LC_RAY_FLAG_DISABLE_ANYHIT = 1u << 0u,
    LC_RAY_FLAG_ENFORCE_ANYHIT = 1u << 1u,
    LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1u << 2u,
    LC_RAY_FLAG_DISABLE_CLOSESTHIT = 1u << 3u,
    LC_RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 1u << 4u,
    LC_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 1u << 5u,
    LC_RAY_FLAG_CULL_DISABLED_ANYHIT = 1u << 6u,
    LC_RAY_FLAG_CULL_ENFORCED_ANYHIT = 1u << 7u,
};

[[nodiscard]] inline auto lc_accel_trace_closest(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT;
    auto r0 = 0u;
    auto r1 = 0u;
    auto r2 = 0u;
    auto r3 = 0u;
    auto r4 = 0u;
    lc_trace_impl<0u, 5u, flags>(LC_PAYLOAD_TYPE_ID_0, accel, ray, mask, r0, r1, r2, r3, r4);
    return LCTriangleHit{r0, r1, lc_make_float2(__uint_as_float(r2), __uint_as_float(r3)), __uint_as_float(r4)};
}

[[nodiscard]] inline auto lc_accel_trace_any(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT |
                           LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                           LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    auto r0 = 0u;
    auto r1 = 0u;
    auto r2 = 0u;
    auto r3 = 0u;
    auto r4 = 0u;
    lc_trace_impl<1u, 1u, flags>(LC_PAYLOAD_TYPE_ID_1, accel, ray, mask, r0, r1, r2, r3, r4);
    return r0 != ~0u;
}

[[nodiscard]] inline auto lc_dispatch_id() noexcept {
    lc_uint u0, u1, u2;
    asm("call (%0), _optix_get_launch_index_x, ();"
        : "=r"(u0)
        :);
    asm("call (%0), _optix_get_launch_index_y, ();"
        : "=r"(u1)
        :);
    asm("call (%0), _optix_get_launch_index_z, ();"
        : "=r"(u2)
        :);
    return lc_make_uint3(u0, u1, u2);
}

[[nodiscard]] inline auto lc_dispatch_size() noexcept {
    lc_uint u0, u1, u2;
    asm("call (%0), _optix_get_launch_dimension_x, ();"
        : "=r"(u0)
        :);
    asm("call (%0), _optix_get_launch_dimension_y, ();"
        : "=r"(u1)
        :);
    asm("call (%0), _optix_get_launch_dimension_z, ();"
        : "=r"(u2)
        :);
    return lc_make_uint3(u0, u1, u2);
}

[[nodiscard]] inline auto lc_thread_id() noexcept {
    return lc_dispatch_id() % lc_block_size();
}

[[nodiscard]] inline auto lc_block_id() noexcept {
    return lc_dispatch_id() / lc_block_size();
}

#else
#define lc_dispatch_size() dispatch_size

[[nodiscard]] __device__ inline auto lc_thread_id() noexcept {
    return lc_make_uint3(lc_uint(threadIdx.x),
                         lc_uint(threadIdx.y),
                         lc_uint(threadIdx.z));
}

[[nodiscard]] __device__ inline auto lc_block_id() noexcept {
    return lc_make_uint3(lc_uint(blockIdx.x),
                         lc_uint(blockIdx.y),
                         lc_uint(blockIdx.z));
}

[[nodiscard]] __device__ inline auto lc_dispatch_id() noexcept {
    return lc_block_id() * lc_block_size() + lc_thread_id();
}

__device__ inline void lc_synchronize_block() noexcept {
    __syncthreads();
}

#endif
