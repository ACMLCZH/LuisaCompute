/**
 * @file src/tests/next/tset/ext/dx/test_dml.cpp
 * @author sailing-innocent, on MuGdxy's previous work on 7/26/2023.
 * @date 2023/11/03
 * @brief the directML test suite
*/

#include "common/config.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <luisa/backends/ext/cuda/lcub/device_scan.h>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>
#include <numeric>
#include <algorithm>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::cuda::lcub;

namespace luisa::test {

int test_lcub_exclusive_scan(Device &device, int num_item, std::vector<int> &input, std::vector<int> &gt) {
    auto stream = device.create_stream();
    auto d_in = device.create_buffer<int>(input.size());
    stream << d_in.copy_from(input.data());
    auto d_out = device.create_buffer<int>(input.size());
    std::vector<int> out(num_item);

    Buffer<int> temp_storage;
    size_t temp_storage_size = -1;
    DeviceScan::ExclusiveSum(temp_storage_size, d_in, d_out, input.size());
    temp_storage = device.create_buffer<int>(temp_storage_size);
    stream << DeviceScan::ExclusiveSum(temp_storage, d_in, d_out, input.size());

    stream << d_out.copy_to(out.data()) << synchronize();
    // check
    for (int i = 0; i < num_item; i++) {
        CHECK_MESSAGE(gt[i] == out[i], "Error: " << i << " " << gt[i] << " " << out[i]);
    }
    return 0;
}

int test_lcub_radix_sort(Device &device, int num_item, std::vector<int> &input, std::vector<int> &gt) {
    auto stream = device.create_stream();
    auto d_in = device.create_buffer<int>(input.size());
    stream << d_in.copy_from(input.data());
    auto d_out = device.create_buffer<int>(input.size());
    std::vector<int> out(num_item);
    std::vector<int> val(num_item);
    std::iota(val.begin(), val.end(), 0);
    Buffer<int> d_val_in = device.create_buffer<int>(val.size());
    stream << d_val_in.copy_from(val.data());
    Buffer<int> d_val_out = device.create_buffer<int>(val.size());
    stream << synchronize();

    Buffer<int> temp_storage;
    size_t temp_storage_size = -1;
    DeviceRadixSort::SortPairs(temp_storage_size, d_in, d_out, d_val_in, d_val_out, num_item);
    temp_storage = device.create_buffer<int>(temp_storage_size);
    stream << DeviceRadixSort::SortPairs(temp_storage, d_in, d_out, d_val_in, d_val_out, num_item);
    stream << d_val_out.copy_to(val.data());
    stream << d_out.copy_to(out.data()) << synchronize();
    // check
    for (int i = 0; i < num_item; i++) {
        CHECK(val[i] == num_item - i - 1);
        CHECK_MESSAGE(gt[i] == out[i], "Error: " << i << " " << gt[i] << " " << out[i]);
    }
    return 0;
}

}// namespace luisa::test

TEST_SUITE("ext_cuda") {
    TEST_CASE("cuda_lcub") {
        Context context{luisa::test::argv()[0]};
        Device device = context.create_device("cuda");

        constexpr auto num_item = 100;
        std::vector<int> input(num_item);
        std::iota(input.begin(), input.end(), 0);
        std::vector<int> gt(num_item);

        SUBCASE("exclusive_scan") {
            std::exclusive_scan(input.begin(), input.end(), gt.begin(), 0);
            REQUIRE(luisa::test::test_lcub_exclusive_scan(device, num_item, input, gt) == 0);
        }

        SUBCASE("radix_sort") {
            for (auto i = 0; i < num_item; i++) {
                input[i] = num_item - i - 1;
                gt[i] = i;
            }
            REQUIRE(luisa::test::test_lcub_radix_sort(device, num_item, input, gt) == 0);
        }
    }
}
