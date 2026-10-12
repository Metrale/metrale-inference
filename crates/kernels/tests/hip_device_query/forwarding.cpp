// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Independently authored host witnesses; not a HIP ABI emulator.
#include <cstdio>
#include <cstring>
#include <initializer_list>

using hipDevice_t = int;
using hipError_t = int;
[[maybe_unused]] constexpr int hipSuccess = 0, hipErrorInvalidValue = 1;
// 2026-10-07: Deliberately unlike CUDA numbers: accidental enum casts must fail.
enum hipDeviceAttribute_t {
    hipDeviceAttributeMaxThreadsPerBlock = 301,
    hipDeviceAttributeMaxSharedMemoryPerBlock = 302,
    hipDeviceAttributeWarpSize = 303,
    hipDeviceAttributeMultiprocessorCount = 304,
    hipDeviceAttributeIntegrated = 305,
    hipDeviceAttributeCanMapHostMemory = 306,
    hipDeviceAttributeMemoryClockRate = 307,
    hipDeviceAttributeUnifiedAddressing = 308
};
static int status = 0, calls = 0, last_device = -1, last_attribute = -1;
static int hipDeviceGet(hipDevice_t* d, int ordinal) {
    ++calls; last_device = ordinal;
    if (status) return status;
    if (ordinal < 0) return 101;
    *d = ordinal; return 0;
}
static int hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attr, int device) {
    ++calls; last_device = device; last_attribute = attr;
    if (status) return status;
    if (!value) return 1;
    if (device < 0) return 101;
    *value = 1000 * device + attr; return 0;
}
static int hipDeviceGetName(char* name, int len, hipDevice_t device) {
    ++calls; last_device = device;
    if (status) return status;
    if (!name || len <= 0) return 1;
    if (device < 0) return 101;
    std::snprintf(name, len, "physical-device-%d", device); return 0;
}
// 2026-10-07: PRODUCTION_DEFINITIONS
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    const int cuda[] = {1, 8, 10, 16, 18, 19, 36, 41};
    const int hip[] = {301, 302, 303, 304, 305, 306, 307, 308};
    for (int device : {2, 7}) {
        for (int i = 0; i < 8; ++i) {
            int out = -77; calls = 0;
            REQUIRE(cuDeviceGetAttribute(&out, cuda[i], device) == 0);
            REQUIRE(calls == 1 && last_device == device && last_attribute == hip[i]);
            REQUIRE(out == 1000 * device + hip[i]);
            for (int error : {3, 101, 801}) {
                status = error; out = -77;
                REQUIRE(cuDeviceGetAttribute(&out, cuda[i], device) == error);
                REQUIRE(out == -77); status = 0;
            }
            out = -77;
            REQUIRE(cuDeviceGetAttribute(&out, cuda[i], -1) == 101 && out == -77);
            REQUIRE(cuDeviceGetAttribute(nullptr, cuda[i], device) == 1);
        }
        for (int attr : {75, 76, 115}) {
            int out = -77; calls = 0;
            REQUIRE(cuDeviceGetAttribute(&out, attr, device) == 0);
            REQUIRE(out == (attr == 75 ? 12 : attr == 76 ? 1 : 0));
            REQUIRE(calls == 1 && last_device == device);
            status = 3; out = -77;
            REQUIRE(cuDeviceGetAttribute(&out, attr, device) == 3 && out == -77);
            status = 0;
            REQUIRE(cuDeviceGetAttribute(&out, attr, -1) == 101 && out == -77);
            REQUIRE(cuDeviceGetAttribute(nullptr, attr, device) == 1);
        }
        char name[64]; calls = 0;
        REQUIRE(cuDeviceGetName(name, sizeof(name), device) == 0);
        char expected[64]; std::snprintf(expected, sizeof(expected), "physical-device-%d", device);
        REQUIRE(std::strcmp(name, expected) == 0 && calls == 1 && last_device == device);
        char tiny[4] = {'x','x','x','x'};
        REQUIRE(cuDeviceGetName(tiny, sizeof(tiny), device) == 0);
        REQUIRE(std::strcmp(tiny, "phy") == 0);
        REQUIRE(cuDeviceGetName(nullptr, 4, device) == 1);
        REQUIRE(cuDeviceGetName(name, 0, device) == 1);
        std::strcpy(name, "sentinel");
        REQUIRE(cuDeviceGetName(name, sizeof(name), -1) == 101);
        REQUIRE(std::strcmp(name, "sentinel") == 0);
        for (int error : {3, 101, 801}) {
            status = error; std::strcpy(name, "sentinel");
            REQUIRE(cuDeviceGetName(name, sizeof(name), device) == error);
            REQUIRE(std::strcmp(name, "sentinel") == 0); status = 0;
        }
    }
    for (int attr : {-1, 0, 99999}) {
        int out = -77; calls = 0;
        REQUIRE(cuDeviceGetAttribute(&out, attr, 2) == 1 && out == -77 && calls == 0);
    }
    std::puts("PASS production device-query forwarding, refusal and synthetic policy");
}
