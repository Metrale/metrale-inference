// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-21: The probe kernels of pdl_capture_gate. pdl_probe_ptx.rs holds this file's PTX:
//   nvcc -ptx -arch=sm_90 -o pdl_probe.ptx pdl_probe.cu
// producer: waits on its own predecessor, lets its dependents launch, spins spin_ns, then writes
// data and the time it finished. consumer: records when it started (before its wait), waits,
// copies data. plain: the consumer without the wait (never launched programmatically).
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
extern "C" __global__ void pdl_probe_producer(unsigned int* data, unsigned long long* t_end, unsigned int value, unsigned int spin_ns) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    const unsigned long long t0 = gt();
    while (gt() - t0 < spin_ns) { }
    data[threadIdx.x] = value + threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) t_end[0] = gt();
}
extern "C" __global__ void pdl_probe_consumer(const unsigned int* data, unsigned int* out, unsigned long long* t_start) {
    if (threadIdx.x == 0) t_start[0] = gt();
    asm volatile("griddepcontrol.wait;" ::: "memory");
    out[threadIdx.x] = data[threadIdx.x];
}
extern "C" __global__ void pdl_probe_plain(const unsigned int* data, unsigned int* out, unsigned long long* t_start) {
    if (threadIdx.x == 0) t_start[0] = gt();
    out[threadIdx.x] = data[threadIdx.x];
}
