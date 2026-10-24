2026-10-24: Fixtures for fatbin_tests.rs, made with nvcc 13.0 (V13.0.88) from pdl_global.cu:
  nvcc --fatbin --no-compress -gencode 'arch=compute_121f,code=[sm_121f,compute_121f]' -O3 pdl_global.cu -o pdl_global.fatbin
  nvcc --fatbin               -gencode 'arch=compute_121f,code=[sm_121f,compute_121f]' -O3 pdl_global.cu -o pdl_global_zstd.fatbin
  nvcc --ptx -arch=sm_121f -O3 pdl_global.cu -o pdl_global.ptx
The first is what the kernel build emits (crates/kernels/build_image.rs); the second is nvcc's
default (PTX zstd-compressed), which the registry must refuse; the third is the same module as
PTX text, which the fatbin's PTX must equal up to comments and whitespace.
