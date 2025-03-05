#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "consts.h"

#include "../../../ynotif/ynotif.h"
#include "../../../ydata/ydata.h"

void fill_f64(f64 *restrict a, f64 c, u64 n)
{
  for (u64 i = 0; i < n; i++)
    a[i] = c;
}

f64 reduc_C(f64 *restrict a, u64 n)
{
  f64 r = 0.0;
  
  for (u64 i = 0; i < n; i++)
    r += a[i];

  return r;
}

f64 reduc_SSE_scalar(f64 *restrict a, u64 n)
{
  f64 r = 0.0;
  u64 s = sizeof(f64) * n;
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "xorpd %%xmm0, %%xmm0;\n"
		    
		    "1:;\n"
		    
		    "addsd (%[_a], %%rcx), %%xmm0;\n"
		    
		    "add $8, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "movsd %%xmm0, (%[_r]);\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (&r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0"
		    );
  
  return r;
}

f64 reduc_SSE_vector_x1(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 1);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) r[2] = { 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "xorpd %%xmm0, %%xmm0;\n"
		    
		    "1:;\n"
		    
		    "addpd (%[_a], %%rcx), %%xmm0;\n"
		    
		    "add $16, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    "movapd %%xmm0, (%[_r]);"

		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1];
}

f64 reduc_SSE_vector_x2(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 3);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) r[2] = { 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "xorpd %%xmm0, %%xmm0;\n"
		    "xorpd %%xmm1, %%xmm1;\n"
		    
		    "1:;\n"
		    
		    "addpd   (%[_a], %%rcx), %%xmm0;\n"
		    "addpd 16(%[_a], %%rcx), %%xmm1;\n"
		    
		    "add $32, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    "addpd %%xmm1, %%xmm0;\n"
		    
		    "movapd %%xmm0, (%[_r]);\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1"
		    );

  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1];
}

f64 reduc_SSE_vector_x4(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 7);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) r[2] = { 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "xorpd %%xmm0, %%xmm0;\n"
		    "xorpd %%xmm1, %%xmm1;\n"
		    "xorpd %%xmm2, %%xmm2;\n"
		    "xorpd %%xmm3, %%xmm3;\n"
		    
		    "1:;\n"
		    
		    "addpd   (%[_a], %%rcx), %%xmm0;\n"
		    "addpd 16(%[_a], %%rcx), %%xmm1;\n"
		    "addpd 32(%[_a], %%rcx), %%xmm2;\n"
		    "addpd 48(%[_a], %%rcx), %%xmm3;\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    "addpd %%xmm1, %%xmm0;\n"
		    "addpd %%xmm3, %%xmm2;\n"
		    "addpd %%xmm2, %%xmm0;\n"
		    
		    "movapd %%xmm0, (%[_r]);\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3"
		    );

  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1];
}

f64 reduc_SSE_vector_x8(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 15);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) r[2] = { 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "xorpd %%xmm0, %%xmm0;\n"
		    "xorpd %%xmm1, %%xmm1;\n"
		    "xorpd %%xmm2, %%xmm2;\n"
		    "xorpd %%xmm3, %%xmm3;\n"
		    "xorpd %%xmm4, %%xmm4;\n"
		    "xorpd %%xmm5, %%xmm5;\n"
		    "xorpd %%xmm6, %%xmm6;\n"
		    "xorpd %%xmm7, %%xmm7;\n"
		    
		    "1:;\n"
		    
		    "addpd    (%[_a], %%rcx), %%xmm0;\n"
		    "addpd  16(%[_a], %%rcx), %%xmm1;\n"
		    "addpd  32(%[_a], %%rcx), %%xmm2;\n"
		    "addpd  48(%[_a], %%rcx), %%xmm3;\n"
		    "addpd  64(%[_a], %%rcx), %%xmm4;\n"
		    "addpd  80(%[_a], %%rcx), %%xmm5;\n"
		    "addpd  96(%[_a], %%rcx), %%xmm6;\n"
		    "addpd 112(%[_a], %%rcx), %%xmm7;\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    "addpd %%xmm1, %%xmm0;\n"
		    "addpd %%xmm3, %%xmm2;\n"
		    "addpd %%xmm5, %%xmm4;\n"
		    "addpd %%xmm7, %%xmm6;\n"
		    
		    "addpd %%xmm2, %%xmm0;\n"
		    "addpd %%xmm6, %%xmm4;\n"
		    
		    "addpd %%xmm4, %%xmm0;\n"

		    "movapd %%xmm0, (%[_r]);\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1];
}

f64 reduc_AVX_vector_x1(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 3);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorpd %%ymm0, %%ymm0, %%ymm0;\n"
		    
		    "1:;\n"
		    
		    "vaddpd (%[_a], %%rcx), %%ymm0, %%ymm0;\n"
		    
		    "add $32, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);"

		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

f64 reduc_AVX_vector_x2(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 7);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorpd %%ymm0, %%ymm0, %%ymm0;\n"
		    "vxorpd %%ymm1, %%ymm1, %%ymm1;\n"
		    
		    "1:;\n"
		    
		    "vaddpd   (%[_a], %%rcx), %%ymm0, %%ymm0;\n"
		    "vaddpd 32(%[_a], %%rcx), %%ymm1, %%ymm1;\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "vaddpd %%ymm1, %%ymm0, %%ymm0;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);"

		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

f64 reduc_AVX_vector_x4(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 15);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorpd %%ymm0, %%ymm0, %%ymm0;\n"
		    "vxorpd %%ymm1, %%ymm1, %%ymm1;\n"
		    "vxorpd %%ymm2, %%ymm2, %%ymm2;\n"
		    "vxorpd %%ymm3, %%ymm3, %%ymm3;\n"
		    
		    "1:;\n"
		    
		    "vaddpd   (%[_a], %%rcx), %%ymm0, %%ymm0;\n"
		    "vaddpd 32(%[_a], %%rcx), %%ymm1, %%ymm1;\n"
		    "vaddpd 64(%[_a], %%rcx), %%ymm2, %%ymm2;\n"
		    "vaddpd 96(%[_a], %%rcx), %%ymm3, %%ymm3;\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "vaddpd %%ymm1, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm3, %%ymm2, %%ymm2;\n"

		    "vaddpd %%ymm2, %%ymm0, %%ymm0;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);"

		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

f64 reduc_AVX_vector_x8(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 31);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorpd %%ymm0, %%ymm0, %%ymm0;\n"
		    "vxorpd %%ymm1, %%ymm1, %%ymm1;\n"
		    "vxorpd %%ymm2, %%ymm2, %%ymm2;\n"
		    "vxorpd %%ymm3, %%ymm3, %%ymm3;\n"
		    "vxorpd %%ymm4, %%ymm4, %%ymm4;\n"
		    "vxorpd %%ymm5, %%ymm5, %%ymm5;\n"
		    "vxorpd %%ymm6, %%ymm6, %%ymm6;\n"
		    "vxorpd %%ymm7, %%ymm7, %%ymm7;\n"

		    "1:;\n"
		    
		    "vaddpd    (%[_a], %%rcx), %%ymm0, %%ymm0;\n"
		    "vaddpd  32(%[_a], %%rcx), %%ymm1, %%ymm1;\n"
		    "vaddpd  64(%[_a], %%rcx), %%ymm2, %%ymm2;\n"
		    "vaddpd  96(%[_a], %%rcx), %%ymm3, %%ymm3;\n"
		    "vaddpd 128(%[_a], %%rcx), %%ymm4, %%ymm4;\n"
		    "vaddpd 160(%[_a], %%rcx), %%ymm5, %%ymm5;\n"
		    "vaddpd 192(%[_a], %%rcx), %%ymm6, %%ymm6;\n"
		    "vaddpd 224(%[_a], %%rcx), %%ymm7, %%ymm7;\n"
		    
		    "add $256, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "vaddpd %%ymm1, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm3, %%ymm2, %%ymm2;\n"
		    "vaddpd %%ymm5, %%ymm4, %%ymm4;\n"
		    "vaddpd %%ymm7, %%ymm6, %%ymm6;\n"

		    "vaddpd %%ymm2, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm6, %%ymm4, %%ymm4;\n"
		    
		    "vaddpd %%ymm4, %%ymm0, %%ymm0;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

f64 reduc_AVX_vector_x16(f64 *restrict a, u64 n)
{
  u64 nn = n - (n & 63);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorpd %%ymm0, %%ymm0, %%ymm0;\n"
		    "vxorpd %%ymm1, %%ymm1, %%ymm1;\n"
		    "vxorpd %%ymm2, %%ymm2, %%ymm2;\n"
		    "vxorpd %%ymm3, %%ymm3, %%ymm3;\n"
		    "vxorpd %%ymm4, %%ymm4, %%ymm4;\n"
		    "vxorpd %%ymm5, %%ymm5, %%ymm5;\n"
		    "vxorpd %%ymm6, %%ymm6, %%ymm6;\n"
		    "vxorpd %%ymm7, %%ymm7, %%ymm7;\n"
		    "vxorpd %%ymm8, %%ymm8, %%ymm8;\n"
		    "vxorpd %%ymm9, %%ymm9, %%ymm9;\n"
		    "vxorpd %%ymm10, %%ymm10, %%ymm10;\n"
		    "vxorpd %%ymm11, %%ymm11, %%ymm11;\n"
		    "vxorpd %%ymm12, %%ymm12, %%ymm12;\n"
		    "vxorpd %%ymm13, %%ymm13, %%ymm13;\n"
		    "vxorpd %%ymm14, %%ymm14, %%ymm14;\n"
		    "vxorpd %%ymm15, %%ymm15, %%ymm15;\n"

		    "1:;\n"
		    
		    "vaddpd    (%[_a], %%rcx), %%ymm0, %%ymm0;\n"
		    "vaddpd  32(%[_a], %%rcx), %%ymm1, %%ymm1;\n"
		    "vaddpd  64(%[_a], %%rcx), %%ymm2, %%ymm2;\n"
		    "vaddpd  96(%[_a], %%rcx), %%ymm3, %%ymm3;\n"
		    "vaddpd 128(%[_a], %%rcx), %%ymm4, %%ymm4;\n"
		    "vaddpd 160(%[_a], %%rcx), %%ymm5, %%ymm5;\n"
		    "vaddpd 192(%[_a], %%rcx), %%ymm6, %%ymm6;\n"
		    "vaddpd 224(%[_a], %%rcx), %%ymm7, %%ymm7;\n"
		    "vaddpd 256(%[_a], %%rcx), %%ymm8, %%ymm8;\n"
		    "vaddpd 288(%[_a], %%rcx), %%ymm9, %%ymm9;\n"
		    "vaddpd 320(%[_a], %%rcx), %%ymm10, %%ymm10;\n"
		    "vaddpd 352(%[_a], %%rcx), %%ymm11, %%ymm11;\n"
		    "vaddpd 384(%[_a], %%rcx), %%ymm12, %%ymm12;\n"
		    "vaddpd 416(%[_a], %%rcx), %%ymm13, %%ymm13;\n"
		    "vaddpd 448(%[_a], %%rcx), %%ymm14, %%ymm14;\n"
		    "vaddpd 480(%[_a], %%rcx), %%ymm15, %%ymm15;\n"
		    
		    "add $512, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "vaddpd %%ymm1, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm3, %%ymm2, %%ymm2;\n"
		    "vaddpd %%ymm5, %%ymm4, %%ymm4;\n"
		    "vaddpd %%ymm7, %%ymm6, %%ymm6;\n"
		    "vaddpd %%ymm9, %%ymm8, %%ymm8;\n"
		    "vaddpd %%ymm11, %%ymm10, %%ymm10;\n"
		    "vaddpd %%ymm13, %%ymm12, %%ymm12;\n"
		    "vaddpd %%ymm15, %%ymm14, %%ymm14;\n"
		    
		    "vaddpd %%ymm2, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm6, %%ymm4, %%ymm4;\n"
		    "vaddpd %%ymm10, %%ymm8, %%ymm8;\n"
		    "vaddpd %%ymm14, %%ymm12, %%ymm12;\n"

		    "vaddpd %%ymm4, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm12, %%ymm8, %%ymm8;\n"
		    
		    "vaddpd %%ymm8, %%ymm0, %%ymm0;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7",
		    "ymm8", "ymm9", "ymm10", "ymm11",
		    "ymm12", "ymm13", "ymm14", "ymm15"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

void benchmark_reduc(const ascii *title, f64 (*kernel)(f64 *restrict, u64), u64 n, u64 r)
{
  f64 v = 0.0;
  f64 e = 0.0;
  struct timespec t0, t1;
  u64 s = n * sizeof(f64);
  ydata_t *d = ydata_create(title, YBW_MAX_SAMPLES);
  
  for (u64 i = 0; i < YBW_MAX_SAMPLES; i++)
    {
      f64 *restrict a = aligned_alloc(64, sizeof(f64) * s);
      
      fill_f64(a, 1.0, n);
      
      do
	{
	  clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

	  for (u64 j = 0; j < r; j++)
	    v = kernel(a, n);
	  
	  clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

	  e = (f64)(t1.tv_nsec - t0.tv_nsec) / (f64)r;
	}
      while (e <= 0.0);

      d->datapoints[i] = e;
      
      free(a);
    }
  
  ydata_analyze(d);

  //Arithmetic intensity in GFLOP/s for 1 FP operation (addition)
  f64 ai = (f64)n / d->min;
  
  //Memory bandwidth in GiB/s for 1 array (load)
  f64 bw = ((f64)s / (1024.0 * 1024.0 * 1024.0)) / (d->min / 1e9);

  printf("%25s; %13.3lf; %13.3lf; %13.3lf; %13.3lf; %13.3lf; %13.3lf; %18.3lf; %8.3lf; %13.3lf; %13.3lf\n",
	 d->title,
	 (f64)s / (1024.0 * 1024.0),
	 v,
	 d->min,
	 d->max,
	 d->mean,
	 d->median,
	 d->min_relative_range,
	 d->coefficient_of_variation,
	 bw,
	 ai);
  
  ydata_destroy(&d);
}

i32 main(i32 argc, ascii **argv)
{
  if (argc < 3)
    return printf("Usage: %s [n] [r]\n", argv[0]), 1;

  u64 n = (u64)atoll(argv[1]);
  u64 r = (u64)atoll(argv[2]);
  
  printf("%25s; %13s; %13s; %13s; %13s; %13s; %13s; %18s; %8s; %13s; %13s\n",
	 "title",
	 "MiB",
	 "value",
	 "min",
	 "max",
	 "mean",
	 "median",
	 "(max-min)/min",
	 "CV",
	 "GiB/s",
	 "GFLOP/s");
  
  benchmark_reduc("reduc C"             , reduc_C             , n, r);
  benchmark_reduc("reduc SSE scalar"    , reduc_SSE_scalar    , n, r);
  benchmark_reduc("reduc SSE vector x1" , reduc_SSE_vector_x1 , n, r);
  benchmark_reduc("reduc SSE vector x2" , reduc_SSE_vector_x2 , n, r);
  benchmark_reduc("reduc SSE vector x4" , reduc_SSE_vector_x4 , n, r);
  benchmark_reduc("reduc SSE vector x8" , reduc_SSE_vector_x8 , n, r);
  benchmark_reduc("reduc AVX vector x1" , reduc_AVX_vector_x1 , n, r);
  benchmark_reduc("reduc AVX vector x2" , reduc_AVX_vector_x2 , n, r);
  benchmark_reduc("reduc AVX vector x4" , reduc_AVX_vector_x4 , n, r);
  benchmark_reduc("reduc AVX vector x8" , reduc_AVX_vector_x8 , n, r);
  benchmark_reduc("reduc AVX vector x16", reduc_AVX_vector_x16, n, r);
  
  return 0;
}
