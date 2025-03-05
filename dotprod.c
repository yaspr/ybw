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

f64 dotprod_C(f64 *restrict a, f64 *restrict b, u64 n)
{
  f64 r = 0.0;
  
  for (u64 i = 0; i < n; i++)
    r += a[i] * b[i];

  return r;
}

f64 dotprod_SSE_scalar(f64 *restrict a, f64 *restrict b, u64 n)
{
  f64 r = 0.0;
  u64 s = sizeof(f64) * n;
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "xorps %%xmm0, %%xmm0;\n"
		    
		    "1:;\n"

		    "movsd (%[_b], %%rcx), %%xmm1;\n"
		    "mulsd (%[_a], %%rcx), %%xmm1;\n"
		    "addsd %%xmm1, %%xmm0;\n"
		    
		    "add $8, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "movsd %%xmm0, (%[_r]);\n"
		    :
		    
		    :
		    [_a] "r" (a),
		    [_b] "r" (b),
		    [_r] "r" (&r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0"
		    );

  return r;
}

f64 dotprod_SSE_vector_x1(f64 *restrict a, f64 *restrict b, u64 n)
{
  u64 nn = n - (n & 1);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) r[2] = { 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "xorps %%xmm0, %%xmm0;\n"
		    
		    "1:;\n"
		    
		    "movapd (%[_b], %%rcx), %%xmm1;\n"
		    "mulpd (%[_a], %%rcx), %%xmm1;\n"
		    "addpd %%xmm1, %%xmm0;\n"
		    
		    "add $16, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "movapd %%xmm0, (%[_r]);\n"
		    :
		    
		    :
		    [_a] "r" (a),
		    [_b] "r" (b),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i] * b[i];
  
  return r[0] + r[1];
}

f64 dotprod_SSE_vector_x2(f64 *restrict a, f64 *restrict b, u64 n)
{
  u64 nn = n - (n & 3);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) r[2] = { 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "xorps %%xmm0, %%xmm0;\n"
		    "xorps %%xmm2, %%xmm2;\n"
		    
		    "1:;\n"
		    
		    "movapd   (%[_b], %%rcx), %%xmm1;\n"
		    "movapd 16(%[_b], %%rcx), %%xmm3;\n"
		    
		    "mulpd   (%[_a], %%rcx), %%xmm1;\n"
		    "mulpd 16(%[_a], %%rcx), %%xmm3;\n"
		    
		    "addpd %%xmm1, %%xmm0;\n"
		    "addpd %%xmm3, %%xmm2;\n"
		    
		    "add $32, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "addpd %%xmm2, %%xmm0;\n"
		    
		    "movapd %%xmm0, (%[_r]);\n"
		    :
		    
		    :
		    [_a] "r" (a),
		    [_b] "r" (b),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i] * b[i];
  
  return r[0] + r[1];
}

f64 dotprod_SSE_vector_x4(f64 *restrict a, f64 *restrict b, u64 n)
{
  u64 nn = n - (n & 3);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) r[2] = { 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "xorps %%xmm0, %%xmm0;\n"
		    "xorps %%xmm2, %%xmm2;\n"
		    "xorps %%xmm4, %%xmm4;\n"
		    "xorps %%xmm6, %%xmm6;\n"
		    
		    "1:;\n"
		    
		    "movapd   (%[_b], %%rcx), %%xmm1;\n"
		    "movapd 16(%[_b], %%rcx), %%xmm3;\n"
		    "movapd 32(%[_b], %%rcx), %%xmm5;\n"
		    "movapd 48(%[_b], %%rcx), %%xmm7;\n"
		    
		    "mulpd   (%[_a], %%rcx), %%xmm1;\n"
		    "mulpd 16(%[_a], %%rcx), %%xmm3;\n"
		    "mulpd 32(%[_a], %%rcx), %%xmm5;\n"
		    "mulpd 48(%[_a], %%rcx), %%xmm7;\n"
		    
		    "addpd %%xmm1, %%xmm0;\n"
		    "addpd %%xmm3, %%xmm2;\n"
		    "addpd %%xmm5, %%xmm4;\n"
		    "addpd %%xmm7, %%xmm6;\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "addpd %%xmm2, %%xmm0;\n"
		    "addpd %%xmm6, %%xmm4;\n"
		    
		    "addpd %%xmm4, %%xmm0;\n"
		    
		    "movapd %%xmm0, (%[_r]);\n"
		    :
		    
		    :
		    [_a] "r" (a),
		    [_b] "r" (b),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3",
		    "xmm4", "xmm5", "xmm6", "xmm7"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i] * b[i];
  
  return r[0] + r[1];
}

f64 dotprod_AVX_vector_x1(f64 *restrict a, f64 *restrict b, u64 n)
{
  u64 nn = n - (n & 3);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorps %%ymm0, %%ymm0, %%ymm0;\n"
		    
		    "1:;\n"
		    
		    "vmovapd (%[_b], %%rcx), %%ymm1;\n"
		    "vfmadd231pd (%[_a], %%rcx), %%ymm1, %%ymm0;\n"
		    
		    "add $32, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);\n"
		    :
		    
		    :
		    [_a] "r" (a),
		    [_b] "r" (b),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i] * b[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

f64 dotprod_AVX_vector_x2(f64 *restrict a, f64 *restrict b, u64 n)
{
  u64 nn = n - (n & 7);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorps %%ymm0, %%ymm0, %%ymm0;\n"
		    "vxorps %%ymm2, %%ymm2, %%ymm2;\n"
		    
		    "1:;\n"
		    
		    "vmovapd   (%[_b], %%rcx), %%ymm1;\n"
		    "vmovapd 32(%[_b], %%rcx), %%ymm3;\n"
		    
		    "vfmadd231pd   (%[_a], %%rcx), %%ymm1, %%ymm0;\n"
		    "vfmadd231pd 32(%[_a], %%rcx), %%ymm3, %%ymm2;\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    "vaddpd %%ymm2, %%ymm0, %%ymm0;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);\n"
		    :
		    
		    :
		    [_a] "r" (a),
		    [_b] "r" (b),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i] * b[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

f64 dotprod_AVX_vector_x4(f64 *restrict a, f64 *restrict b, u64 n)
{
  u64 nn = n - (n & 15);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorps %%ymm0, %%ymm0, %%ymm0;\n"
		    "vxorps %%ymm2, %%ymm2, %%ymm2;\n"
		    "vxorps %%ymm4, %%ymm4, %%ymm4;\n"
		    "vxorps %%ymm6, %%ymm6, %%ymm6;\n"
		    
		    "1:;\n"
		    
		    "vmovapd   (%[_b], %%rcx), %%ymm1;\n"
		    "vmovapd 32(%[_b], %%rcx), %%ymm3;\n"
		    "vmovapd 64(%[_b], %%rcx), %%ymm5;\n"
		    "vmovapd 96(%[_b], %%rcx), %%ymm7;\n"
		    
		    "vfmadd231pd   (%[_a], %%rcx), %%ymm1, %%ymm0;\n"
		    "vfmadd231pd 32(%[_a], %%rcx), %%ymm3, %%ymm2;\n"
		    "vfmadd231pd 64(%[_a], %%rcx), %%ymm5, %%ymm4;\n"
		    "vfmadd231pd 96(%[_a], %%rcx), %%ymm7, %%ymm6;\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "vaddpd %%ymm2, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm6, %%ymm4, %%ymm4;\n"

		    "vaddpd %%ymm4, %%ymm0, %%ymm0;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);\n"
		    :
		    
		    :
		    [_a] "r" (a),
		    [_b] "r" (b),
		    [_r] "r" (r),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7"
		    );
  
  for (u64 i = nn; i < n; i++)
    r[0] += a[i] * b[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

f64 dotprod_AVX_vector_x8(f64 *restrict a, f64 *restrict b, u64 n)
{
  u64 nn = n - (n & 31);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) r[4] = { 0.0, 0.0, 0.0, 0.0 };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "vxorps %%ymm0, %%ymm0, %%ymm0;\n"
		    "vxorps %%ymm2, %%ymm2, %%ymm2;\n"
		    "vxorps %%ymm4, %%ymm4, %%ymm4;\n"
		    "vxorps %%ymm6, %%ymm6, %%ymm6;\n"
		    "vxorps %%ymm8, %%ymm8, %%ymm8;\n"
		    "vxorps %%ymm10, %%ymm10, %%ymm10;\n"
		    "vxorps %%ymm12, %%ymm12, %%ymm12;\n"
		    "vxorps %%ymm14, %%ymm14, %%ymm14;\n"
		    
		    "1:;\n"
		    
		    "vmovapd    (%[_b], %%rcx), %%ymm1;\n"
		    "vmovapd  32(%[_b], %%rcx), %%ymm3;\n"
		    "vmovapd  64(%[_b], %%rcx), %%ymm5;\n"
		    "vmovapd  96(%[_b], %%rcx), %%ymm7;\n"
		    "vmovapd 128(%[_b], %%rcx), %%ymm9;\n"
		    "vmovapd 160(%[_b], %%rcx), %%ymm11;\n"
		    "vmovapd 192(%[_b], %%rcx), %%ymm13;\n"
		    "vmovapd 224(%[_b], %%rcx), %%ymm15;\n"
		    
		    "vfmadd231pd    (%[_a], %%rcx), %%ymm1, %%ymm0;\n"
		    "vfmadd231pd  32(%[_a], %%rcx), %%ymm3, %%ymm2;\n"
		    "vfmadd231pd  64(%[_a], %%rcx), %%ymm5, %%ymm4;\n"
		    "vfmadd231pd  96(%[_a], %%rcx), %%ymm7, %%ymm6;\n"
		    "vfmadd231pd 128(%[_a], %%rcx), %%ymm9, %%ymm8;\n"
		    "vfmadd231pd 160(%[_a], %%rcx), %%ymm11, %%ymm10;\n"
		    "vfmadd231pd 192(%[_a], %%rcx), %%ymm13, %%ymm12;\n"
		    "vfmadd231pd 224(%[_a], %%rcx), %%ymm15, %%ymm14;\n"
		    
		    "add $256, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"

		    "vaddpd %%ymm2, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm6, %%ymm4, %%ymm4;\n"
		    "vaddpd %%ymm8, %%ymm10, %%ymm10;\n"
		    "vaddpd %%ymm12, %%ymm12, %%ymm14;\n"

		    "vaddpd %%ymm4, %%ymm0, %%ymm0;\n"
		    "vaddpd %%ymm14, %%ymm10, %%ymm10;\n"
		    
		    "vaddpd %%ymm10, %%ymm0, %%ymm0;\n"
		    
		    "vmovapd %%ymm0, (%[_r]);\n"
		    :
		    
		    :
		    [_a] "r" (a),
		    [_b] "r" (b),
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
    r[0] += a[i] * b[i];
  
  return r[0] + r[1] + r[2] + r[3];
}

void benchmark_dotprod(const ascii *title, f64 (*kernel)(f64 *restrict, f64 *restrict, u64), u64 n, u64 r)
{
  f64 v = 0.0;
  f64 e = 0.0;
  struct timespec t0, t1;
  u64 s = n * sizeof(f64);
  ydata_t *d = ydata_create(title, YBW_MAX_SAMPLES);
  
  for (u64 i = 0; i < YBW_MAX_SAMPLES; i++)
    {
      f64 *restrict a = aligned_alloc(64, sizeof(f64) * s);
      f64 *restrict b = aligned_alloc(64, sizeof(f64) * s);
      
      fill_f64(a, 1.0, n);
      fill_f64(b, 1.0, n);
      
      do
	{
	  clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
	  
	  for (u64 j = 0; j < r; j++)
	    v = kernel(a, b, n);
	  
	  clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
	  
	  e = (f64)(t1.tv_nsec - t0.tv_nsec) / (f64)r;
	}
      while (e <= 0.0);
      
      d->datapoints[i] = e;
      
      free(a);
      free(b);
    }
  
  ydata_analyze(d);
  
  //Arithmetic intensity in GFLOP/s for 2 FP operations (multiplication + addition)
  f64 ai = (f64)(n * 2) / d->min;
  
  //Memory bandwidth in GiB/s for 2 arrays (load + load)
  f64 bw = ((f64)(s * 2) / (1024.0 * 1024.0 * 1024.0)) / (d->min / 1e9);

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

  if (n < 1024)
    {
      printf("ERROR: 'n' cannot be below 1024\n");
      exit(1);
    }

  if (!r)
    {
      printf("ERROR: 'r' cannot be 0\n");
      exit(1);
    }
    
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
  
  benchmark_dotprod("dotprod C"             , dotprod_C             , n, r);
  benchmark_dotprod("dotprod SSE scalar"    , dotprod_SSE_scalar    , n, r);
  benchmark_dotprod("dotprod SSE vector x1" , dotprod_SSE_vector_x1 , n, r);
  benchmark_dotprod("dotprod SSE vector x2" , dotprod_SSE_vector_x2 , n, r);
  benchmark_dotprod("dotprod SSE vector x4" , dotprod_SSE_vector_x4 , n, r);
  benchmark_dotprod("dotprod AVX vector x1" , dotprod_AVX_vector_x1 , n, r);
  benchmark_dotprod("dotprod AVX vector x2" , dotprod_AVX_vector_x2 , n, r);
  benchmark_dotprod("dotprod AVX vector x4" , dotprod_AVX_vector_x4 , n, r);
  benchmark_dotprod("dotprod AVX vector x8" , dotprod_AVX_vector_x8 , n, r);
  
  return 0;
}
