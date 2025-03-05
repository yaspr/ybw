#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "consts.h"

#include "../../../ynotif/ynotif.h"
#include "../../../ydata/ydata.h"

void init_C(f64 *restrict a, f64 c, u64 n)
{
  for (u64 i = 0; i < n; i++)
    a[i] = c;
}

void init_SSE_scalar(f64 *restrict a, f64 c, u64 n)
{
  u64 s = sizeof(f64) * n;
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "movsd (%[_c]), %%xmm0;\n"
		    
		    "1:;\n"
		    
		    "movsd %%xmm0, (%[_a], %%rcx);\n"
		    
		    "add $8, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (&c),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0"
		    );
}

void init_SSE_vector_x1(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 1);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) cc[2] = { c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "movapd (%[_c]), %%xmm0;\n"
		    
		    "1:;\n"
		    
		    "movapd %%xmm0, (%[_a], %%rcx);\n"
		    
		    "add $16, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void init_SSE_vector_x2(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 3);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) cc[2] = { c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "movapd (%[_c]), %%xmm0;\n"
		    "movapd %%xmm0, %%xmm1;\n"
		    
		    "1:;\n"
		    
		    "movapd %%xmm0,   (%[_a], %%rcx);\n"
		    "movapd %%xmm1, 16(%[_a], %%rcx);\n"
		    
		    "add $32, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void init_SSE_vector_x4(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 7);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) cc[2] = { c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "movapd (%[_c]), %%xmm0;\n"
		    "movapd %%xmm0, %%xmm1;\n"
		    "movapd %%xmm0, %%xmm2;\n"
		    "movapd %%xmm0, %%xmm3;\n"
		    
		    "1:;\n"
		    
		    "movapd %%xmm0,   (%[_a], %%rcx);\n"
		    "movapd %%xmm1, 16(%[_a], %%rcx);\n"
		    "movapd %%xmm2, 32(%[_a], %%rcx);\n"
		    "movapd %%xmm3, 48(%[_a], %%rcx);\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void init_SSE_vector_x8(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 15);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(16))) cc[2] = { c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "movapd (%[_c]), %%xmm0;\n"
		    "movapd %%xmm0, %%xmm1;\n"
		    "movapd %%xmm0, %%xmm2;\n"
		    "movapd %%xmm0, %%xmm3;\n"
		    "movapd %%xmm0, %%xmm4;\n"
		    "movapd %%xmm0, %%xmm5;\n"
		    "movapd %%xmm0, %%xmm6;\n"
		    "movapd %%xmm0, %%xmm7;\n"
		    
		    "1:;\n"
		    
		    "movapd %%xmm0,    (%[_a], %%rcx);\n"
		    "movapd %%xmm1,  16(%[_a], %%rcx);\n"
		    "movapd %%xmm2,  32(%[_a], %%rcx);\n"
		    "movapd %%xmm3,  48(%[_a], %%rcx);\n"
		    "movapd %%xmm4,  64(%[_a], %%rcx);\n"
		    "movapd %%xmm5,  80(%[_a], %%rcx);\n"
		    "movapd %%xmm6,  96(%[_a], %%rcx);\n"
		    "movapd %%xmm7, 112(%[_a], %%rcx);\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3",
		    "xmm4", "xmm5", "xmm6", "xmm7"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void init_AVX_vector_x1(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 3);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) cc[4] = { c, c, c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "vmovapd (%[_c]), %%ymm0;\n"
		    
		    "1:;\n"
		    
		    "vmovapd %%ymm0, (%[_a], %%rcx);\n"
		    
		    "add $32, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void init_AVX_vector_x2(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 7);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) cc[4] = { c, c, c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "vmovapd (%[_c]), %%ymm0;\n"
		    "vmovapd %%ymm0, %%ymm1;\n"
		    
		    "1:;\n"
		    
		    "vmovapd %%ymm0,   (%[_a], %%rcx);\n"
		    "vmovapd %%ymm1, 32(%[_a], %%rcx);\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void init_AVX_vector_x4(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 15);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) cc[4] = { c, c, c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "vmovapd (%[_c]), %%ymm0;\n"
		    "vmovapd %%ymm0, %%ymm1;\n"
		    "vmovapd %%ymm0, %%ymm2;\n"
		    "vmovapd %%ymm0, %%ymm3;\n"
		    
		    "1:;\n"
		    
		    "vmovapd %%ymm0,   (%[_a], %%rcx);\n"
		    "vmovapd %%ymm1, 32(%[_a], %%rcx);\n"
		    "vmovapd %%ymm2, 64(%[_a], %%rcx);\n"
		    "vmovapd %%ymm3, 96(%[_a], %%rcx);\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void init_AVX_vector_x8(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 31);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) cc[4] = { c, c, c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "vmovapd (%[_c]), %%ymm0;\n"
		    "vmovapd %%ymm0, %%ymm1;\n"
		    "vmovapd %%ymm0, %%ymm2;\n"
		    "vmovapd %%ymm0, %%ymm3;\n"
		    "vmovapd %%ymm0, %%ymm4;\n"
		    "vmovapd %%ymm0, %%ymm5;\n"
		    "vmovapd %%ymm0, %%ymm6;\n"
		    "vmovapd %%ymm0, %%ymm7;\n"
		    
		    "1:;\n"
		    
		    "vmovapd %%ymm0,    (%[_a], %%rcx);\n"
		    "vmovapd %%ymm1,  32(%[_a], %%rcx);\n"
		    "vmovapd %%ymm2,  64(%[_a], %%rcx);\n"
		    "vmovapd %%ymm3,  96(%[_a], %%rcx);\n"
		    "vmovapd %%ymm4, 128(%[_a], %%rcx);\n"
		    "vmovapd %%ymm5, 160(%[_a], %%rcx);\n"
		    "vmovapd %%ymm6, 192(%[_a], %%rcx);\n"
		    "vmovapd %%ymm7, 224(%[_a], %%rcx);\n"
		    
		    "add $256, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void init_AVX_vector_x16(f64 *restrict a, f64 c, u64 n)
{
  u64 nn = n - (n & 63);
  u64 s = sizeof(f64) * nn;
  f64 __attribute__((aligned(32))) cc[4] = { c, c, c, c };
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"

		    "vmovapd (%[_c]), %%ymm0;\n"
		    "vmovapd %%ymm0, %%ymm1;\n"
		    "vmovapd %%ymm0, %%ymm2;\n"
		    "vmovapd %%ymm0, %%ymm3;\n"
		    "vmovapd %%ymm0, %%ymm4;\n"
		    "vmovapd %%ymm0, %%ymm5;\n"
		    "vmovapd %%ymm0, %%ymm6;\n"
		    "vmovapd %%ymm0, %%ymm7;\n"
		    "vmovapd %%ymm0, %%ymm8;\n"
		    "vmovapd %%ymm0, %%ymm9;\n"
		    "vmovapd %%ymm0, %%ymm10;\n"
		    "vmovapd %%ymm0, %%ymm11;\n"
		    "vmovapd %%ymm0, %%ymm12;\n"
		    "vmovapd %%ymm0, %%ymm13;\n"
		    "vmovapd %%ymm0, %%ymm14;\n"
		    "vmovapd %%ymm0, %%ymm15;\n"
		    
		    "1:;\n"
		    
		    "vmovapd %%ymm0,     (%[_a], %%rcx);\n"
		    "vmovapd %%ymm1,   32(%[_a], %%rcx);\n"
		    "vmovapd %%ymm2,   64(%[_a], %%rcx);\n"
		    "vmovapd %%ymm3,   96(%[_a], %%rcx);\n"
		    "vmovapd %%ymm4,  128(%[_a], %%rcx);\n"
		    "vmovapd %%ymm5,  160(%[_a], %%rcx);\n"
		    "vmovapd %%ymm6,  192(%[_a], %%rcx);\n"
		    "vmovapd %%ymm7,  224(%[_a], %%rcx);\n"
		    "vmovapd %%ymm8,  256(%[_a], %%rcx);\n"
		    "vmovapd %%ymm9,  288(%[_a], %%rcx);\n"
		    "vmovapd %%ymm10, 320(%[_a], %%rcx);\n"
		    "vmovapd %%ymm11, 352(%[_a], %%rcx);\n"
		    "vmovapd %%ymm12, 384(%[_a], %%rcx);\n"
		    "vmovapd %%ymm13, 416(%[_a], %%rcx);\n"
		    "vmovapd %%ymm14, 448(%[_a], %%rcx);\n"
		    "vmovapd %%ymm15, 480(%[_a], %%rcx);\n"
		    
		    "add $512, %%rcx;\n"
		    "cmp %[_s], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_a] "r" (a),
		    [_c] "r" (cc),
		    [_s] "r" (s)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7",
		    "ymm8", "ymm9", "ymm10", "ymm11",
		    "ymm12", "ymm13", "ymm14", "ymm15"
		    );

  for (u64 i = nn; i < n; i++)
    a[i] = c;
}

void benchmark_init(const ascii *title, void (*kernel)(f64 *restrict, f64, u64), u64 n, u64 r)
{
  f64 v = 0.0;
  f64 e = 0.0;
  struct timespec t0, t1;
  u64 s = n * sizeof(f64);
  ydata_t *d = ydata_create(title, YBW_MAX_SAMPLES);
  
  for (u64 i = 0; i < YBW_MAX_SAMPLES; i++)
    {
      f64 *restrict a = aligned_alloc(64, sizeof(f64) * s);
      
      kernel(a, 0.0, n);
      
      do
	{
	  clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

	  for (u64 j = 0; j < r; j++)
	    kernel(a, 1.1, n);
	  
	  clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

	  e = (f64)(t1.tv_nsec - t0.tv_nsec) / (f64)r;
	}
      while (e <= 0.0);

      d->datapoints[i] = e;

      v = a[0];
      
      free(a);
    }

  ydata_analyze(d);

  //GiB/s for 1 array (store)
  f64 bw = ((f64)s / (1024.0 * 1024.0 * 1024.0)) / (d->min / 1e9);

  printf("%25s; %13.3lf; %13.3lf; %13.3lf; %13.3lf; %13.3lf; %13.3lf; %18.3lf; %8.3lf; %13.3lf\n",
	 d->title,
	 (f64)s / (1024.0 * 1024.0),
	 v,
	 d->min,
	 d->max,
	 d->mean,
	 d->median,
	 d->min_relative_range,
	 d->coefficient_of_variation,
	 bw);
  
  ydata_destroy(&d);
}

i32 main(i32 argc, ascii **argv)
{
  if (argc < 3)
    return printf("Usage: %s [n] [r]\n", argv[0]), 1;

  u64 n = (u64)atoll(argv[1]);
  u64 r = (u64)atoll(argv[2]);
  
  printf("%25s; %13s; %13s; %13s; %13s; %13s; %13s; %18s; %8s; %13s\n",
	 "title",
	 "MiB",
	 "value",
	 "min",
	 "max",
	 "mean",
	 "median",
	 "(max-min)/min",
	 "CV",
	 "GiB/s");
  
  benchmark_init("init C"             , init_C             , n, r);
  benchmark_init("init SSE scalar"    , init_SSE_scalar    , n, r);
  benchmark_init("init SSE vector x1" , init_SSE_vector_x1 , n, r);
  benchmark_init("init SSE vector x2" , init_SSE_vector_x2 , n, r);
  benchmark_init("init SSE vector x4" , init_SSE_vector_x4 , n, r);
  benchmark_init("init SSE vector x8" , init_SSE_vector_x8 , n, r);
  benchmark_init("init AVX vector x1" , init_AVX_vector_x1 , n, r);
  benchmark_init("init AVX vector x2" , init_AVX_vector_x2 , n, r);
  benchmark_init("init AVX vector x4" , init_AVX_vector_x4 , n, r);
  benchmark_init("init AVX vector x8" , init_AVX_vector_x8 , n, r);
  benchmark_init("init AVX vector x16", init_AVX_vector_x16, n, r);
  
  return 0;
}
