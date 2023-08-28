#include <stdio.h>
#include <immintrin.h>
#include <time.h>

typedef struct matrix {  // Row-first
    float data[3][4];
} t_matrix;

typedef struct __attribute__((aligned(16))) vector {  // Column-first
    float data[4];
} t_vector;


t_matrix build_matrix(const float* row1, const float* row2, const float* row3) {
    t_matrix m;

    for (int i = 0; i < 3; i++) {
        m.data[0][i] = row1[i];
        m.data[1][i] = row2[i];
        m.data[2][i] = row3[i];
    }

    m.data[0][3] = 0;
    m.data[1][3] = 0;
    m.data[2][3] = 0;

    return m;
}

t_vector build_vector(float x, float y, float z) {
    t_vector v;

    v.data[0] = x;
    v.data[1] = y;
    v.data[2] = z;
    v.data[3] = 0;

    return v;
}


void naive_method(t_matrix *m, t_vector* vs, t_vector* out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i].data[0] = m->data[0][0] * vs[i].data[0] + m->data[0][1] * vs[i].data[1] + m->data[0][2] * vs[i].data[2] + m->data[0][3] * vs[i].data[3];
        out[i].data[1] = m->data[1][0] * vs[i].data[0] + m->data[1][1] * vs[i].data[1] + m->data[1][2] * vs[i].data[2] + m->data[1][3] * vs[i].data[3];
        out[i].data[2] = m->data[2][0] * vs[i].data[0] + m->data[2][1] * vs[i].data[1] + m->data[2][2] * vs[i].data[2] + m->data[2][3] * vs[i].data[3];
    }
}

__attribute__((optimize("O0"))) // Basically telling the compiler to NOT TOUCH THIS FUNCTION AT ALL, IT IS PERFECT AS IS
void vectorized_method(t_matrix *m, t_vector* vs, t_vector* out, long n) {
    float __attribute__((aligned(32))) row1[8];
    float __attribute__((aligned(32))) row2[8];
    float __attribute__((aligned(32))) row3[8];

    long n_even = n - n % 8;

    for (int i = 0; i < 4; i++) {
        row1[i] = m->data[0][i];
        row1[i + 4] = m->data[0][i];

        row2[i] = m->data[1][i];
        row2[i + 4] = m->data[1][i];

        row3[i] = m->data[2][i];
        row3[i + 4] = m->data[2][i];
    }

    __asm__ volatile (
            "xor %%r11, %%r11\n\t"
            "mov %3,    %%r8\n\t"
            "mov %4,    %%r9\n\t"
            "vmovaps (%0), %%ymm0\n\t" // Load row_1 into ymm0
            "vmovaps (%1), %%ymm1\n\t" // Load row_2 into ymm0
            "vmovaps (%2), %%ymm2\n\t" // Load row_3 into ymm2

        "vectorized_method_loop_start:\n\t"
            "vpxor %%ymm15, %%ymm15, %%ymm15\n\t" // Zero out ymm15
            "vpxor %%ymm11, %%ymm11, %%ymm11\n\t" // Zero out ymm11
            "vpxor %%ymm12, %%ymm12, %%ymm12\n\t" // Zero out ymm12
            "vpxor %%ymm13, %%ymm13, %%ymm13\n\t" // Zero out ymm13
            "vpxor %%ymm14, %%ymm14, %%ymm14\n\t" // Zero out ymm14

            "vmovaps   (%%r8), %%ymm3\n\t" // Load v12_source into ymm3
            "vmovaps 32(%%r8), %%ymm4\n\t" // Load v34_source into ymm4
            "vmovaps 64(%%r8), %%ymm5\n\t" // Load v56_source into ymm5
            "vmovaps 96(%%r8), %%ymm6\n\t" // Load v78_source into ymm6

            // Compute the Z component
            "vmulps %%ymm2, %%ymm3, %%ymm7\n\t" // Multiply row_1 by v12_source and store in ymm7 (v12)
            "vmulps %%ymm2, %%ymm4, %%ymm8\n\t" // Multiply row_1 by v34_source and store in ymm8 (v34)
            "vmulps %%ymm2, %%ymm5, %%ymm9\n\t" // Multiply row_1 by v56_source and store in ymm9 (v56)
            "vmulps %%ymm2, %%ymm6, %%ymm10\n\t" // Multiply row_1 by v78_source and store in ymm10 (v78)

            // Horizontal summation
            "vhaddps %%ymm7,  %%ymm7,  %%ymm7\n\t"
            "vhaddps %%ymm8,  %%ymm8,  %%ymm8\n\t"
            "vhaddps %%ymm9,  %%ymm9,  %%ymm9\n\t"
            "vhaddps %%ymm10, %%ymm10, %%ymm10\n\t"

            "vhaddps %%ymm7,  %%ymm7,  %%ymm7\n\t"
            "vhaddps %%ymm8,  %%ymm8,  %%ymm8\n\t"
            "vhaddps %%ymm9,  %%ymm9,  %%ymm9\n\t"
            "vhaddps %%ymm10, %%ymm10, %%ymm10\n\t"

            // Zero out unused components
            "vblendps $0b11101110, %%ymm15, %%ymm7,  %%ymm11\n\t" // Zeroing (v12_out)
            "vblendps $0b11101110, %%ymm15, %%ymm8,  %%ymm12\n\t" // Zeroing (v34_out)
            "vblendps $0b11101110, %%ymm15, %%ymm9,  %%ymm13\n\t" // Zeroing (v56_out)
            "vblendps $0b11101110, %%ymm15, %%ymm10, %%ymm14\n\t" // Zeroing (v78_out)

            // Moving z components from 0 to 2 place of each lane
            "vpermilps $0b11001111, %%ymm11, %%ymm11\n\t"
            "vpermilps $0b11001111, %%ymm12, %%ymm12\n\t"
            "vpermilps $0b11001111, %%ymm13, %%ymm13\n\t"
            "vpermilps $0b11001111, %%ymm14, %%ymm14\n\t"

            // Compute the Y component
            "vmulps %%ymm1, %%ymm3, %%ymm7\n\t" // Multiply row_2 by v12_source and store in ymm7 (v12)
            "vmulps %%ymm1, %%ymm4, %%ymm8\n\t" // Multiply row_2 by v34_source and store in ymm8 (v34)
            "vmulps %%ymm1, %%ymm5, %%ymm9\n\t" // Multiply row_2 by v56_source and store in ymm9 (v56)
            "vmulps %%ymm1, %%ymm6, %%ymm10\n\t" // Multiply row_2 by v78_source and store in ymm10 (v78)

            // Horizontal summation
            "vhaddps %%ymm7,  %%ymm7,  %%ymm7\n\t"
            "vhaddps %%ymm8,  %%ymm8,  %%ymm8\n\t"
            "vhaddps %%ymm9,  %%ymm9,  %%ymm9\n\t"
            "vhaddps %%ymm10, %%ymm10, %%ymm10\n\t"

            "vhaddps %%ymm7,  %%ymm7,  %%ymm7\n\t"
            "vhaddps %%ymm8,  %%ymm8,  %%ymm8\n\t"
            "vhaddps %%ymm9,  %%ymm9,  %%ymm9\n\t"
            "vhaddps %%ymm10, %%ymm10, %%ymm10\n\t"

            // Zero out unused components
            "vblendps $0b11101110, %%ymm15, %%ymm7,  %%ymm7\n\t" // Zeroing (v12_out)
            "vblendps $0b11101110, %%ymm15, %%ymm8,  %%ymm8\n\t" // Zeroing (v34_out)
            "vblendps $0b11101110, %%ymm15, %%ymm9,  %%ymm9\n\t" // Zeroing (v56_out)
            "vblendps $0b11101110, %%ymm15, %%ymm10, %%ymm10\n\t" // Zeroing (v78_out)

            // Adding the newly calculated component to the partially calculated vector
            "vaddps %%ymm11, %%ymm7,  %%ymm11\n\t"
            "vaddps %%ymm12, %%ymm8,  %%ymm12\n\t"
            "vaddps %%ymm13, %%ymm9,  %%ymm13\n\t"
            "vaddps %%ymm14, %%ymm10, %%ymm14\n\t"

            // Moving y components
            "vpermilps $0b11100011, %%ymm11, %%ymm11\n\t"
            "vpermilps $0b11100011, %%ymm12, %%ymm12\n\t"
            "vpermilps $0b11100011, %%ymm13, %%ymm13\n\t"
            "vpermilps $0b11100011, %%ymm14, %%ymm14\n\t"

            // Compute the X component
            "vmulps %%ymm0, %%ymm3, %%ymm7\n\t" // Multiply row_2 by v12_source and store in ymm7 (v12)
            "vmulps %%ymm0, %%ymm4, %%ymm8\n\t" // Multiply row_2 by v34_source and store in ymm8 (v34)
            "vmulps %%ymm0, %%ymm5, %%ymm9\n\t" // Multiply row_2 by v56_source and store in ymm9 (v56)
            "vmulps %%ymm0, %%ymm6, %%ymm10\n\t" // Multiply row_2 by v78_source and store in ymm10 (v78)

            // Horizontal summation
            "vhaddps %%ymm7, %%ymm7, %%ymm7\n\t"
            "vhaddps %%ymm8, %%ymm8, %%ymm8\n\t"
            "vhaddps %%ymm9, %%ymm9, %%ymm9\n\t"
            "vhaddps %%ymm10, %%ymm10, %%ymm10\n\t"

            "vhaddps %%ymm7,  %%ymm7,  %%ymm7\n\t"
            "vhaddps %%ymm8,  %%ymm8,  %%ymm8\n\t"
            "vhaddps %%ymm9,  %%ymm9,  %%ymm9\n\t"
            "vhaddps %%ymm10, %%ymm10, %%ymm10\n\t"

            // Zero out unused components
            "vblendps $0b11101110, %%ymm15, %%ymm7,  %%ymm7\n\t" // Zeroing (v12_out)
            "vblendps $0b11101110, %%ymm15, %%ymm8,  %%ymm8\n\t" // Zeroing (v34_out)
            "vblendps $0b11101110, %%ymm15, %%ymm9,  %%ymm9\n\t" // Zeroing (v56_out)
            "vblendps $0b11101110, %%ymm15, %%ymm10, %%ymm10\n\t" // Zeroing (v78_out)

            // Adding the newly calculated component to the partially calculated vector
            "vaddps %%ymm11, %%ymm7,  %%ymm11\n\t"
            "vaddps %%ymm12, %%ymm8,  %%ymm12\n\t"
            "vaddps %%ymm13, %%ymm9,  %%ymm13\n\t"
            "vaddps %%ymm14, %%ymm10, %%ymm14\n\t"

            "vmovaps %%ymm11,   (%%r9)\n\t" // Load v12_source into ymm1
            "vmovaps %%ymm12, 32(%%r9)\n\t" // Load v34_source into ymm2
            "vmovaps %%ymm13, 64(%%r9)\n\t" // Load v56_source into ymm3
            "vmovaps %%ymm14, 96(%%r9)\n\t" // Load v78_source into ymm4

            "add $8,   %%r11\n\t"
            "add $128, %%r8\n\t"
            "add $128, %%r9\n\t"

            "cmp %5, %%r11\n\t"
            "jge vectorized_method_loop_start\n\t"

            :
            : "r" (row1), "r" (row2), "r" (row3), "r" (vs), "r" (out), "r" (n_even)
            :   "ymm0",
                "ymm1",
                "ymm2",
                "ymm3",
                "ymm4",
                "ymm5",
                "ymm6",
                "ymm7",
                "ymm8",
                "ymm9",
                "ymm10",
                "ymm11",
                "ymm12",
                "ymm13",
                "ymm14",
                "ymm15"
    );

    if (n_even < n) {
        naive_method(m, vs + n_even, out + n_even, n - n_even);
    }
}


void print_vectors(t_vector* vs, int n) {
    for (int i = 0; i < n; i++) {
        printf("(%.3f, %.3f, %.3f)\n", vs[i].data[0], vs[i].data[1], vs[i].data[2]);
    }
}


int main() {
    t_matrix m = build_matrix((float[3]) {1, 2, 3}, (float[3]) {4, 5, 6}, (float[3]) {7, 8, 9});

    t_vector __attribute__((aligned(32)))  vs[140000];
    t_vector __attribute__((aligned(32)))  out[140000];

    for (int i = 0; i < 140000; i++) {
        vs[i] = build_vector(1, 2, 3);
    }

    clock_t naive, vectorized;

    // Running a Naive method 10 000 times
    printf("Naive method:\n");
    naive = clock();
    for (volatile int i = 0; i < 10000; i++) {
        naive_method(&m, vs, out, 140000);
    }
    naive = clock() - naive;
    print_vectors(out + rand() % 140000, 1);
    printf("\n\n");

    // Running a Vectorized method 10 000 times
    printf("Method 3:\n");
    vectorized = clock();
    for (volatile int i = 0; i < 10000; i++) {
        vectorized_method(&m, vs, out, 140000);
    }
    vectorized = clock() - vectorized;
    print_vectors(out + rand() % 140000, 1);
    printf("\n\n");

    printf("Naive method took %f seconds\n", ((double)naive) / CLOCKS_PER_SEC);
    printf("Vectorized method took %f seconds\n", ((double)vectorized) / CLOCKS_PER_SEC);

    return 0;
}
