#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <sys/time.h>
#include "generic_utils.h"
#include "spqlios/lagrangehalfc_impl.h"
#include "poc_64types.h"
#include <chrono>
#include <random>
#include <immintrin.h>
//using GlweCipher64 = TLweSample64;
extern Random* global_random;

// --- FFT processor accessor (thread-safe) ---
// In some TFHE/SPQLIOS variants only a global `fftp2048` object is exposed.
// Parallel FFT calls can break correctness if the processor uses internal scratch buffers.
// We therefore provide a thread-local accessor used by *_fft/ifft_lvl2 wrappers below.
#if defined(USE_FFT)
#ifndef HAVE_GET_FFTP2048
// -----------------------------------------------------------------------------
// OpenMP note: the SPQLIOS FFT processor uses internal scratch state and is not
// guaranteed to be re-entrant.  We therefore serialize calls to fftp2048's
// execute_* methods under OpenMP to preserve correctness.
// If you later confirm your FFT backend is re-entrant, you can remove the criticals.
// -----------------------------------------------------------------------------
static inline FFT_Processor_Spqlios& get_fftp2048() {
    return fftp2048;
}

static inline void fftp2048_execute_reverse_int(double* out, const int* in) {
#ifdef _OPENMP
#pragma omp critical(spqlios_fftp2048_exec)
    { fftp2048.execute_reverse_int(out, in); }
#else
    fftp2048.execute_reverse_int(out, in);
#endif
}

static inline void fftp2048_execute_direct_torus64(Torus64* out, const double* in) {
#ifdef _OPENMP
#pragma omp critical(spqlios_fftp2048_exec)
    { fftp2048.execute_direct_torus64(out, in); }
#else
    fftp2048.execute_direct_torus64(out, in);
#endif
}

static inline void fftp2048_execute_reverse_torus64(double* out, const Torus64* in) {
#ifdef _OPENMP
#pragma omp critical(spqlios_fftp2048_exec)
    { fftp2048.execute_reverse_torus64(out, in); }
#else
    fftp2048.execute_reverse_torus64(out, in);
#endif
}
#endif
#endif

// ǥ C++ ⸦ ̿ þ ÷
inline double sample_gaussian(double sigma) {
    // 帶
    static thread_local std::mt19937_64 gen(std::random_device{}());
    static thread_local std::normal_distribution<double> dist(0.0, 1.0);
    return dist(gen) * sigma; // N(0, sigma^2)
}


void torus64PolynomialMultNaive_plain_aux(Torus64* __restrict result, const int* __restrict poly1, const Torus64* __restrict poly2, const int N) {
    const int _2Nm1 = 2*N-1;
    Torus64 ri;
    for (int i=0; i<N; i++) {
        ri=0;
        for (int j=0; j<=i; j++) ri += poly1[j]*poly2[i-j];
        result[i]=ri;
    }
    for (int i=N; i<_2Nm1; i++) {
        ri=0;
        for (int j=i-N+1; j<N; j++) ri += poly1[j]*poly2[i-j];
        result[i]=ri;
    }
}


// A and B of size = size
// R of size = 2*size-1
void Karatsuba64_aux(Torus64* R, const int* A, const Torus64* B, const int size, const char* buf){
    const int h = size / 2;
    const int sm1 = size-1;

    //we stop the karatsuba recursion at h=4, because on my machine,
    //it seems to be optimal
    if (h<=4) {
        torus64PolynomialMultNaive_plain_aux(R, A, B, size);
        return;
    }

    //we split the polynomials in 2
    int* Atemp = (int*) buf; buf += h*sizeof(int);
    Torus64* Btemp = (Torus64*) buf; buf+= h*sizeof(Torus64);
    Torus64* Rtemp = (Torus64*) buf; buf+= size*sizeof(Torus64);
    //Note: in the above line, I have put size instead of sm1 so that buf remains aligned on a power of 2

    for (int i = 0; i < h; ++i) Atemp[i] = A[i] + A[h+i];
    for (int i = 0; i < h; ++i) Btemp[i] = B[i] + B[h+i];

    // Karatsuba recursivly
    Karatsuba64_aux(R, A, B, h, buf); // (R[0],R[2*h-2]), (A[0],A[h-1]), (B[0],B[h-1])
    Karatsuba64_aux(R+size, A+h, B+h, h, buf); // (R[2*h],R[4*h-2]), (A[h],A[2*h-1]), (B[h],B[2*h-1])
    Karatsuba64_aux(Rtemp, Atemp, Btemp, h, buf);
    R[sm1]=0; //this one needs to be set manually
    for (int i = 0; i < sm1; ++i) Rtemp[i] -= R[i] + R[size+i];
    for (int i = 0; i < sm1; ++i) R[h+i] += Rtemp[i];
}




// poly1, poly2 and result are polynomials mod X^N+1
void torus64PolynomialMultKaratsuba_lvl2(Torus64Polynomial* result, const IntPolynomiala* poly1, const Torus64Polynomial* poly2, const Globals* env){
    const int N2 = env->N;
    Torus64* R = new Torus64[2*N2-1];
    char* buf = new char[32*N2]; //that's large enough to store every tmp variables (2*2*N*8)

    // Karatsuba
    Karatsuba64_aux(R, poly1->coefs, poly2->coefs, N2, buf);

    // reduction mod X^N+1
    for (int i = 0; i < N2-1; ++i) result->coefs[i] = R[i] - R[N2+i];
    result->coefs[N2-1] = R[N2-1];

    delete[] R;
    delete[] buf;
}





void torus64PolynomialMultAddKaratsuba_lvl2(Torus64Polynomial* result, const IntPolynomiala* poly1, const Torus64Polynomial* poly2, const Globals* env){
    const int N2 = env->N;
    Torus64* R = new Torus64[2*N2-1];
    char* buf = new char[32*N2]; //that's large enough to store every tmp variables (2*2*N*8)

    // Karatsuba
    Karatsuba64_aux(R, poly1->coefs, poly2->coefs, N2, buf);

    // reduction mod X^N+1
    for (int i = 0; i < N2-1; ++i) result->coefs[i] += R[i] - R[N2+i];
    result->coefs[N2-1] += R[N2-1];

    delete[] R;
    delete[] buf;
}


void tLwe64EncryptZero(TLweSample64* cipher, const double stdev, const Globals* env){
    const int N = env->N;
    const int k= env->k;
    for (int j = 0; j < N; ++j) cipher->b->coefs[j] = random_gaussian64(0, stdev);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < N; ++j) cipher->a[i].coefs[j] = random_int64();
    }
    for (int i = 0; i < k; ++i) torus64PolynomialMultAddKaratsuba_lvl2(cipher->b, &env->tlwekey[i], &cipher->a[i], env);
}

void tLwe64Encrypt(TLweSample64* cipher,const Torus64Polynomial* mess, const double stdev, const Globals* env){
    const int N = env->N;
    // const int k= env->k;

   tLwe64EncryptZero(cipher, stdev, env);

    for (int32_t j = 0; j < N; ++j)
       cipher->b->coefs[j] += mess->coefs[j];
}

void tLwe64EncryptZero_debug(TLweSample64* cipher, const double stdev, const Globals* env){
    const int N = env->N;
    const int k= env->k;

    // ??   ߰
    for (int j = 0; j < N; ++j)
        cipher->b->coefs[j] = random_gaussian64(0, stdev); //0

    //   'a' ׽
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < N; ++j)
            cipher->a[i].coefs[j] = random_int64();
    }

    //  (s * a)
    for (int i = 0; i < k; ++i)
        torus64PolynomialMultAddKaratsuba_lvl2(cipher->b, &env->tlwekey[i], &cipher->a[i], env);

}

void tLwe64Encrypt_debug(TLweSample64* cipher, const Torus64Polynomial* mess, const double stdev, const Globals* env){
    const int N = env->N;


    tLwe64EncryptZero_debug(cipher, stdev, env);

    for (int32_t j = 0; j < N; ++j)
        cipher->b->coefs[j] += mess->coefs[j];

}



void tLwe64Phase_lvl2(Torus64Polynomial* phase, const TLweSample64* cipher, const Globals* env){
    const int N = env->N;
    const int k= env->k;

    //since we only have AddMult, we compute the opposite of the phase
    for (int j = 0; j < N; ++j) {
        phase->coefs[j] = -cipher->b->coefs[j];
    }



    for (int i = 0; i < k; ++i) {
        torus64PolynomialMultAddKaratsuba_lvl2(phase, &env->tlwekey[i], &cipher->a[i], env);
    }


    //and we negate the result
    for (int j = 0; j < N; ++j) {
        phase->coefs[j] = -phase->coefs[j];
    }

}


void circuitPrivKS(TLweSample64* result, const int u, const LweSample64* x, const Globals* env) {
    const int kslen = env->t;
    const int k=env->k;
    const int N = env->N; // N_lvl1 = n_lvl1
    const int basebit = env->basebit;
    const int base = 1<<basebit;       // base=2 in [CGGI16]
    const int mask = base - 1;
    const int64_t prec_offset = UINT64_C(1)<<(64-(1+basebit*kslen)); //precision ILA: revoir

    // clear result
    for (int i = 0; i <= k ; ++i) {
        for (int j = 0; j < N; ++j) {
            result->a[i].coefs[j] = 0;
        }
    }

    // Private Key Switching
    for (int i = 0; i <= N; ++i) {
        const uint64_t aibar = x->a[i] + prec_offset;

        for (int j = 0; j < kslen; ++j) {
            const uint64_t aij = (aibar>>(64-(j+1)*basebit)) & mask;

            if (aij != 0){
                for (int q = 0; q <= k; ++q) {
                        for(int p=0;p<N;++p){
        //result->a[q].values[p] -= env->privKS[u][i][j][aij].a[q].values[p]; }

result->a[q].coefs[p] -= env->privKS[u][i][j][aij].a[q].coefs[p];}
                }
            }
        }
    }

}

/////////////////////////////////////////////////////////////////
//     FFT SPECIFIC SECTION                                    //
/////////////////////////////////////////////////////////////////
/*
LagrangeHalfCPolynomiala* new_LagrangeHalfCPolynomiala_array(int nbelts, int N) {
    return new_array1<LagrangeHalfCPolynomiala>(nbelts,N);
}

void delete_LagrangeHalfCPolynomial_array(int nbelts, LagrangeHalfCPolynomiala* data) {
    delete_array1<LagrangeHalfCPolynomiala>(data);
}
*/

#ifdef USE_FFT
void IntPolynomial_ifft_lvl2(LagrangeHalfCPolynomiala* result, const IntPolynomiala* source, const Globals* env) {
    assert(env->N==2048);
    fftp2048_execute_reverse_int(result->values, source->coefs);
}


void LagrangeHalfCPolynomialClear_lvl2(LagrangeHalfCPolynomiala* result, const Globals* env) {
    const int N = env->N;
#if defined(__AVX2__)
    const __m256d z = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        _mm256_storeu_pd(result->values + i, z);
    }
    for (; i < N; ++i) result->values[i] = 0.0;
#else
    for (int i = 0; i < N; ++i) result->values[i] = 0.0;
#endif
}
void LagrangeHalfCPolynomialAddTo_lvl2(LagrangeHalfCPolynomiala* result, const LagrangeHalfCPolynomiala* a, const Globals* env) {
    const int N = env->N;
#if defined(__AVX2__)
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        __m256d r = _mm256_loadu_pd(result->values + i);
        __m256d x = _mm256_loadu_pd(a->values + i);
        r = _mm256_add_pd(r, x);
        _mm256_storeu_pd(result->values + i, r);
    }
    for (; i < N; ++i) result->values[i] += a->values[i];
#else
    for (int i = 0; i < N; ++i) result->values[i] += a->values[i];
#endif
}
void LagrangeHalfCPolynomialSubTo_lvl2(LagrangeHalfCPolynomiala* result, const LagrangeHalfCPolynomiala* a, const Globals* env) {
    const int N = env->N;
#if defined(__AVX2__)
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        __m256d r = _mm256_loadu_pd(result->values + i);
        __m256d x = _mm256_loadu_pd(a->values + i);
        r = _mm256_sub_pd(r, x);
        _mm256_storeu_pd(result->values + i, r);
    }
    for (; i < N; ++i) result->values[i] -= a->values[i];
#else
    for (int i = 0; i < N; ++i) result->values[i] -= a->values[i];
#endif
}
void LagrangeHalfCPolynomialAddMul_lvl2(LagrangeHalfCPolynomiala* result, const LagrangeHalfCPolynomiala* a, const LagrangeHalfCPolynomiala* b, const Globals* env) {
    const int Ns2 = env->N/2;
    LagrangeHalfCPolynomialAddMulASM(result->values, a->values, b->values, Ns2);
}

void TorusPolynomial64_fft_lvl2(Torus64Polynomial* result, const LagrangeHalfCPolynomiala* source, const Globals* env) {
    assert(env->N==2048);
    fftp2048_execute_direct_torus64(result->coefs, source->values);
}

void TorusPolynomial64_ifft_lvl2(LagrangeHalfCPolynomiala* result, const Torus64Polynomial* source, const Globals* env) {
    assert(env->N==2048);
    fftp2048_execute_reverse_torus64(result->values, source->coefs);
}


#else
//these are fake and slow versions of the FFT, that use Karatsuba instead
void IntPolynomial_ifft_lvl2(LagrangeHalfCPolynomiala* result, const IntPolynomiala* source, const Globals* env) {
    assert(env->N==2048);
    result->setIntPoly(source, 2048);
}


void LagrangeHalfCPolynomialClear_lvl2(LagrangeHalfCPolynomiala* result, const Globals* env) {
    assert(env->N==2048);
    result->setZeroTorus64Poly(2048);
}

void LagrangeHalfCPolynomialAddMul_lvl2(LagrangeHalfCPolynomiala* result, const LagrangeHalfCPolynomiala* a, const LagrangeHalfCPolynomiala* b, const Globals* env) {
    assert(env->N==2048);
    assert(result->torus64Poly!=0);
assert(a->intPoly!=0);
    assert(b->torus64Poly!=0);
    torus64PolynomialMultAddKaratsuba_lvl2(result->torus64Poly, a->intPoly, b->torus64Poly, env);
}

void TorusPolynomial64_fft_lvl2(Torus64Polynomial* result, const LagrangeHalfCPolynomiala* source, const Globals* env) {
    assert(env->N==2048);
    assert(source->torus64Poly!=0);
    for (int i=0; i<2048; i++) result->coefs[i]=source->torus64Poly->coefs[i];
}

void TorusPolynomial64_ifft_lvl2(LagrangeHalfCPolynomiala* result, const Torus64Polynomial* source, const Globals* env) {
    assert(env->N==2048);
    result->setTorus64Poly(source, 2048);
}
#endif



void tLwe64NoiselessTrivial(TLweSample64* cipher, const Torus64Polynomial* mess, const Globals* env){
    const int N = env->N;
    const int k= env->k;

     for (int i = 0; i <= k ; ++i) {
        for (int j = 0; j < N; ++j) {
            cipher->a[i].coefs[j] = 0;
        }
    }
 for (int j = 0; j < N; ++j) {

    cipher->b->coefs[j] = mess->coefs[j];
  }
}

void int_to_bin_digit(unsigned int in, int count, int64_t* out)
{
        unsigned int mask =1U << (count-1);
                int k;
                        for (k=0;k< count; k++){
                                                out[k]=(in & mask) ? 1 : 0;
                                                                       in <<=1;


                                                                                        }

}

void tGswTorus64PolynomialDecompH(IntPolynomiala* result, const Torus64Polynomial* sample, const Globals* env){
            const int N = env->N;
            const int l = env->l;
            const int Bgbit = env->bgbit;
            // peut etre tout cela dans le env
            const uint64_t Bg = UINT64_C(1)<<Bgbit;
            const uint64_t mask = Bg-1;
            const int64_t halfBg = Bg/2;
            uint64_t* buf = env->torusDecompBuf;
            const uint64_t offset = env->torusDecompOffset;

    //First, add offset to everyone
    for (int j = 0; j < N; ++j) buf[j]=sample->coefs[j]+offset;

    //then, do the decomposition (in parallel)
    for (int p = 0; p < l; ++p) {
        const int decal = (64-(p+1)*Bgbit);
         int* res_p = result[p].coefs; // res is a int (ok 32)
        for (int j = 0; j < N; ++j) {
            uint64_t temp1 = (buf[j] >> decal) & mask;
            res_p[j] = temp1 - halfBg;
        }
    }
}

        void tGsw64DecompH(IntPolynomiala* result, const TLweSample64* sample, const Globals* env){
    const int l = env->l;
    const int k=env->k;
    for (int i = 0; i <= k; ++i) tGswTorus64PolynomialDecompH(result+(i*l), &sample->a[i], env);
}

        void tGswExternMulToTLwe1(TLweSample64 *accum, const TGswSample64 *sample, const Globals *env) {

    const int32_t N = env->N;
    const int32_t k= env->k;
    const int l = env->l;
    const int32_t kpl = (k+1)*l;
    //TODO: improve this new/delete

 IntPolynomiala* decomp = new_array1<IntPolynomiala>(kpl,N);
    tGsw64DecompH(decomp, accum, env);
    // tLweClear(accum, par);
 for (int i = 0; i <= k ; ++i) {
        for (int j = 0; j < N; ++j) {
            accum->a[i].coefs[j] = 0;
        }
    }

    for (int32_t i = 0; i < kpl; i++) {
      //   tLweAddMulRTo(accum, &dec[i], &sample->all_sample[i], par);
      //  }

     for (int j = 0; j <= k; ++j) torus64PolynomialMultAddKaratsuba_lvl2(accum->a+j, &decomp[i], &sample->allsamples[i].a[j], env);
    }

}


        void CMux(TLweSample64 *result, const TGswSample64 *eps, const TLweSample64 *c0, TLweSample64 *c1, const Globals* env){

        const int l=env->l;
        const int N = env->N;
        const int k= env->k;
        const int kpl=(k+1)*l;

         IntPolynomiala* decomp = new_array1<IntPolynomiala>(kpl,N);
         LagrangeHalfCPolynomiala* decompFFT = new_array1<LagrangeHalfCPolynomiala>(kpl,N); Torus64Polynomial* phase = new Torus64Polynomial(N);

         TLweSampleFFTa* accFFT = new TLweSampleFFTa(N);
         TGswSampleFFTa* epsFFT= new TGswSampleFFTa(l,N);

         for (int i=0;i<kpl;i++)
                for (int q=0;q<=k;q++)
                  TorusPolynomial64_ifft_lvl2(&epsFFT->allsamples[i].a[q],&eps->allsamples[i].a[q],  env);

 //     tLweSubTo(c1,c0, params2);//c1=c1-c0
         for (int q = 0; q <= k; ++q)
         for (int j = 0; j < N; ++j) c1->a[q].coefs[j] -= c0->a[q].coefs[j];

 //tGswFFTExternMulToTLwe(c1, eps, params1);//c1=c1*eps

        tGsw64DecompH(decomp, c1, env);
        for (int p = 0; p < kpl; ++p) IntPolynomial_ifft_lvl2(decompFFT+p,decomp+p, env);
        // accFFT initialization
        for (int q = 0; q <= k; ++q) LagrangeHalfCPolynomialClear_lvl2(accFFT->a+q, env);

        // external product FFT
auto start = std::chrono::high_resolution_clock::now();
         for (int p = 0; p < kpl; ++p)
          for (int q = 0; q <= k; ++q) LagrangeHalfCPolynomialAddMul_lvl2(accFFT->a+q, decompFFT+p, &epsFFT->allsamples[p].a[q], env);

auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double,std::milli> execution_time = end-start;
/* debug disabled */

        // conversion from FFT
        for (int q = 0; q <= k; ++q) TorusPolynomial64_fft_lvl2(c1->a+q,accFFT->a+q, env);


//       tLweAddTo(c1, c0, params2);//c1=c1+c0

        for (int q = 0; q <= k; ++q)
          for (int j = 0; j < N; ++j) c1->a[q].coefs[j] += c0->a[q].coefs[j];
for (int q = 0; q <= k; ++q)
          for (int j = 0; j < N; ++j) result->a[q].coefs[j] = c1->a[q].coefs[j];

}




void CMuxFFT(TLweSample64 *result, const TGswSampleFFTa *eps, const TLweSample64 *c0, TLweSample64 *c1, const Globals* env){
    const int N = env->N;
    const int k = env->k;
    const int l = env->l;
    const int kpl = (k+1)*l;

    // Cache scratch buffers to avoid per-call allocations (single-thread assumption).
    struct Cache {
        int N=0, k=0, l=0, kpl=0;
        IntPolynomiala* decomp=nullptr;
        LagrangeHalfCPolynomiala* decompFFT=nullptr;
        TLweSampleFFTa* accFFT=nullptr;
    };
    static thread_local Cache cache;

    auto ensure_cache = [&]() {
        if (cache.accFFT && cache.N==N && cache.k==k && cache.l==l) return;
        if (cache.decomp)    delete_array1<IntPolynomiala>(cache.decomp);
        if (cache.decompFFT) delete_array1<LagrangeHalfCPolynomiala>(cache.decompFFT);
        if (cache.accFFT)    delete cache.accFFT;
        cache.N=N; cache.k=k; cache.l=l; cache.kpl=kpl;
        cache.decomp    = new_array1<IntPolynomiala>(kpl, N);
        cache.decompFFT = new_array1<LagrangeHalfCPolynomiala>(kpl, N);
        cache.accFFT    = new TLweSampleFFTa(N);
    };
    ensure_cache();

    // c1 = c1 - c0  (in coefficient/torus domain)
    for (int q = 0; q <= k; ++q)
        for (int j = 0; j < N; ++j)
            c1->a[q].coefs[j] -= c0->a[q].coefs[j];

    // Decompose (torus) and convert decomposition to FFT domain
    tGsw64DecompH(cache.decomp, c1, env);
    for (int p = 0; p < kpl; ++p)
        IntPolynomial_ifft_lvl2(cache.decompFFT + p, cache.decomp + p, env);

    // accFFT = 0
    for (int q = 0; q <= k; ++q)
        LagrangeHalfCPolynomialClear_lvl2(cache.accFFT->a + q, env);

    // accFFT += decompFFT ? eps  (external product in FFT domain)
    for (int p = 0; p < kpl; ++p)
        for (int q = 0; q <= k; ++q)
            LagrangeHalfCPolynomialAddMul_lvl2(cache.accFFT->a + q, cache.decompFFT + p, &eps->allsamples[p].a[q], env);

    // Convert back from FFT to torus: c1 = accFFT
    for (int q = 0; q <= k; ++q)
        TorusPolynomial64_fft_lvl2(c1->a + q, cache.accFFT->a + q, env);

    // c1 += c0
    for (int q = 0; q <= k; ++q)
        for (int j = 0; j < N; ++j)
            c1->a[q].coefs[j] += c0->a[q].coefs[j];

    // result = c1
    for (int q = 0; q <= k; ++q)
        for (int j = 0; j < N; ++j)
            result->a[q].coefs[j] = c1->a[q].coefs[j];
}
void CMuxFFTdb(TLweSampleFFTa *result, const TGswSampleFFTa *eps, const Torus64 c0, const Torus64 c1, const Globals* env){

        const int l=env->l;
        const int N = env->N;
        const int k= env->k;
        const int kpl=(k+1)*l;

        IntPolynomiala* decomp = new_array1<IntPolynomiala>(kpl,N);
         LagrangeHalfCPolynomiala* decompFFT = new_array1<LagrangeHalfCPolynomiala>(kpl,N);
        TLweSampleFFTa* accFFT= new TLweSampleFFTa(N);
        TLweSampleFFTa* tempFFT= new TLweSampleFFTa(N);
        TLweSample64 *temp = new TLweSample64(N);
        Torus64 cn= 0;
        for (int q = 0; q <= k; ++q) LagrangeHalfCPolynomialClear_lvl2(accFFT->a+q, env);

        for (int j = 0; j < N; ++j) {
                temp->a[0].coefs[j]=0;
                temp->a[1].coefs[j] =0;

                                      }
           temp->a[1].coefs[0]=c0;// Set data c0 as a noiseless TRLWE sample

        for (int q = 0; q <= k; ++q)

        TorusPolynomial64_ifft_lvl2(tempFFT->a+q,temp->a+q, env);// convert noiseless TRLWE sample c0 to fft form




         cn= c1-c0;//c1=c1-c0;


         for (int j = 0; j < N; ++j)
                temp->a[1].coefs[0] =cn;


        tGsw64DecompH(decomp, temp , env);


        for (int p = 0; p < kpl; ++p)

        IntPolynomial_ifft_lvl2(decompFFT+p,decomp+p, env);




        // external product FFT


         for (int p = 0; p < kpl; ++p)
          for (int q = 0; q <= k; ++q) LagrangeHalfCPolynomialAddMul_lvl2(accFFT->a+q, decompFFT+p, &eps->allsamples[p].a[q], env);


        // conversion from FFT

   for (int q = k; q <= k; ++q) LagrangeHalfCPolynomialAddTo_lvl2(accFFT->a+q,tempFFT->a+q, env); //c1=c1+c0


        for (int q = 0; q <= k; ++q)
          for (int j = 0; j < N; ++j) result->a[q].values[j] = accFFT->a[q].values[j];


        delete_array1<IntPolynomiala>(decomp);
        delete_array1<LagrangeHalfCPolynomiala>(decompFFT);
        delete accFFT;
        delete tempFFT;
        delete temp;


}
void CMuxFFTa(TLweSampleFFTa *result, const TGswSampleFFTa *eps, const TLweSampleFFTa *c0, TLweSampleFFTa *c1, const Globals* env){
    const int N = env->N;
    const int k = env->k;
    const int l = env->l;
    const int kpl = (k+1)*l;

    // Cache scratch buffers to avoid per-call allocations (single-thread assumption).
    struct Cache {
        int N=0, k=0, l=0, kpl=0;
        IntPolynomiala* decomp=nullptr;
        LagrangeHalfCPolynomiala* decompFFT=nullptr;
        TLweSampleFFTa* accFFT=nullptr;
        TLweSample64*  accTorus=nullptr; // used for gadget decomposition input
    };
    static thread_local Cache cache;

    auto ensure_cache = [&]() {
        if (cache.accFFT && cache.N==N && cache.k==k && cache.l==l) return;
        if (cache.decomp)    delete_array1<IntPolynomiala>(cache.decomp);
        if (cache.decompFFT) delete_array1<LagrangeHalfCPolynomiala>(cache.decompFFT);
        if (cache.accFFT)    delete cache.accFFT;
        if (cache.accTorus)  delete cache.accTorus;
        cache.N=N; cache.k=k; cache.l=l; cache.kpl=kpl;
        cache.decomp    = new_array1<IntPolynomiala>(kpl, N);
        cache.decompFFT = new_array1<LagrangeHalfCPolynomiala>(kpl, N);
        cache.accFFT    = new TLweSampleFFTa(N);
        cache.accTorus  = new TLweSample64(N);
    };
    ensure_cache();

    // accFFT = 0
    for (int q = 0; q <= k; ++q)
        LagrangeHalfCPolynomialClear_lvl2(cache.accFFT->a + q, env);

    // c1 = c1 - c0   (in FFT domain)
    for (int q = 0; q <= k; ++q) {
        LagrangeHalfCPolynomialSubTo_lvl2(c1->a + q, c0->a + q, env);
        // Convert diff (FFT) -> torus for decomposition
        TorusPolynomial64_fft_lvl2(cache.accTorus->a + q, c1->a + q, env);
    }

    // Decompose in torus domain and convert decomposition to FFT domain
    tGsw64DecompH(cache.decomp, cache.accTorus, env);
    for (int p = 0; p < kpl; ++p)
        IntPolynomial_ifft_lvl2(cache.decompFFT + p, cache.decomp + p, env);

    // accFFT += decompFFT ? eps
    for (int p = 0; p < kpl; ++p)
        for (int q = 0; q <= k; ++q)
            LagrangeHalfCPolynomialAddMul_lvl2(cache.accFFT->a + q, cache.decompFFT + p, &eps->allsamples[p].a[q], env);

    // accFFT += c0  (so result = c0 + (c1-c0)*eps)
    for (int q = 0; q <= k; ++q)
        LagrangeHalfCPolynomialAddTo_lvl2(cache.accFFT->a + q, c0->a + q, env);

    // Copy to result
    for (int q = 0; q <= k; ++q)
        for (int j = 0; j < N; ++j)
            result->a[q].values[j] = cache.accFFT->a[q].values[j];
}
void CMuxDecompFFT(TLweSample64* c0 ,const TGswSampleFFTa *eps, const  LagrangeHalfCPolynomiala* decompFFT, const Globals* env){
    const int N = env->N;
    const int k = env->k;
    const int l = env->l;
    const int kpl = (k+1)*l;

    // Cache scratch buffers (single-thread assumption).
    struct Cache {
        int N=0, k=0, l=0, kpl=0;
        TLweSampleFFTa* accFFTa=nullptr;
        TLweSample64*   tmpTorus=nullptr;
    };
    static thread_local Cache cache;

    auto ensure_cache = [&]() {
        if (cache.accFFTa && cache.N==N && cache.k==k && cache.l==l) return;
        if (cache.accFFTa)  delete cache.accFFTa;
        if (cache.tmpTorus) delete cache.tmpTorus;
        cache.N=N; cache.k=k; cache.l=l; cache.kpl=kpl;
        cache.accFFTa = new TLweSampleFFTa(N);
        cache.tmpTorus = new TLweSample64(N);
    };
    ensure_cache();

    for (int q = 0; q <= k; ++q)
        LagrangeHalfCPolynomialClear_lvl2(cache.accFFTa->a + q, env);

    for (int p = 0; p < kpl; ++p)
        for (int q = 0; q <= k; ++q)
            LagrangeHalfCPolynomialAddMul_lvl2(cache.accFFTa->a + q, decompFFT + p, &eps->allsamples[p].a[q], env);

    // back to torus and add to c0
    for (int q = 0; q <= k; ++q)
        TorusPolynomial64_fft_lvl2(cache.tmpTorus->a + q, cache.accFFTa->a + q, env);

    for (int q = 0; q <= k; ++q)
        for (int j = 0; j < N; ++j)
            c0->a[q].coefs[j] += cache.tmpTorus->a[q].coefs[j];
}
void CMuxDecompFFTa(TLweSampleFFTa *c0 ,const TGswSampleFFTa *eps, const  LagrangeHalfCPolynomiala* decompFFT, const Globals* env){
    const int N = env->N;
    const int k = env->k;
    const int l = env->l;
    const int kpl = (k+1)*l;

    // Cache accumulator (single-thread assumption).
    struct Cache {
        int N=0, k=0, l=0, kpl=0;
        TLweSampleFFTa* accFFTa=nullptr;
    };
    static thread_local Cache cache;

    auto ensure_cache = [&]() {
        if (cache.accFFTa && cache.N==N && cache.k==k && cache.l==l) return;
        if (cache.accFFTa) delete cache.accFFTa;
        cache.N=N; cache.k=k; cache.l=l; cache.kpl=kpl;
        cache.accFFTa = new TLweSampleFFTa(N);
    };
    ensure_cache();

    for (int q = 0; q <= k; ++q)
        LagrangeHalfCPolynomialClear_lvl2(cache.accFFTa->a + q, env);

    for (int p = 0; p < kpl; ++p)
        for (int q = 0; q <= k; ++q)
            LagrangeHalfCPolynomialAddMul_lvl2(cache.accFFTa->a + q, decompFFT + p, &eps->allsamples[p].a[q], env);

    for (int q = 0; q <= k; ++q)
        LagrangeHalfCPolynomialAddTo_lvl2(c0->a + q, cache.accFFTa->a + q, env);
}
void shift(TLweSample64 *result, int j,TLweSample64 *sample,int N){

//int N = env->N;
//int k=env->k;


        for (int i=0; i<N; i++){
          if (i+j<N){
  result->a[0].coefs[j+i] = sample->a[0].coefs[i];
   result->a[1].coefs[j+i] = sample->a[1].coefs[i];
          }
          else
           {
   result->a[0].coefs[(i+j)%N]= -sample->a[0].coefs[i];
   result->a[1].coefs[(i+j)%N] = -sample->a[1].coefs[i];

            }


         }
        }

        void tGsw64Encrypt(TGswSample64* cipher, const int mess, const double stdev, const Globals* env){
            const int l = env->l;
            const int Bgbit = env->bgbit;
            const int k=env->k;
         for (int bloc = 0; bloc <= k; ++bloc) {
         for (int i = 0; i < l; ++i) {
            // encryption of 0
            tLwe64EncryptZero(&cipher->samples[bloc][i], stdev, env);
            // add mess*h[i]
            cipher->samples[bloc][i].a[bloc].coefs[0] += mess * (UINT64_C(1) << (64-(i+1)*Bgbit));
                                                }
                                     }
          }


    void tGsw64Encrypt_poly(TGswSample64* cipher, const IntPolynomiala* mess, const double stdev, const Globals* env) {
    const int l = env->l;
    const int Bgbit = env->bgbit;
    const int k = env->k;

    for (int bloc = 0; bloc <= k; ++bloc) {
        for (int i = 0; i < l; ++i) {
            // Step 1: ȣȭ 0
            tLwe64EncryptZero(&cipher->samples[bloc][i], stdev, env);

            // Step 2: message * gadget vector h[i] ߰
            for (int j = 0; j < env->N; ++j) {
                cipher->samples[bloc][i].a[bloc].coefs[j] += mess->coefs[j] * (UINT64_C(1) << (64 - (i + 1) * Bgbit));
            }
        }
    }
}


    void tGsw64Encrypt_poly_2(TGswSample64* cipher, const IntPolynomiala* mess, const double stdev, const Globals* env){
    const int N = env->N;
    const int l = env->l;
    const int Bgbit = env->bgbit;
    const int k = env->k;

    //for (int bloc = 0; bloc <= k; ++bloc) {
        for (int i = 0; i < l; ++i) {
            // Step 1: ȣȭ 0
            tLwe64EncryptZero_debug(&cipher->samples[0][i], stdev, env);
            tLwe64EncryptZero_debug(&cipher->samples[1][i], stdev, env);


            // Step 2: message * gadget vector h[i] ߰
            for (int j = 0; j < N; ++j) {
                cipher->samples[0][i].a[0].coefs[j] += mess->coefs[j] * (UINT64_C(1) << (64 - (i + 1) * Bgbit));
                cipher->samples[1][i].a[1].coefs[j] += mess->coefs[j] * (UINT64_C(1) << (64 - (i + 1) * Bgbit));
            }
        }
    //}
}

    void tGsw64Encrypt_poly_3(TGswSample64* cipher, const IntPolynomiala* mess, const double stdev, const Globals* env){
    const int N = env->N;
    const int l = env->l;
    const int Bgbit = env->bgbit;
    const int k = env->k;

    for (int bloc = 0; bloc <= k; ++bloc) {
        for (int i = 0; i < l; ++i) {
            // Step 1: ȣȭ 0
            tLwe64EncryptZero_debug(&cipher->samples[bloc][i], stdev, env);


            // Step 2: message * gadget vector h[i] ߰
            for (int j = 0; j < env->N; ++j) {
                cipher->samples[bloc][i].a[1].coefs[j] += mess->coefs[j] * (UINT64_C(1) << (64 - (i + 1) * Bgbit));
            }
        }
    }
}

void tLweExtractLweSampleIndex64(LweSample64* result, const TLweSample64* x, const int32_t index, const Globals *env) {
    const int32_t N = env->N;
    const int32_t k = env->k;
    assert(env->smalln == k*N);

    for (int32_t i=0; i<k; i++) {
      for (int32_t j=0; j<=index; j++)
        result->a[i*N+j] = x->a[i].coefs[index-j];
      for (int32_t j=index+1; j<N; j++)
        result->a[i*N+j] = -x->a[i].coefs[N+index-j];
    }
    result->a[N] = x->a[k].coefs[index];
}



Torus64 lwe64Phase_lvl2(const LweSample64* cipher, const Globals* env) {
    const int n = env->N;
    Torus64 res = *cipher->b;
    for (int i = 0; i < n; ++i) {
        res -= cipher->a[i]*env->lwekey[i];
    }
    return res;
}


void packing_algorithm2(TLweSample64* rlweResult, const TGswSample64** ksk, const TLweSample64* rlweInput, const Globals* env) {
    // ȯ
    const int k = env->k;            // TLWE Ű
    const int N = env->N;            // ׽
    const int t = env->t;            //  ܰ
    const int basebit = env->basebit; // Gadget  basebit

    // Gadget  g
    const int base = 1 << basebit;   // base = 2^basebit
    const int l = env->l;            // Gadget
    const int kpl = (k+1)*l;

    // ӽ Decomposition
    IntPolynomiala* decomp = new_array1<IntPolynomiala>(kpl, N);
    LagrangeHalfCPolynomiala* decompFFT = new_array1<LagrangeHalfCPolynomiala>(kpl, N);

    // ߰   accFFT
    TLweSampleFFTa** accFFT = new TLweSampleFFTa*[N];
    for (int j = 0; j < N; ++j) {
        accFFT[j] = new TLweSampleFFTa(N);
        for (int q = 0; q <= k; ++q) {
            LagrangeHalfCPolynomialClear_lvl2(accFFT[j]->a + q, env);
        }
    }

    // KSK FFT
    TGswSampleFFTa** kskFFT = new TGswSampleFFTa*[N];
    for (int j = 0; j < N; ++j) {
        kskFFT[j] = new TGswSampleFFTa(l, N); //  KSK FFT ʱȭ
        for (int i = 0; i < kpl; ++i) {
            for (int q = 0; q <= k; ++q) {
                TorusPolynomial64_ifft_lvl2(&kskFFT[j]->allsamples[i].a[q], &ksk[j]->allsamples[i].a[q], env);
            }
        }
    }

    //  ʱȭ
    for (int z = 0; z <= k; ++z) {
        for (int i = 0; i < N; ++i) {
            rlweResult->a[z].coefs[i] = 0; //  ʱȭ
        }
    }

    // RLWE Է ù ° component decomposition ϰ, FFT ȯ
    tGswTorus64PolynomialDecompH(decomp, &rlweInput->a[0], env);
    for (int p = 0; p < kpl; ++p) {
        IntPolynomial_ifft_lvl2(decompFFT + p, decomp + p, env);
    }

    // N KSK   g inverse ksk pointwiseϰ ϰ ϱ
    for (int j = 0; j < N; ++j) {
        for (int p = 0; p < kpl; ++p) {
            for (int q = 0; q <= k; ++q) {
                LagrangeHalfCPolynomialAddMul_lvl2(
                    accFFT[j]->a + q,                 //   ġ
                    decompFFT + p,                   // FFT ȯ decomp
                    &kskFFT[j]->allsamples[p].a[q],  // KSK FFT
                    env
                );
            }
        }
    }

    // accFFT Ϲ · ȯϿ  TLweSample64 ȯ
    TLweSample64** acc = new TLweSample64*[N];
    for (int j = 0; j < N; ++j) {
        acc[j] = new TLweSample64(N);
        for (int q = 0; q <= k; ++q) {
            TorusPolynomial64_fft_lvl2(acc[j]->a + q, accFFT[j]->a + q, env);
        }
    }

    // ȯ acc rlweResult ջ
    for (int z = 0; z <= k; ++z) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                rlweResult->a[z].coefs[i] += acc[j]->a[z].coefs[i];
            }
        }
    }

    // Extract   b ߰
    for (int q = 0; q <= k; ++q) {
        for (int i = 0; i < N; ++i) {
            rlweResult->b->coefs[i] += rlweInput->b->coefs[i];
        }
    }

    // ޸
    delete_array1<IntPolynomiala>(decomp);
    delete_array1<LagrangeHalfCPolynomiala>(decompFFT);
    for (int j = 0; j < N; ++j) {
        delete accFFT[j];
        delete kskFFT[j];
        delete acc[j];
    }
    delete[] accFFT;
    delete[] kskFFT;
    delete[] acc;

    std::cout << "algorithm2 packing done" << std::endl;
}


void KSKGen_RGSW(TGswSample64* ksk, const IntPolynomiala* info_sk, const Globals* env) {
    // ȯ
    const int l = env->l;            //  ܰ
    const int Bgbit = env->bgbit;   // gadget  basebit
    const int N = env->N;           // ׽
    const double stdev = pow(2., -55); // ȣȭ Ǵ ǥ

    // gadget
    std::vector<uint64_t> gadget_vector(l);
    for (int i = 0; i < l; ++i) {
        // gadget vector Ƿ  ϰ
        gadget_vector[i] = (UINT64_C(1) << (64 - (i + 1) * Bgbit));
    }

    // Key Switching Key
    for (int i = 0; i < l; ++i) {  //
        // 0 ȣȭ
        tLwe64EncryptZero(&ksk->samples[0][i], stdev, env);

        // gadget vector ̿ sk info_sk ߰
        for (int j = 0; j < N; ++j) {
            ksk->samples[0][i].a[0].coefs[j] += info_sk->coefs[j] * gadget_vector[i];
            ksk->samples[0][i].a[1].coefs[j] += info_sk->coefs[j] * gadget_vector[i];
        }
    }

    std::cout << "Key switching key generation completed." << std::endl;
}

void KSKGen_RGSW_2_debug(TGswSample64* ksk, const IntPolynomiala* info_sk, const Globals* env) {
    // ȯ
    const int l = env->l;            //  ܰ
    const int Bgbit = env->bgbit;     // gadget  basebit
    const int N = env->N;             // ׽
    const double stdev = pow(2., -55); // ȣȭ Ǵ ǥ

    // ? Step 1: gadget
    std::vector<uint64_t> gadget_vector(l);
    for (int i = 0; i < l; ++i) {
        gadget_vector[i] = (UINT64_C(1) << (64 - (i + 1) * Bgbit));
    }
    // Key Switching Key
    for (int i = 0; i < l; ++i) {  //

        for (int j = 0; j < N; ++j) {
            ksk->samples[0][i].b->coefs[j] = 0; //random_gaussian64(0, stdev);
        }

        // ?? Step 2:  a ׽
        for (int j = 0; j < N; ++j) {
            ksk->samples[0][i].a[0].coefs[j] = random_int64();
        }

        // ?? Step 3: secret key (env->tlwekey) a ׽
        torus64PolynomialMultAddKaratsuba_lvl2(ksk->samples[0][i].b, env->tlwekey, &ksk->samples[0][i].a[0], env);

        // ?? Step 4: gadget_vector Ͽ info_sk  ߰ (a[1] )
        for (int j = 0; j < N; ++j) {
            ksk->samples[0][i].a[1].coefs[j] += info_sk->coefs[j] * gadget_vector[i];

        }
    }

}


void unpacking_algorithm4(TGswSample64* result, const TLweSample64** rlweInputs, const TGswSample64* convk, const Globals* env) {
    // ȯ
    const int l = env->l;            //  ܰ
    const int N = env->N;            // ׽

    //  ʱȭ
    for (int k = 0; k <= env->k; ++k) {
        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < N; ++j) {
                result->samples[k][i].a[0].coefs[j] = rlweInputs[i]->a[0].coefs[j];
                result->samples[k][i].a[1].coefs[j] = rlweInputs[i]->a[1].coefs[j];
            }
        }
    }

    // External Product
    for (int i = 0; i < l; ++i) {
        // C[i] <- ExternalProd(C_i, convk)
        tGswExternMulToTLwe1(&result->samples[0][i], convk, env);

    }
}

void unpacking_algorithm5(TGswSample64* result, const TLweSample64** rlweInputs, const TGswSample64* convk, const Globals* env) {
    // ȯ
    const int l = env->l;            //  ܰ
    const int N = env->N;            // ׽
    const int k = env->k;            //

    //  ʱȭ
    for (int k = 0; k <= env->k; ++k) {
        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < N; ++j) {
               result->samples[k][i].a[k].coefs[j] = rlweInputs[i]->a[k].coefs[j];
            }
        }
    }

    // Algorithm 5
    for (int i = 0; i < l; ++i) {
        // Step 1: Decompose g^{-1}(ci[1])
        IntPolynomiala* decomp = new_array1<IntPolynomiala>(env->l, N);
        tGswTorus64PolynomialDecompH(decomp, &rlweInputs[i]->a[0], env);

        // Step 2: Compute g^{-1}(ci[1]) * convk using direct multiplications
        TLweSample64* temp = new TLweSample64(N);
        for (int j = 0; j < env->l; ++j) {
            for (int z = 0; z <= k; ++z) {
                torus64PolynomialMultAddKaratsuba_lvl2(temp->a + z, &decomp[j], &convk->samples[z][j].a[z], env);
            }
        }

        // Step 3: Compute C[i] = (0, ci[2]) - temp
        //for (int z = 0; z <= k; ++z) {
            for (int j = 0; j < N; ++j) {
                result->samples[0][i].a[0].coefs[j] = -temp->a[0].coefs[j];
                result->samples[0][i].b->coefs[j] = rlweInputs[i]->b->coefs[j] - temp->b->coefs[j];
            }
        //}

        // ޸
        delete_array1<IntPolynomiala>(decomp);
        delete temp;
    }

    std::cout << "Unpacking algorithm5 completed." << std::endl;
}



void Alg3_XPowerShift(TLweSample64* result, const TLweSample64* input, int shift, const Globals* env) {
    const int N = env->N;  // ׽  (: 1024)

    for (int j = 0; j < N; ++j) {
        int new_index = (j * shift) % N;  // X^N = -1 ݿϿ ο
        Torus64 sign = ((j * shift) / N) % 2 == 0 ? 1 : -1;  // X^N = -1   ȣ

        result->a[0].coefs[new_index] = sign * input->a[0].coefs[j]; // ȣ
        result->b->coefs[new_index] = sign * input->b->coefs[j]; // b  ϰ
    }
}

void Alg3_XPowerShift_sk(IntPolynomiala* result, const IntPolynomiala* input, int shift, const Globals* env) {
    const int N = env->N;  // ׽  (: 1024)

    for (int j = 0; j < N; ++j) {
        int new_index = (j * shift) % N;  // X^N = -1 ݿϿ ο
        Torus64 sign = ((j * shift) / N) % 2 == 0 ? 1 : -1;  // X^N = -1   ȣ

        result->coefs[new_index] = sign * input->coefs[j]; // ȯ
    }
}

void Alg3_XPowerShift_TorusPoly(Torus64Polynomial* result, const Torus64Polynomial* input, int shift, const Globals* env) {
    const int N = env->N;  // ׽  (: 1024)

    for (int j = 0; j < N; ++j) {
        int new_index = (j * shift) % N;  // X^N = -1 ݿϿ ο
        Torus64 sign = ((j * shift) / N) % 2 == 0 ? 1 : -1;  // X^N = -1   ȣ

        result->coefs[new_index] = sign * input->coefs[j]; // ȯ
    }
}

void packing_algorithm3(TLweSample64* result, const TLweSample64* rlweInput, const TGswSample64** ksk, const Globals* env) {
    const int k = env->k;
    const int N = env->N;
    const int l = env->l;
    const int logN = log2(N);

    // ?? logN  kskFFT
    TGswSampleFFTa** kskFFT = new TGswSampleFFTa*[logN];
    for (int i = 0; i < logN; ++i) {
        kskFFT[i] = new TGswSampleFFTa(l, N);
        for (int p = 0; p < l; ++p)
            for (int q = 0; q <= k; ++q)
                TorusPolynomial64_ifft_lvl2(&kskFFT[i]->allsamples[p].a[q], &ksk[i]->allsamples[p].a[q], env);
    }

    //  rlweInput ʱȭ
    for (int i = 0; i <= k; ++i) {
        for (int j = 0; j < N; ++j) {
            result->a[i].coefs[j] = rlweInput->a[i].coefs[j];
        }
    }


    for (int iter = 0; iter < logN; ++iter) {
        int shift = (N / (1 << iter)) + 1;

        TLweSample64* c_prime = new TLweSample64(N);
        Alg3_XPowerShift(c_prime, result, shift, env);


        // c_prime a[0], a[1] и
        Torus64Polynomial* c_prime_a0 = new Torus64Polynomial(N);
        Torus64Polynomial* c_prime_a1 = new Torus64Polynomial(N);

        for (int j = 0; j < N; ++j) {
            c_prime_a0->coefs[j] = c_prime->a[0].coefs[j];
            c_prime_a1->coefs[j] = c_prime->a[1].coefs[j];
        }

        // c_prime_a0 decomp · ȯ
        IntPolynomiala* decomp_c_prime_a0 = new_array1<IntPolynomiala>(l, N);
        tGswTorus64PolynomialDecompH(decomp_c_prime_a0, c_prime_a0, env);

        // FFT ȯ
        LagrangeHalfCPolynomiala* decompFFT_c_prime_a0 = new_array1<LagrangeHalfCPolynomiala>(l, N);

        for (int p = 0; p < l; ++p) {
            IntPolynomial_ifft_lvl2(decompFFT_c_prime_a0 + p, decomp_c_prime_a0 + p, env);
        }

        // External product  (?? iter  kskFFT )
        TLweSampleFFTa* accFFT = new TLweSampleFFTa(N);

        //accFFT initialization
        for(int q=0; q<=k; ++q) LagrangeHalfCPolynomialClear_lvl2(accFFT->a+q, env);

        for (int p = 0; p < l; ++p) {
            LagrangeHalfCPolynomialAddMul_lvl2(accFFT->a, decompFFT_c_prime_a0 + p, &kskFFT[iter]->allsamples[p].a[0], env);
            LagrangeHalfCPolynomialAddMul_lvl2(accFFT->a + 1, decompFFT_c_prime_a0 + p, &kskFFT[iter]->allsamples[p].a[1], env);
        }


        // Convert back from FFT domain
        TLweSample64* acc = new TLweSample64(N);
        for (int q = 0; q <= k; ++q)
            TorusPolynomial64_fft_lvl2(acc->a + q, accFFT->a + q, env);


        // temp_result = 0 - acc for a, c_prime->b - acc for b
        TLweSample64* temp_result = new TLweSample64(N);
        for (int j = 0; j < N; ++j){
            temp_result->a[0].coefs[j] = -acc->a[0].coefs[j];
            temp_result->b->coefs[j] = c_prime_a1->coefs[j] - acc->b->coefs[j];
        }


        // result = temp_result + result
        for (int i = 0; i <= k; ++i)
            for (int j = 0; j < N; ++j)
                result->a[i].coefs[j] += temp_result->a[i].coefs[j];



        // ޸
        delete_array1<IntPolynomiala>(decomp_c_prime_a0);
        delete_array1<LagrangeHalfCPolynomiala>(decompFFT_c_prime_a0);

        delete accFFT;
        delete acc;
        delete temp_result;
        delete c_prime_a0;
        delete c_prime_a1;
        delete c_prime;
    }

    // ?? logN  kskFFT
    for (int i = 0; i < logN; ++i) {
        delete kskFFT[i];
    }
    delete[] kskFFT;

}

void packing_algorithm3_debug(TLweSample64* result, const TLweSample64* rlweInput, const TGswSample64** ksk, const Globals* env) {
    const int k = env->k;
    const int N = env->N;
    const int l = env->l;
    const int logN = log2(N);

    std::cout << "?? Starting packing_algorithm3" << std::endl;

    // ?? logN  kskFFT
    TGswSampleFFTa** kskFFT = new TGswSampleFFTa*[logN];
    for (int i = 0; i < logN; ++i) {
        kskFFT[i] = new TGswSampleFFTa(l, N);
        std::cout << "? kskFFT[" << i << "] allocated." << std::endl;
        for (int p = 0; p < l; ++p)
            for (int q = 0; q <= k; ++q)
                TorusPolynomial64_ifft_lvl2(&kskFFT[i]->allsamples[p].a[q], &ksk[i]->allsamples[p].a[q], env);
    }

    //  rlweInput ʱȭ
    for (int i = 0; i <= k; ++i) {
        for (int j = 0; j < N; ++j) {
            result->a[i].coefs[j] = rlweInput->a[i].coefs[j];
        }
    }
    std::cout << "? Result initialized with rlweInput" << std::endl;

    for (int iter = 0; iter < logN; ++iter) {
        int shift = (N / (1 << iter)) + 1;
        std::cout << "?? Iteration: " << iter << ", Shift: " << shift << std::endl;

        TLweSample64* c_prime = new TLweSample64(N);
        Alg3_XPowerShift(c_prime, result, shift, env);

        // ?? Debug: c_prime  Ȯ
        std::cout << "? c_prime first 5 coefficients (a[0]): ";
        for (int i = 0; i < 5; i++) std::cout << c_prime->a[0].coefs[i] << " ";
        std::cout << std::endl;

        // c_prime a[0], a[1] и
        Torus64Polynomial* c_prime_a0 = new Torus64Polynomial(N);
        Torus64Polynomial* c_prime_a1 = new Torus64Polynomial(N);

        for (int j = 0; j < N; ++j) {
            c_prime_a0->coefs[j] = c_prime->a[0].coefs[j];
            c_prime_a1->coefs[j] = c_prime->a[1].coefs[j];
        }

        // c_prime_a0, c_prime_a1 decomp · ȯ
        IntPolynomiala* decomp_c_prime_a0 = new_array1<IntPolynomiala>(l, N);
        //IntPolynomiala* decomp_c_prime_a1 = new_array1<IntPolynomiala>(l, N);

        tGswTorus64PolynomialDecompH(decomp_c_prime_a0, c_prime_a0, env);
        //tGswTorus64PolynomialDecompH(decomp_c_prime_a1, c_prime_a0, env);

        // ?? Debug: Decomposed values Ȯ
        std::cout << "? Decomposed first 5 coefficients (c_prime_a0): ";
        for (int i = 0; i < 5; i++) std::cout << decomp_c_prime_a0->coefs[i] << " ";
        std::cout << std::endl;

        // FFT ȯ
        LagrangeHalfCPolynomiala* decompFFT_c_prime_a0 = new_array1<LagrangeHalfCPolynomiala>(l, N);
        //LagrangeHalfCPolynomiala* decompFFT_c_prime_a1 = new_array1<LagrangeHalfCPolynomiala>(l, N);

        for (int p = 0; p < l; ++p) {
            IntPolynomial_ifft_lvl2(decompFFT_c_prime_a0 + p, decomp_c_prime_a0 + p, env);
            //IntPolynomial_ifft_lvl2(decompFFT_c_prime_a1 + p, decomp_c_prime_a1 + p, env);
        }

        // External product  (?? iter  kskFFT )
        TLweSampleFFTa* accFFT = new TLweSampleFFTa(N);

        //accFFT initialization
        for(int q=0; q<=k; ++q) LagrangeHalfCPolynomialClear_lvl2(accFFT->a+q, env);

        for (int p = 0; p < l; ++p) {
            LagrangeHalfCPolynomialAddMul_lvl2(accFFT->a, decompFFT_c_prime_a0 + p, &kskFFT[iter]->allsamples[p].a[0], env);
            LagrangeHalfCPolynomialAddMul_lvl2(accFFT->a + 1, decompFFT_c_prime_a0 + p, &kskFFT[iter]->allsamples[p].a[1], env);
        }


        // Convert back from FFT domain
        TLweSample64* acc = new TLweSample64(N);
        for (int q = 0; q <= k; ++q)
            TorusPolynomial64_fft_lvl2(acc->a + q, accFFT->a + q, env);

        // ?? Debug: acc ȯ  Ȯ
        std::cout << "? acc first 5 coefficients (a[0]): ";
        for (int i = 0; i < 5; i++) std::cout << acc->a[0].coefs[i] << " ";
        std::cout << std::endl;


        // temp_result = 0 - acc for a, c_prime->b - acc for b
        TLweSample64* temp_result = new TLweSample64(N);
        for (int j = 0; j < N; ++j){
            temp_result->a[0].coefs[j] = -acc->a[0].coefs[j];
            temp_result->b->coefs[j] = c_prime_a1->coefs[j] - acc->b->coefs[j];
        }

        // ?? Debug: temp_result ȯ  Ȯ
        std::cout << "? temp_result first 5 coefficients (a[0]): ";
        for (int i = 0; i < 5; i++) std::cout << temp_result->a[0].coefs[i] << " ";
        std::cout << std::endl;

        // result = temp_result + result
        for (int i = 0; i <= k; ++i)
            for (int j = 0; j < N; ++j)
                result->a[i].coefs[j] += temp_result->a[i].coefs[j];

        // ?? Debug: Result Ʈ  Ȯ
        std::cout << "? Result after update first 5 coefficients: ";
        for (int i = 0; i < 5; i++) std::cout << result->a[0].coefs[i] << " ";
        std::cout << std::endl;

        // ޸
        delete_array1<IntPolynomiala>(decomp_c_prime_a0);
        //delete_array1<IntPolynomiala>(decomp_c_prime_a1);
        delete_array1<LagrangeHalfCPolynomiala>(decompFFT_c_prime_a0);
        //delete_array1<LagrangeHalfCPolynomiala>(decompFFT_c_prime_a1);

        delete accFFT;
        delete acc;
        delete temp_result;
        delete c_prime_a0;
        delete c_prime_a1;
        delete c_prime;
    }

    // ?? logN  kskFFT
    for (int i = 0; i < logN; ++i) {
        delete kskFFT[i];
    }
    delete[] kskFFT;

    std::cout << "? Algorithm 3 packing completed successfully!" << std::endl;
}





void left_shift_by_one(TLweSample64 *output, TLweSample64 *input, int N){
    for(int i=0; i<N; i++){
        output->a[0].coefs[i] = input->a[0].coefs[i+1];
        output->a[1].coefs[i] = input->a[1].coefs[i+1];
    }
    output->a[0].coefs[N-1] = -input->a[0].coefs[0];
    output->a[1].coefs[N-1] = -input->a[1].coefs[0];
}

void left_shift_by_one_poly(IntPolynomiala *output, IntPolynomiala *input, int N){
    for(int i=0; i<N; i++){
        output->coefs[i] = input->coefs[i+1];
    }

}

// j  j*d    e = j*d = qN + r, X^e = (-1)^q X^r
inline void tau_d_pos_sign(int j, int d, int N,
                           int& out_index, int& out_sign) {
    long long e = 1LL * j * d;
    int q = (int)(e / N);
    int r = (int)(e % N);
    if (r < 0) { r += N; q -= 1; }

    out_index = r;
    out_sign  = (q & 1) ? -1 : 1;
}

// R = Z[X]/(X^N+1)  _d(f)(X) = f(X^d)
// IntPolynomiala(  ׽)
void tau_d_secret(IntPolynomiala* out,
                  const IntPolynomiala* in,
                  int d,
                  const Globals* env)
{
    const int N = env->N;

    //  0 ʱȭ
    for (int i = 0; i < N; ++i)
        out->coefs[i] = 0;

    for (int j = 0; j < N; ++j) {
        int coeff = in->coefs[j];
        if (coeff == 0) continue;

        int idx, sgn;
        tau_d_pos_sign(j, d, N, idx, sgn);

        out->coefs[idx] += sgn * coeff;
    }
}

// Torus  ׽Ŀ _d : f(X) -> f(X^d)
void tau_d_poly(Torus64Polynomial* out,
                const Torus64Polynomial* in,
                int d,
                const Globals* env)
{
    const int N = env->N;

    for (int i = 0; i < N; ++i)
        out->coefs[i] = 0;

    for (int j = 0; j < N; ++j) {
        Torus64 coeff = in->coefs[j];
        if (coeff == 0) continue;

        int idx, sgn;
        tau_d_pos_sign(j, d, N, idx, sgn);

        out->coefs[idx] += (Torus64)sgn * coeff;
    }
}


// GLWE ct = (a[0](X), a[1](X)=b(X))  _d
// in: Ű S Ʒ ȣ C
// out: Ű _d(S) Ʒ ȣ C^(d)
void tau_d_cipher(TLweSample64* out,
                  const TLweSample64* in,
                  int d,
                  const Globals* env)
{
    const int k = env->k; // ⼱ k=1

    for (int i = 0; i <= k; ++i) {
        tau_d_poly(&out->a[i], &in->a[i], d, env);
    }
    tau_d_poly(out->b, in->b, d, env);
}



// IntPolynomiala ( ׽)
//  i  ° 2^{64-(i+1)*bgbit}  ؼ
// ϳ GLWE ȣ ȣȭ
void glwe_encrypt_poly_scaled_level(
    GlweCipher64* ct,
    const IntPolynomiala* msg, //  ޽ m(X)
    int level,                 // 0 <= level < l
    double stdev,
    const Globals* env
) {
    const int N = env->N;
    const int Bgbit = env->bgbit;

    // 1.  ׽ Torus64 ׽ ؼ ű
    Torus64Polynomial* tor_msg = new Torus64Polynomial(N);

    // gadget i°  : 2^{64-(i+1)*Bgbit}
    uint64_t scale = UINT64_C(1) << (64 - (level + 1) * Bgbit);

    for (int j = 0; j < N; ++j) {
        // msg->coefs[j] * scale  Torus64
        tor_msg->coefs[j] = (Torus64)msg->coefs[j] * (Torus64)scale;
    }

    // 2.  TLWE(=GLWE) ȣȭ ״ ȣ
    tLwe64Encrypt(ct, tor_msg, stdev, env);

    delete tor_msg;
}

// IntPolynomiala ( S_d(X))
// level i  ° 2^{64-(i+1)*bgbit}  ؼ
// ϳ "GLWE Ű"   (⼭ a=0, b=scaled(S_d))
void glwe_encrypt_poly_scaled_level_noiseless(
    GlweCipher64* ct,
    const IntPolynomiala* msg, // S_d(X)
    int level,                 // 0 <= level < l
    const Globals* env
) {
    const int N     = env->N;
    const int Bgbit = env->bgbit;

    uint64_t scale = UINT64_C(1) << (64 - (level + 1) * Bgbit);

    // ⼭ ¥ ȣȭ  ϰ,
    //   a = 0,  b = msg * scale
    // θ  (noiseless ȣ)
    for (int j = 0; j < N; ++j) {
        ct->a[0].coefs[j] = 0;
        ct->a[1].coefs[j] = 0;          // k=1̴ϱ    0
        ct->b->coefs[j]   = (Torus64)msg->coefs[j] * (Torus64)scale;
    }
}


// Ű(GLev): S_d(X) ؼ ֵ, b þ  ߰
void glwe_encrypt_poly_scaled_level_evalkey(
    GlweCipher64* ct,
    const IntPolynomiala* msg, // S_d(X)
    int level,                 // 0 <= level < l
    double stdev,              // Ű   ǥ (torus )
    const Globals* env
) {
    const int N     = env->N;
    const int Bgbit = env->bgbit;
    const int k     = env->k;   // ⼱ k=1

    // gadget i°  : 2^{64-(i+1)*Bgbit}
    const uint64_t scale = UINT64_C(1) << (64 - (level + 1) * Bgbit);

    for (int j = 0; j < N; ++j) {
        // a κ KS Ŀ  ϱ 0
        ct->a[0].coefs[j] = 0;
        if (k >= 1) ct->a[1].coefs[j] = 0;

        // (1)  κ: ϵ S_d
        Torus64 m_scaled = (Torus64)msg->coefs[j] * (Torus64)scale;

        // (2) KS  : e ~ N(0, stdev^2) in R/Z
        //  - Random Ŭ ȿ gaussian(double)  Լ ִٰ
        //    Լ ̸ ٸ ⸸   ° ٲ!
        double e_real = sample_gaussian(stdev);

        // torus : 1.0 ? 2^63 ó ϰ
        long double w = (long double)e_real * (long double)((unsigned long long)1 << 63);
        Torus64 e_torus = (Torus64) llround(w);

        // (3) b = m_scaled + e
        ct->b->coefs[j] = m_scaled + e_torus;
    }
}


// msg(X) = S_d(X) ޾Ƽ
//     i  glwe_encrypt_poly_scaled_level_noiseless  ä
void GLevEncryptPoly(
    GLevCipher64* glev,
    const IntPolynomiala* msg,   // S_d(X)
double stdev,
const Globals* env
) {
    const int l = glev->l;
    for (int i = 0; i < l; ++i) {
        glwe_encrypt_poly_scaled_level_evalkey(
            (*glev)[i],   // i° GLWE
            msg,
            i, stdev,
            env
        );
    }
}


AutoKsKey64* AutoKsKeyGen64(const Globals* env, int d) {
    const int N = env->N;
    const int l = env->l;

    // 1.  Ű S
    const IntPolynomiala* S = env->tlwekey;

    // 2. S_d = tau_d(S)
    IntPolynomiala* S_d = new IntPolynomiala(N);
    tau_d_secret(S_d, S, d, env);

    // 3. AutoKsKey Ҵ
    AutoKsKey64* aks = new AutoKsKey64(d, l, N);

    // 4. GLev_q,S(S_d) = ( "ȣ"  ƴ϶, ϵ S_d Ʈ )
double ks_stdev = pow(2.,-55);
GLevEncryptPoly(aks->glev, S_d, ks_stdev, env);

    delete S_d;
    return aks;
}



// GLWE.KS: (k=1 RLWE ) ŰĪ
void GLWE_KS_only(
    TLweSample64* out,                 // : Ű S   ȣ
    const TLweSample64* in,            // Է: Ű S_d  ȣ
    const AutoKsKey64* aks,            // Atk_d = GLev_q,S(S_d)
    const Globals* env
) {
    const int N = env->N;
    const int l = env->l;
    const int k = env->k;  // k=1

    // 0. out ʱȭ: a=0, b = in.b ()
    for (int q = 0; q <= k; ++q)
        for (int j = 0; j < N; ++j)
            out->a[q].coefs[j] = 0;

    for (int j = 0; j < N; ++j)
        out->b->coefs[j] = in->b->coefs[j];

    // 1. in.a[0](X) gadget base 2^{bgbit}
    IntPolynomiala* decomp = new_array1<IntPolynomiala>(l, N);
    tGswTorus64PolynomialDecompH(decomp, &in->a[0], env);

    // 2. ӽ
    Torus64Polynomial* tmp = new Torus64Polynomial(N);

    // 3. out.b -= sum_p decomp[p] * GLev[p].b
    for (int p = 0; p < l; ++p) {
        GlweCipher64* kct = (*aks->glev)[p];  // level p "Ű"
        // tmp(X) = decomp[p](X) * kct->b(X)
        torus64PolynomialMultKaratsuba_lvl2(tmp, &decomp[p], kct->b, env);

        // out.b -= tmp
        for (int j = 0; j < N; ++j)
            out->b->coefs[j] -= tmp->coefs[j];
    }

    delete_array1<IntPolynomiala>(decomp);
    delete tmp;
}


void EvalAuto(
    GlweCipher64* out,
    const GlweCipher64* in,
    const AutoKsKey64* aks,
    const Globals* env
) {
    const int N = env->N;

    // 1. tau_d_cipher: Ű S  S_d ٲ GLWE
    GlweCipher64* ct_tau = new GlweCipher64(N);
    tau_d_cipher(ct_tau, in, aks->d, env);  // ̹  Ǵ  Ȯ

    // 2. GLWE.KS_only: S_d  S ŰĪ
    GLWE_KS_only(out, ct_tau, aks, env);

    delete ct_tau;
}

// x 2^bits 鼭 "ݿø"ؼ Torus64  ModSwitch ٿ
inline Torus64 modswitch_down_Torus64(Torus64 x, int bits) {
    if (bits <= 0) return x;
    // ȣ ϸ鼭 round(x / 2^bits)
    Torus64 half = (Torus64)1 << (bits - 1);  //

    if (x >= 0) {
        return (x + half) >> bits;
    } else {
        Torus64 y = -x;
        y = (y + half) >> bits;
        return -y;
    }
}


inline Torus64 modraise_up_Torus64(Torus64 x, int bits) {
    (void)bits;  //
    return x;    // RevHomTrace  ModRaise ""
}


// GLWE ModSwitchDown: () ⷯ q -> q/2^bits  δٰ
void GlweModSwitchDown(
    TLweSample64* out,
    const TLweSample64* in,
    int bits,
    const Globals* env
) {
    const int N = env->N;
    const int k = env->k;

    // a κ
    for (int j = 0; j < N; ++j) {
        for (int q = 0; q <= k; ++q) {
            out->a[q].coefs[j] = modswitch_down_Torus64(in->a[q].coefs[j], bits);
        }
    }

    // b κ
    for (int j = 0; j < N; ++j) {
        out->b->coefs[j] = modswitch_down_Torus64(in->b->coefs[j], bits);
    }
}

// GLWE ModRaiseUp: () ⷯ q/2^bits -> q  øٰ
void GlweModRaiseUp(
    TLweSample64* out,
    const TLweSample64* in,
    int bits,
    const Globals* env
) {
    (void)bits;  // RevHomTrace

    const int N = env->N;
    const int k = env->k;

    // a κ: ܼ
    for (int j = 0; j < N; ++j) {
        for (int q = 0; q <= k; ++q) {
            out->a[q].coefs[j] = in->a[q].coefs[j];
        }
    }

    // b κ: ܼ
    for (int j = 0; j < N; ++j) {
        out->b->coefs[j] = in->b->coefs[j];
    }
}


// ------------------------------------------------------
//  d_k = 2^k + 1   AutoKsKey
// ------------------------------------------------------

// ǿ: log2(N)
inline int int_log2(int x) {
    int r = 0;
    while ((1 << r) < x) ++r;
    return r;
}

// AutoKsKeyAll[i]
//   k = logn+1+i  شϴ d_k = 2^k + 1  Ű Ű
//
// i : 0 .. (logN - logn - 1)
// , ü  = logN - logn
AutoKsKey64** AutoKsKeyGenAll(const Globals* env, int n) {
    const int N    = env->N;
    const int logN = int_log2(N);
    const int logn = int_log2(n);

    const int num_k = logN - logn;          //  k
    AutoKsKey64** aks_list = new AutoKsKey64*[num_k];

    for (int i = 0; i < num_k; ++i) {
        int k = logn + 1 + i;              //  ˰ k
        int d = (1 << k) + 1;              // d_k = 2^k + 1
        aks_list[i] = AutoKsKeyGen64(env, d);
    }
    return aks_list;
}

void DeleteAutoKsKeyAll(AutoKsKey64** aks_list, const Globals* env, int n) {
    if (!aks_list) return;

    const int N    = env->N;
    const int logN = int_log2(N);
    const int logn = int_log2(n);
    const int num_k = logN - logn;

    for (int i = 0; i < num_k; ++i) {
        if (aks_list[i]) {
            // AutoKsKey64 ȿ new  ͵(glev )
            // ʵ Ҹ/  ǵǾ ־ .
            delete aks_list[i];
        }
    }
    delete[] aks_list;
}



// ------------------------------------------------------
// (IntPolynomial)  RevHomTrace
//   in : m(X)
//   out: ( ˰ 5   )
// ------------------------------------------------------
// Algorithm 5 plaintext
//   in  : M(X)
//   out : C' شϴ ׽ (˰ 5  C')
void RevHomTrace_plaintext(
    IntPolynomiala* out,
    const IntPolynomiala* in,
    int n,
    const Globals* env
) {
    const int N    = env->N;
    const int logN = int_log2(N);
    const int logn = int_log2(n);

    // C' <- M
    IntPolynomiala* Cprime = new IntPolynomiala(N);
    IntPolynomiala* Cbar   = new IntPolynomiala(N);
    IntPolynomiala* Cauto  = new IntPolynomiala(N);

    for (int i = 0; i < N; ++i)
        Cprime->coefs[i] = in->coefs[i];

    // for k = log n + 1 .. log N:
    //   Cbar  <- C'
    //   Cauto <- tau_{d_k}(Cbar),  d_k = 2^k + 1
    //   C'    <- Cbar + Cauto
    for (int k = logn + 1; k <= logN; ++k) {
        int d = (1 << k) + 1; // d_k

        // Cbar <- C'
        for (int i = 0; i < N; ++i)
            Cbar->coefs[i] = Cprime->coefs[i];

        // Cauto <- tau_d(Cbar)
        tau_d_secret(Cauto, Cbar, d, env);

        // C' <- Cbar + Cauto
        for (int i = 0; i < N; ++i)
            Cprime->coefs[i] = Cbar->coefs[i] + Cauto->coefs[i];
    }

    //
    for (int i = 0; i < N; ++i)
        out->coefs[i] = Cprime->coefs[i];

    delete Cprime;
    delete Cbar;
    delete Cauto;
}


// ------------------------------------------------------
// ȣ(GLWE)  RevHomTrace (Algorithm 5)
//   in  : Enc_S(m(X))
//   aks_list[k] : d_k = 2^k + 1   AutoKsKey64*
//   n   : target  (˰ 5 log n, log N  )
// ------------------------------------------------------
/*
void RevHomTrace(
    GlweCipher64* out,              //  Enc_S(RevHomTrace(m))
    const GlweCipher64* in,         // Է Enc_S(m)
    AutoKsKey64** aks_list,         // automorphism KS Ű  (k=logn+1..logN )
    int n,                          // trace
    const Globals* env
){
    const int N    = env->N;
    const int kdim = env->k;          // k=1
    const int logN = int_log2(N);
    const int logn = int_log2(n);

    // C' <- in  (ȣ ʿ " ")
    GlweCipher64* Cprime   = new GlweCipher64(N);
    GlweCipher64* Cbar     = new GlweCipher64(N);
    GlweCipher64* Cbar_low = new GlweCipher64(N);
    GlweCipher64* Cauto    = new GlweCipher64(N);

    // Cprime = in
    for (int q = 0; q <= kdim; ++q) {
        for (int j = 0; j < N; ++j) {
            Cprime->a[q].coefs[j] = in->a[q].coefs[j];
        }
    }
    for (int j = 0; j < N; ++j) {
        Cprime->b->coefs[j] = in->b->coefs[j];
    }

    // Algorithm 5:
    // for k = log n + 1 .. log N:
    //   d_k = 2^k + 1
    //   C? <- C'
    //   C' <- C? + _{d_k}(C?)
    //
    // 츮 ʿ:
    //   _{d_k}  EvalAuto + (ModSwitchDown/Up) + AutoKsKey64[k]
    for (int kk = logn + 1; kk <= logN; ++kk) {
        AutoKsKey64* ak = aks_list[kk];
        if (!ak) {
            //    ߻ϸ  , Ȥ  ŵ
            continue;
        }

        // C? <- C'
        for (int q = 0; q <= kdim; ++q) {
            for (int j = 0; j < N; ++j) {
                Cbar->a[q].coefs[j] = Cprime->a[q].coefs[j];
            }
        }
        for (int j = 0; j < N; ++j) {
            Cbar->b->coefs[j] = Cprime->b->coefs[j];
        }

        // ( ModSwitch ܰ)
        // q -> q/2, ٽ q : ModSwitchDown + ModRaiseUp
        GlweModSwitchDown(Cbar_low, Cbar, 1, env);
        GlweModRaiseUp   (Cbar,     Cbar_low, 1, env);

        // _{d_k} : EvalAuto(Cauto, Cbar, ak)
        EvalAuto(Cauto, Cbar, ak, env);

        // C' <- C? + Cauto
        for (int q = 0; q <= kdim; ++q) {
            for (int j = 0; j < N; ++j) {
                Cprime->a[q].coefs[j] = Cbar->a[q].coefs[j] + Cauto->a[q].coefs[j];
            }
        }
        //for (int j = 0; j < N; ++j) {
        //   Cprime->b->coefs[j] = Cbar->b->coefs[j] + Cauto->b->coefs[j];
        //}
    }

    //   out <- C'
    for (int q = 0; q <= kdim; ++q) {
        for (int j = 0; j < N; ++j) {
            out->a[q].coefs[j] = Cprime->a[q].coefs[j];
        }
    }
    for (int j = 0; j < N; ++j) {
        out->b->coefs[j] = Cprime->b->coefs[j];
    }

    delete Cprime;
    delete Cbar;
    delete Cbar_low;
    delete Cauto;
}
*/

// RevHomTrace: Algorithm 5 (Lee?Yoon, "Homomorphic Field Trace Revisited")
//
// Input : in      = GLWE_q,S(m(X))
//         aks_list[k-1] = AutoKS key for d = 2^k + 1, k = 1..logN
//         n       = power of two, n | N
// Output: out     = GLWE_q,S( (n/N)  Tr_{K/K_n}(m(X)) )
//                   Ư n=1̸  m_0 ״ .
//
void RevHomTrace_Alg5(
    GlweCipher64* out,
    const GlweCipher64* in,
    AutoKsKey64** aks_list,
    int n,
    const Globals* env
){
    const int N = env->N;
    const int k = env->k;

    // --- logN, logn ---
    int logN = 0;
    {
        int tmp = N;
        while (tmp > 1) { tmp >>= 1; ++logN; }
    }
    int logn = 0;
    {
        int tmp = n;
        while (tmp > 1) { tmp >>= 1; ++logn; }
    }

    // --- C' <- C ---
    for (int q = 0; q <= k; ++q)
        for (int j = 0; j < N; ++j)
            out->a[q].coefs[j] = in->a[q].coefs[j];
    //for (int j = 0; j < N; ++j)
    //    out->b->coefs[j] = in->b->coefs[j];

    if (n == N) return;

    GlweCipher64* Cbar_low = new GlweCipher64(N);
    GlweCipher64* Cbar     = new GlweCipher64(N);
    GlweCipher64* Cauto    = new GlweCipher64(N);

    // k = logn+1 .. logN
    for (int kk = logn + 1; kk <= logN; ++kk) {
        // C?_low = MS(C')
        GlweModSwitchDown(Cbar_low, out, /*ell=*/1, env);
        // C? = MR(C?_low)
        GlweModRaiseUp   (Cbar,     Cbar_low, /*ell=*/1, env);

        // ε: 0-based
        int idx = kk - (logn + 1);           // 0 .. (logN-logn-1)
        AutoKsKey64* aks = aks_list[idx];    // d = 2^kk + 1  شϴ Ű

        // Cauto = EvalAuto(C?, d_kk)
        EvalAuto(Cauto, Cbar, aks, env);

        // C' = C? + Cauto
        for (int q = 0; q <= k; ++q)
            for (int j = 0; j < N; ++j)
                out->a[q].coefs[j] = Cbar->a[q].coefs[j] + Cauto->a[q].coefs[j];

        //for (int j = 0; j < N; ++j)
        //    out->b->coefs[j] = Cbar->b->coefs[j] + Cauto->b->coefs[j];
    }

    delete Cbar_low;
    delete Cbar;
    delete Cauto;
}