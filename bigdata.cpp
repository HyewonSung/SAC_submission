#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>
#include <sys/time.h>
#include "generic_utils.h"
#include "spqlios/lagrangehalfc_impl.h"
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstring>
#include "64header.h"
#include "poc_64types.h"

Random* global_random = new Random();


static inline std::string pretty_bytes(uint64_t bytes) {
    const double KiB = 1024.0;
    const double MiB = KiB * 1024.0;
    const double GiB = MiB * 1024.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (bytes >= (uint64_t)GiB) {
        oss << (bytes / GiB) << " GiB";
    } else if (bytes >= (uint64_t)MiB) {
        oss << (bytes / MiB) << " MiB";
    } else if (bytes >= (uint64_t)KiB) {
        oss << (bytes / KiB) << " KiB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

static inline void print_preprocess_memory_model(
    uint64_t bytes_DB_rows,
    uint64_t bytes_R_cols,
    uint64_t bytes_R_colsFFT,
    uint64_t bytes_enc_dataa,
    uint64_t bytes_enc_temp,
    uint64_t bytes_enc_data) {
    const uint64_t subtotal_db = bytes_DB_rows + bytes_R_cols + bytes_R_colsFFT;
    const uint64_t subtotal_scratch = bytes_enc_dataa + bytes_enc_temp + bytes_enc_data;
    std::cerr << "\n[PREPROCESS / MEMORY ESTIMATE]\n";
    std::cerr << "  - DB_rows (Torus64)           : " << pretty_bytes(bytes_DB_rows) << "\n";
    std::cerr << "  - R_cols (coef domain)        : " << pretty_bytes(bytes_R_cols) << "  (model)\n";
    std::cerr << "  - R_colsFFT (FFT domain)      : " << pretty_bytes(bytes_R_colsFFT) << "  (model)\n";
    std::cerr << "  - subtotal (db preprocess)    : " << pretty_bytes(subtotal_db) << "  (model)\n";
    std::cerr << "  - enc_dataa (scratch, coef)   : " << pretty_bytes(bytes_enc_dataa) << "  (model)\n";
    std::cerr << "  - enc_temp (scratch, FFT)     : " << pretty_bytes(bytes_enc_temp) << "  (model)\n";
    std::cerr << "  - enc_data (scratch, FFT)     : " << pretty_bytes(bytes_enc_data) << "  (model)\n";
    std::cerr << "  - subtotal + scratch          : " << pretty_bytes(subtotal_db + subtotal_scratch) << "  (model)\n";
}






static inline void torusPolyClear(Torus64Polynomial* p, int N) {
    std::memset(p->coefs, 0, sizeof(Torus64) * N);
}
static inline void torusPolyCopy(Torus64Polynomial* dst, const Torus64Polynomial* src, int N) {
    std::memcpy(dst->coefs, src->coefs, sizeof(Torus64) * N);
}
static inline void torusPolySubTo(Torus64Polynomial* dst, const Torus64Polynomial* src, int N) {
    for (int i = 0; i < N; i++) dst->coefs[i] -= src->coefs[i];
}

static inline void lagrangeCopy(LagrangeHalfCPolynomiala* dst, const LagrangeHalfCPolynomiala* src, int N) {
    std::memcpy(dst->values, src->values, sizeof(double) * (size_t)N);
}

static inline void tlweFFTClear(TLweSampleFFTa* ct, int N, const Globals* env) {
    (void)N;
    
    LagrangeHalfCPolynomialClear_lvl2(&ct->a[0], env);
    LagrangeHalfCPolynomialClear_lvl2(ct->b, env);
}

static inline void tlweFFTCopy(TLweSampleFFTa* dst, const TLweSampleFFTa* src, int N) {
    lagrangeCopy(&dst->a[0], &src->a[0], N);
    lagrangeCopy(dst->b, src->b, N);
}










static inline uint64_t torus_to_uint_round(Torus64 t, int MM1) {
    const int shift = 64 - MM1;
    if (shift == 0) return (uint64_t)t; 
    const Torus64 half = (Torus64(1) << (shift - 1));
    const uint64_t mask = (MM1 == 64) ? ~UINT64_C(0) : ((UINT64_C(1) << MM1) - 1);
    return (uint64_t)((t + half) >> shift) & mask;
}
static inline int64_t torus_to_int_center_round(Torus64 t, int MM1) {
    uint64_t u = torus_to_uint_round(t, MM1);
    const uint64_t p = (MM1 == 64) ? ~UINT64_C(0) : (UINT64_C(1) << MM1);
    const uint64_t half = (MM1 == 0) ? 0 : (p >> 1);
    int64_t s = (int64_t)u;
    if (MM1 != 64 && u >= half) s -= (int64_t)p;
    return s;
}
static inline Torus64 uint_to_torus_center(uint64_t u, int MM1) {
    const uint64_t p = (MM1 == 64) ? ~UINT64_C(0) : (UINT64_C(1) << MM1);
    const uint64_t half = (MM1 == 0) ? 0 : (p >> 1);
    int64_t s = (int64_t)u;
    if (MM1 != 64 && u >= half) s -= (int64_t)p;
    const int shift = 64 - MM1;
    return (Torus64)(s << shift);
}



static inline void kappa_embed(IntPolynomiala* out, const IntPolynomiala* s0, int N_big, int N0) {
    const int k = N_big / N0;
    for (int i = 0; i < N_big; i++) out->coefs[i] = 0;
    for (int j = 0; j < N0; j++) out->coefs[j * k] = s0->coefs[j];
}


static void tlwePhaseSmall(Torus64* phase, const TLweSample64* ct_small, const IntPolynomiala* s0, int N0) {
    
    for (int i = 0; i < N0; i++) phase[i] = ct_small->b->coefs[i];

    
    const Torus64* a = ct_small->a[0].coefs;
    for (int ai = 0; ai < N0; ai++) {
        const Torus64 av = a[ai];
        if (av == 0) continue;
        for (int sj = 0; sj < N0; sj++) {
            const int s = s0->coefs[sj];
            if (!s) continue;
            const int t = ai + sj;
            if (t < N0) phase[t] -= av;
            else        phase[t - N0] += av; 
        }
    }
}


static TLweSample64** build_ringswitch_key(const Globals* env,
                                          const IntPolynomiala* s_big,
                                          const IntPolynomiala* rep_s0,
                                          int l, int bgbit,
                                          double stdev) {
    const int N = env->N;
    TLweSample64** rswk = new TLweSample64*[l];

    for (int p = 0; p < l; p++) {
        rswk[p] = new TLweSample64(N);

        
        for (int j = 0; j < N; j++) rswk[p]->a[0].coefs[j] = random_int64();

        
        for (int j = 0; j < N; j++) rswk[p]->b->coefs[j] = (Torus64)random_gaussian64(0, stdev);

        
        torus64PolynomialMultAddKaratsuba_lvl2(rswk[p]->b, rep_s0, &rswk[p]->a[0], env);

        
        const int shift = 64 - (p + 1) * bgbit;
        const Torus64 Lp = (shift >= 0 && shift < 64) ? (Torus64)(UINT64_C(1) << shift) : 0;
        for (int j = 0; j < N; j++) {
            if (s_big->coefs[j]) rswk[p]->b->coefs[j] += Lp;
        }
    }
    return rswk;
}




static void ringSwitch_N_to_N0(TLweSample64* out_small,
                              const TLweSample64* in_big,
                              TLweSample64** rswk,
                              int N0,
                              const Globals* env) {
    const int N = env->N;
    const int k = N / N0;

    
    IntPolynomiala* adec = new_array1<IntPolynomiala>(env->l, N);
    tGswTorus64PolynomialDecompH(adec, &in_big->a[0], env);

    
    Torus64Polynomial a_prime(N);
    Torus64Polynomial b_prime(N);
    torusPolyClear(&a_prime, N);
    torusPolyCopy(&b_prime, in_big->b, N);

    Torus64Polynomial tmp(N);

    for (int p = 0; p < env->l; p++) {
        torus64PolynomialMultKaratsuba_lvl2(&tmp, &adec[p], &rswk[p]->a[0], env);
        torusPolySubTo(&a_prime, &tmp, N);

        torus64PolynomialMultKaratsuba_lvl2(&tmp, &adec[p], rswk[p]->b, env);
        torusPolySubTo(&b_prime, &tmp, N);
    }

    
    for (int i = 0; i < N0; i++) {
        out_small->a[0].coefs[i] = a_prime.coefs[i * k];
        out_small->b->coefs[i]   = b_prime.coefs[i * k];
    }

    delete_array1(adec);
}



const int Globals::k = 1;
const int Globals::N = 2048;
const int Globals::t = 12;
const int Globals::smalln = N;
const int Globals::bgbit = 9;
const int Globals::l = 6;
const int Globals::basebit = 4;

using namespace std;

Globals::Globals() {
    
    torusDecompOffset = 0;
    for (int i = 0; i <= l; ++i) torusDecompOffset |= (UINT64_C(1) << (63 - i * bgbit));

    
    torusDecompBuf = new uint64_t[N];

    
    lwekey = new int[N];
    for (int i = 0; i < N; ++i) lwekey[i] = random_bit();

    tlwekey = new IntPolynomiala(N);
    for (int i = 0; i < N; ++i) tlwekey->coefs[i] = lwekey[i];

    
    in_key = new int64_t[N + 1];
    for (int i = 0; i < N; ++i) in_key[i] = lwekey[i];
    in_key[N] = -1;
}

int64_t array_max(Torus64Polynomial* a, int size) {
    int64_t max = llabs(a->coefs[0]);
    for (int i = 1; i < size; i++) {
        int64_t temp = llabs(a->coefs[i]);
        if (temp > max) max = temp;
    }
    return max;
}




static inline size_t bytes_TLweSample64(int k, int N) {
    return (size_t)(k + 1) * (size_t)N * sizeof(int64_t); 
}


static inline size_t bytes_TGswSample64(int k, int l, int N) {
    return (size_t)(k + 1) * (size_t)l * bytes_TLweSample64(k, N);
}


static inline size_t bytes_query_cipher(int k, int l, int N) {
    return (size_t)l * bytes_TLweSample64(k, N);
}


static inline size_t bytes_AutoKsKey64(int k, int l, int N) {
    return (size_t)l * bytes_TLweSample64(k, N);
}

static inline double MB10(size_t bytes) { return (double)bytes / 1e6; } 


using Clock = std::chrono::high_resolution_clock;
static inline double ms_between(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static inline void print_ms(const char* label, const Clock::time_point& a, const Clock::time_point& b) {
    const double ms = ms_between(a, b);
    std::cout << "[TIME] " << std::left << std::setw(38) << label
              << " : " << std::right << std::fixed << std::setprecision(3) << ms << " ms"
              << " (" << std::setprecision(3) << (ms / 1000.0) << " s)\n";
}

static inline void print_ms_val(const char* label, double ms) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[TIME] " << std::left << std::setw(38) << label
              << " : " << ms << " ms (" << (ms/1000.0) << " s)" << std::endl;
}

static inline bool is_power_of_two_u64(uint64_t x) {
    return x && ((x & (x - 1)) == 0);
}
static inline int log2_u64_exact(uint64_t x) {
    int r = 0;
    while (x > 1) { x >>= 1; r++; }
    return r;
}






static void torusPolyMulByXNegJ(Torus64* out, const Torus64* in, int N, int j) {
    j %= N;
    if (j < 0) j += N;
    if (j == 0) {
        std::memcpy(out, in, sizeof(Torus64) * N);
        return;
    }
    for (int i = 0; i < N; i++) {
        const int src = i + j;
        if (src < N) out[i] = in[src];
        else         out[i] = -in[src - N];
    }
}

static void tlweMulByXNegJ(TLweSample64* out, const TLweSample64* in, int N, int k, int j) {
    
    for (int u = 0; u < k; u++) {
        torusPolyMulByXNegJ(out->a[u].coefs, in->a[u].coefs, N, j);
    }
    
    torusPolyMulByXNegJ(out->b->coefs, in->b->coefs, N, j);
    
}




int main(int argc, char** argv) {
    
    bool verbose = false; 
    const bool profile = true;
    const auto t_program_start = Clock::now();

    
    size_t summary_query_bytes     = 0;   
    size_t summary_convk_bytes     = 0;
    size_t summary_aks_bytes       = 0;
    size_t summary_rswk_bytes      = 0;
    size_t summary_evalkey_bytes   = 0;

    
    double summary_T_unpack_ms     = 0.0;
    double summary_T_main_ms       = 0.0;
    double summary_T_total_ms      = 0.0;

    double summary_T_total_no_pre_ms = 0.0; 
    
    
    
    
    
    
    double summary_throughput_MBps = 0.0;
    double summary_effective_throughput_MBps = 0.0;
    double summary_record_MBps = 0.0;
    double summary_payload_MBps = 0.0;

    
    size_t summary_resp_before_bytes = 0; 
    size_t summary_resp_after_bytes  = 0; 

    
    double summary_output_noise_budget = 0.0;
    bool   summary_decryption_failure  = false;

    
    int    summary_N = 0;
    int    summary_logN = 0;
    int    summary_MM1 = 0;
    uint64_t summary_MM = 0;
    double  summary_alpha = 0.0;
    int     summary_log2m = 0; 
uint64_t summary_n = 0;          
uint64_t summary_pack = 0;
uint64_t summary_m = 0;          
uint64_t summary_cols = 0;       
uint64_t summary_D1 = 0;
uint64_t summary_mm = 0;



    
    
    
    
    
    
    
    
    
    
    const bool FULL_PACKING_MODE = true;
    const bool RUN_A1 = true;
    const bool RUN_A2 = true;

    double alpha1 = pow(2., -50);
    summary_alpha = alpha1;

    uint64_t n    = 1ull << 20; 
    uint64_t pack = 1;
    uint64_t m    = n / pack;




const uint64_t record_bytes_assumed = 256;




int MM1 = (argc >= 4 ? std::stoi(argv[3]) : 16);

if (MM1 <= 0 || MM1 >= 63 || (MM1 % 8) != 0) {
    std::cerr << "[ERROR] MM1 must be in {8,16,24,32,40,48,56} (got " << MM1 << ").\n";
    return 1;
}



const int torus_shift = 64 - MM1;


const uint64_t plain_mod  = (UINT64_C(1) << MM1);
const uint64_t plain_half = (plain_mod >> 1);
const uint64_t plain_mask = (plain_mod - 1);

const uint64_t coeff_bytes = (uint64_t)MM1 / 8;
if (record_bytes_assumed % coeff_bytes != 0) {
    std::cerr << "[ERROR] record_bytes_assumed (" << record_bytes_assumed
              << ") must be divisible by (MM1/8)=" << coeff_bytes << ".\n";
    return 1;
}



const uint64_t N1_target = record_bytes_assumed / coeff_bytes;



if (((uint64_t)Globals::N % N1_target) != 0) {
    std::cerr << "[ERROR] Globals::N=" << Globals::N
              << " is not divisible by N1_target=" << N1_target
              << " (record_bytes_assumed=" << record_bytes_assumed << ", MM1=" << MM1 << ").\n";
    return 1;
}

const uint64_t D1_default = (uint64_t)Globals::N / N1_target;


uint64_t D1   = (argc >= 2 ? (uint64_t)std::stoull(argv[1]) : D1_default);


uint64_t cols = n / D1;



uint64_t mm   = (argc >= 3 ? (uint64_t)std::stoull(argv[2]) : (1ull << 10));

summary_n    = n;
summary_pack = pack;
summary_m    = m;
summary_D1   = D1;
summary_cols = cols;
summary_mm   = mm;


    
    
    
    
    int index = (argc >= 5 ? std::stoi(argv[4]) : (rand() % (int)m));
    int index_r = (int)(index / (int)cols);
    int index_c = (int)(index % (int)cols);

std::cout << "[TARGET] index=" << index
          << " -> (r,c)=(" << index_r << ", " << index_c << ")\n";
    const auto t_env0 = Clock::now();
    Globals* env = new Globals();
    const auto t_env1 = Clock::now();
    if (profile) print_ms("Init Globals()", t_env0, t_env1);


    int32_t k = env->k;
    int32_t N = env->N;
    int32_t l = env->l;
    int Bgbit = env->bgbit;

    summary_N = N;
    summary_logN = (int)std::log2((double)N);

    
    Torus32 log2m = (Torus32)std::ceil(std::log2((double)cols));
    summary_log2m = (int)log2m;

    
    Torus32 log2mm = (Torus32)std::ceil(std::log2((double)mm));

    
    uint64_t MM = plain_mod;
    uint64_t logt = (uint64_t)std::ceil(std::log2((double)MM));
    summary_MM1 = MM1;
    summary_MM  = MM;

    
const int log2m_i  = (int)log2m;
const int log2mm_i = (int)log2mm;

const uint64_t mm_pow2 = 1ULL << log2mm_i;
if (mm != mm_pow2) {
    std::cerr << "[WARN] mm is not a power of two. Overriding mm=" << mm
              << " -> " << mm_pow2 << " (2^log2mm)." << std::endl;
    mm = mm_pow2;
}

const uint64_t tot_blocks = (cols + mm - 1) / mm; 
const int log2blocks = log2m_i - log2mm_i;
const uint64_t tot_padded = (log2blocks <= 0) ? 1ULL : (1ULL << log2blocks);

if (tot_padded < tot_blocks) {
    std::cerr << "[ERROR] tot_padded < tot_blocks. Check (cols, mm) sizing." << std::endl;
    return 1;
}

const uint64_t mct = std::max<uint64_t>(mm, tot_padded);

TLweSample64* enc_dataa = new TLweSample64(N);
TLweSampleFFTa* enc_temp = new_array1<TLweSampleFFTa>(mct, N);
TLweSampleFFTa* enc_data = new_array1<TLweSampleFFTa>(mct, N);

    
    
    
    {
        const uint64_t k_tlwe = (uint64_t)env->k; 

        
        
        const uint64_t N1_model = (uint64_t)N / (uint64_t)D1;
        const uint64_t bytes_DB_rows = (uint64_t)D1 * (uint64_t)cols * N1_model * (uint64_t)sizeof(Torus64);

        
        
        const uint64_t bytes_R_cols = (uint64_t)cols * (uint64_t)N * (uint64_t)sizeof(int32_t);

        
        
        const uint64_t bytes_R_colsFFT = (uint64_t)cols * (uint64_t)N * (uint64_t)sizeof(double);

        const uint64_t subtotal_preprocess = bytes_DB_rows + bytes_R_cols + bytes_R_colsFFT;

        
        
        const uint64_t bytes_tlwe64 = (k_tlwe + 1) * (uint64_t)N * (uint64_t)sizeof(Torus64);
        
        const uint64_t bytes_tlweFFT = (k_tlwe + 1) * (uint64_t)N * (uint64_t)sizeof(double);
        const uint64_t bytes_enc_temp = (uint64_t)mct * bytes_tlweFFT;
        const uint64_t bytes_enc_data = (uint64_t)mct * bytes_tlweFFT;
        const uint64_t bytes_enc_dataa = bytes_tlwe64;

        const uint64_t subtotal_scratch = bytes_enc_temp + bytes_enc_data + bytes_enc_dataa;

        std::cerr << "\n[PREPROCESS / MEMORY ESTIMATE]\n";
        std::cerr << "  - DB_rows (Torus64)           : " << pretty_bytes(bytes_DB_rows) << "\n";
        std::cerr << "  - R_cols (IntPoly)            : " << pretty_bytes(bytes_R_cols) << "  (model)\n";
        std::cerr << "  - R_colsFFT (FFT, double)     : " << pretty_bytes(bytes_R_colsFFT) << "  (model)\n";
        std::cerr << "  - subtotal (db preprocess)    : " << pretty_bytes(subtotal_preprocess) << "  (model)\n";
        std::cerr << "  - enc_temp (scratch, FFT)     : " << pretty_bytes(bytes_enc_temp) << "  (model)\n";
        std::cerr << "  - enc_data (scratch, FFT)     : " << pretty_bytes(bytes_enc_data) << "  (model)\n";
        std::cerr << "  - enc_dataa (scratch, coef)   : " << pretty_bytes(bytes_enc_dataa) << "  (model)\n";
        std::cerr << "  - subtotal + scratch          : " << pretty_bytes(subtotal_preprocess + subtotal_scratch) << "  (model)\n";
    }

    
    
    
    
    
    
    if ((uint64_t)N % D1 != 0) {
        std::cerr << "[ERROR] Need D1 | N for polynomial-record packing. "
                  << "N=" << N << ", D1=" << D1 << std::endl;
        return 1;
    }
    const uint64_t N1 = (uint64_t)N / D1; 

    
    
    const auto t_db0 = Clock::now();
    Torus64Polynomial** DB_rows = new Torus64Polynomial*[D1];
    for (uint64_t rr = 0; rr < D1; ++rr) {
        DB_rows[rr] = new Torus64Polynomial(cols * N1);

for (uint64_t cc = 0; cc < cols; ++cc) {
            for (uint64_t i = 0; i < N1; ++i) {
                
                
                const uint64_t u = ((uint64_t)random_int64()) & plain_mask; 
                DB_rows[rr]->coefs[cc * N1 + i] = uint_to_torus_center(u, MM1);
            }
        }    }


    const auto t_db1 = Clock::now();
    if (profile) print_ms("Build DB_rows (u8->torus)", t_db0, t_db1);

    
    const uint64_t showK = std::min<uint64_t>(8, N1);


const uint64_t record_bytes_effective_dbg = N1 * coeff_bytes;
const uint64_t showBytes = std::min<uint64_t>(8, record_bytes_effective_dbg);
std::cout << "[DB] expected record bytes (first " << showBytes << "): ";
uint64_t printed = 0;
for (uint64_t i = 0; i < N1 && printed < showBytes; ++i) {
    const Torus64 tor = DB_rows[index_r]->coefs[(uint64_t)index_c * N1 + i];
    const uint64_t u = torus_to_uint_round(tor, MM1);
    
    for (uint64_t b = 0; b < coeff_bytes && printed < showBytes; ++b) {
        const uint8_t byte = (uint8_t)((u >> (8 * b)) & 0xFF);
        std::cout << (int)byte << " ";
        printed++;
    }
}
std::cout << "\n"; 
    const auto t_rcols0 = Clock::now();
    IntPolynomiala** R_cols = new IntPolynomiala*[cols];
    for (uint64_t c = 0; c < cols; ++c) {
        R_cols[c] = new IntPolynomiala(N);
        for (int j = 0; j < N; ++j) R_cols[c]->coefs[j] = 0;


for (uint64_t r = 0; r < D1; ++r) {
            for (uint64_t i = 0; i < N1; ++i) {
                const Torus64 tor = DB_rows[r]->coefs[c * N1 + i];
                const uint64_t u = torus_to_uint_round(tor, MM1); 
                int64_t msg = (int64_t)u;
                if (MM1 != 64 && u >= plain_half) msg -= (int64_t)plain_mod; 
                const uint64_t idx = r + D1 * i; 
                R_cols[c]->coefs[idx] = (int)msg;
            }
        }    }
    const auto t_rcols1 = Clock::now();
    if (profile) print_ms("Build R_cols (interleaving)", t_rcols0, t_rcols1);





const auto t_pre0 = Clock::now();
LagrangeHalfCPolynomiala* R_colsFFT = new_array1<LagrangeHalfCPolynomiala>(cols, N);
for (uint64_t c = 0; c < cols; ++c) {
    IntPolynomial_ifft_lvl2(&R_colsFFT[c], R_cols[c], env);
}
const auto t_pre1 = Clock::now();
if (profile) print_ms("Preprocess DB: IFFT R_cols -> R_colsFFT", t_pre0, t_pre1);

    const auto t_online_start = Clock::now(); 





    
    
    
    
    int pos = FULL_PACKING_MODE ? 0 : (int)((N - index_r) % N);

    
    const auto t_gt0 = Clock::now();
    Torus64Polynomial* indexdata = new Torus64Polynomial(N);
    for (int t = 0; t < N; ++t) indexdata->coefs[t] = 0;

    for (uint64_t r = 0; r < D1; ++r) {
        for (uint64_t i = 0; i < N1; ++i) {
            const uint64_t base = r + D1 * i; 
            const int msg = (int)R_cols[index_c]->coefs[base];
            const int64_t tor = ((int64_t)msg) << torus_shift;

            
            int t = (int)base + pos;
            if (t < N) indexdata->coefs[t] = +tor;
            else { t -= N; indexdata->coefs[t] = -tor; }
        }
    }
    const auto t_gt1 = Clock::now();
    if (profile) print_ms("Build ground-truth packed poly", t_gt0, t_gt1);

    

    Torus64Polynomial* index_r_one_hot_poly = new Torus64Polynomial(N);
    for (int j = 0; j < N; ++j) index_r_one_hot_poly->coefs[j] = 0;
    
    index_r_one_hot_poly->coefs[pos] = +(int64_t(1) << torus_shift);
    const auto t_rowq0 = Clock::now();
    TLweSample64* index_r_one_hot_ct = new TLweSample64(N);
    tLwe64Encrypt(index_r_one_hot_ct, index_r_one_hot_poly, alpha1, env);
    const auto t_rowq1 = Clock::now();
    if (profile) print_ms("Encrypt row one-hot query", t_rowq0, t_rowq1);

    const auto t_colq0 = Clock::now();
    
    std::vector<int64_t> bit(log2m);
    int_to_bin_digit(index_c, log2m, bit.data());

    
    Torus64Polynomial** message = new Torus64Polynomial*[l];
    for (int i = 0; i < l; i++) {
        message[i] = new Torus64Polynomial(N);
        for (int j = 0; j < N; j++) message[i]->coefs[j] = 0;
    }
    for (int i = 0; i < l; i++) {
        for (int32_t j = log2m - 1; j >= 0; --j) {
            message[i]->coefs[log2m - 1 - j] =
                bit[j] * (UINT64_C(1) << (64 - (i + 1) * Bgbit));
        }
    }
    
    

    
        const auto t_colq1 = Clock::now();
    if (profile) print_ms("Build column query plaintexts", t_colq0, t_colq1);

std::cout << "[Step 3] Encrypting messages into cipher (no 1/N scaling)..." << std::endl;
    const auto t_step3_0 = Clock::now();
    TLweSample64** cipher = new TLweSample64*[l];
    for (int i = 0; i < l; i++) {
        cipher[i] = new TLweSample64(N);
        tLwe64Encrypt_debug(cipher[i], message[i], alpha1, env);
    }

    
        const auto t_step3_1 = Clock::now();
    if (profile) print_ms("Step3: Encrypt column query ciphers", t_step3_0, t_step3_1);

std::cout << "[Step 4] Creating rotated cipher array..." << std::endl;
    const auto t_step4_0 = Clock::now();
    TLweSample64*** rot_cipher = new TLweSample64**[l];
    for (int i = 0; i < l; i++) {
        rot_cipher[i] = new TLweSample64*[log2m];

        rot_cipher[i][0] = new TLweSample64(N);
for (int q = 0; q <= k; ++q) {
    for (int u = 0; u < N; ++u) rot_cipher[i][0]->a[q].coefs[u] = cipher[i]->a[q].coefs[u];
}

        for (int j = 1; j < log2m; j++) {
            rot_cipher[i][j] = new TLweSample64(N);
            left_shift_by_one(rot_cipher[i][j], rot_cipher[i][j - 1], N);
        }
    }

    


    const auto t_step4_1 = Clock::now();
    if (profile) print_ms("Step4: Build rotated cipher array", t_step4_0, t_step4_1);


int n_trace = 1;  
    const auto t_aks0 = Clock::now();
    AutoKsKey64** aks_list = AutoKsKeyGenAll(env, n_trace);
    const auto t_aks1 = Clock::now();
    if (profile) print_ms("Step5: AutoKsKeyGenAll (aks_list)", t_aks0, t_aks1);

std::cout << "[Step 6] Applying RevHomTrace_Alg5 to build cipher_prime..." << std::endl;
    const auto t_step6_0 = Clock::now();

    TLweSample64*** cipher_prime = new TLweSample64**[l];

    for (int i = 0; i < l; i++) {
        cipher_prime[i] = new TLweSample64*[log2m];
        for (int j = 0; j < log2m; j++) {
            cipher_prime[i][j] = new TLweSample64(N);

            GlweCipher64* ct_in = new GlweCipher64(N);
            for (int q = 0; q <= k; ++q) {
                for (int t = 0; t < N; ++t)
                    ct_in->a[q].coefs[t] = rot_cipher[i][j]->a[q].coefs[t];
            }
            GlweCipher64* ct_out = new GlweCipher64(N);
            RevHomTrace_Alg5(ct_out, ct_in, aks_list, n_trace, env);

            for (int q = 0; q <= k; ++q) {
                for (int t = 0; t < N; ++t)
                    cipher_prime[i][j]->a[q].coefs[t] = ct_out->a[q].coefs[t];
            }
            delete ct_in;
            delete ct_out;
        }
    }

    
    const auto t_step6_1 = Clock::now();
    if (profile) print_ms("Step6: Build cipher_prime (RevHomTrace)", t_step6_0, t_step6_1);

summary_query_bytes =  bytes_TLweSample64(k, N); 

    
    IntPolynomiala* minus_sk = new IntPolynomiala(N);
    for (int i = 0; i < N; i++)
        minus_sk->coefs[i] = -env->tlwekey->coefs[i];

    TGswSample64* convk = new TGswSample64(l, N);

    
    
    const int N0 = N / D1;

    
    
    
    summary_resp_before_bytes = bytes_TLweSample64(k, N);
    summary_resp_after_bytes  = bytes_TLweSample64(k, N0);

    IntPolynomiala* sk0 = new IntPolynomiala(N0);   
    std::mt19937 rng(0xC0FFEE);
    for (int i = 0; i < N0; i++) sk0->coefs[i] = (int)(rng() & 1);
IntPolynomiala* rep_sk0 = new IntPolynomiala(N); 
    kappa_embed(rep_sk0, sk0, N, N0);

const double rswk_stdev = alpha1;
    const auto t_rswk0 = Clock::now();
TLweSample64** rswk = build_ringswitch_key(env, env->tlwekey, rep_sk0, l, env->bgbit, rswk_stdev);
const auto t_rswk1 = Clock::now();
if (profile) print_ms("Gen rswk (RingSwitch key)", t_rswk0, t_rswk1);
summary_rswk_bytes = (size_t)l * 2 * (size_t)N * sizeof(Torus64); 
    const auto t_convk0 = Clock::now();
tGsw64Encrypt_poly_2(convk, minus_sk, alpha1, env);
const auto t_convk1 = Clock::now();
if (profile) print_ms("Gen convk (tGsw encrypt)", t_convk0, t_convk1);

    const int aks_count = summary_logN;  
    summary_convk_bytes   = bytes_TGswSample64(k, l, N);
    summary_aks_bytes     = (size_t)aks_count * bytes_AutoKsKey64(k, l, N);
    summary_evalkey_bytes = summary_convk_bytes + summary_aks_bytes + summary_rswk_bytes;

    
    TGswSample64* extract = new_array1<TGswSample64>(log2m, l, N);
    TGswSampleFFTa* extFFT = new_array1<TGswSampleFFTa>(log2m, l, N);

    auto start_unpack = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < log2m; j++) {
        TLweSample64** rlweInputs = new TLweSample64*[l];
        for (int i = 0; i < l; i++) rlweInputs[i] = cipher_prime[i][j];

        unpacking_algorithm4(&extract[j], const_cast<const TLweSample64**>(rlweInputs), convk, env);
        delete[] rlweInputs;
    }
    auto end_unpack = std::chrono::high_resolution_clock::now();
    summary_T_unpack_ms = std::chrono::duration<double, std::milli>(end_unpack - start_unpack).count();
    if (profile) std::cout << "[TIME] Unpacking loop total           : " << std::fixed << std::setprecision(3) << summary_T_unpack_ms << " ms\n";

    
    auto start_main = std::chrono::high_resolution_clock::now();


const auto t_main_fr0 = Clock::now();




LagrangeHalfCPolynomiala row_a_fft(N);
LagrangeHalfCPolynomiala row_b_fft(N);
TorusPolynomial64_ifft_lvl2(&row_a_fft, &index_r_one_hot_ct->a[0], env);
TorusPolynomial64_ifft_lvl2(&row_b_fft, index_r_one_hot_ct->b, env);

const auto t_main_fr1 = Clock::now();
if (profile) print_ms("Main: IFFT row-query (a,b)", t_main_fr0, t_main_fr1);

int32_t kpl = (k + 1) * l;

    
    const auto t_main_ext0 = Clock::now();
    for (int s = 0; s < log2m; s++) {
        for (int i = 0; i < kpl; i++)
            for (int q = 0; q <= k; q++)
                TorusPolynomial64_ifft_lvl2(&extFFT[s].allsamples[i].a[q],
                                            &extract[s].allsamples[i].a[q], env);
    }

    const auto t_main_ext1 = Clock::now();
    if (profile) print_ms("Main: IFFT extract -> extFFT", t_main_ext0, t_main_ext1);





if (cols % mm != 0) {
    std::cerr << "[ERROR] Need cols divisible by mm for current CMux reduction. "
              << "cols=" << cols << ", mm=" << mm << std::endl;
    return 1;
}

const auto t_main_tree0 = Clock::now();



double t_main_1d_ms = 0.0;
double t_main_cmux_ms = 0.0;

const auto t_main_cmux0 = Clock::now();


#ifdef _OPENMP
#pragma omp parallel
#endif
{
    TLweSampleFFTa* tree_nodes_block = new_array1<TLweSampleFFTa>(mm, N);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (uint64_t z = 0; z < tot_padded; ++z) {
        
        const auto t_1d0 = Clock::now();
        for (uint64_t jj = 0; jj < mm; ++jj) {
            const uint64_t cidx = z * mm + jj;
            if (cidx < cols) {
                
                LagrangeHalfCPolynomialClear_lvl2(&tree_nodes_block[jj].a[0], env);
                LagrangeHalfCPolynomialAddMul_lvl2(&tree_nodes_block[jj].a[0], &row_a_fft, &R_colsFFT[cidx], env);

                LagrangeHalfCPolynomialClear_lvl2(tree_nodes_block[jj].b, env);
                LagrangeHalfCPolynomialAddMul_lvl2(tree_nodes_block[jj].b, &row_b_fft, &R_colsFFT[cidx], env);
            } else {
                
                LagrangeHalfCPolynomialClear_lvl2(&tree_nodes_block[jj].a[0], env);
                LagrangeHalfCPolynomialClear_lvl2(tree_nodes_block[jj].b, env);
            }
        }
        const auto t_1d1 = Clock::now();
        #ifdef _OPENMP
    #pragma omp atomic
    #endif
                t_main_1d_ms += ms_between(t_1d0, t_1d1);

        
        const auto t_cm0 = Clock::now();
        uint64_t cur_sz = mm;
        for (int b = 0; b < (int)log2mm; ++b) {
            const uint64_t next_sz = cur_sz >> 1;
            for (uint64_t i = 0; i < next_sz; ++i) {
                CMuxFFTa(&tree_nodes_block[i], &extFFT[b], &tree_nodes_block[2 * i], &tree_nodes_block[2 * i + 1], env);
            }
            cur_sz = next_sz;
        }

        
        tlweFFTCopy(&enc_temp[z], &tree_nodes_block[0], N);

        const auto t_cm1 = Clock::now();
        #ifdef _OPENMP
    #pragma omp atomic
    #endif
                t_main_cmux_ms += ms_between(t_cm0, t_cm1);
    }


    delete_array1(tree_nodes_block);
}

const auto t_cmg0 = Clock::now();
uint64_t cur_sz = tot_padded;
for (int b = (int)log2mm; b < (int)log2m; ++b) {
    const uint64_t next_sz = cur_sz >> 1;
    for (uint64_t i = 0; i < next_sz; ++i) {
        CMuxFFTa(&enc_temp[i], &extFFT[b], &enc_temp[2 * i], &enc_temp[2 * i + 1], env);
    }
    cur_sz = next_sz;
}
const auto t_cmg1 = Clock::now();
t_main_cmux_ms += ms_between(t_cmg0, t_cmg1);

const auto t_main_cmux1 = Clock::now();
    if (profile) {
        print_ms_val("Main: 1D build (FFT mul/add)", t_main_1d_ms);
        print_ms_val("Main: CMux total (FFT domain)", t_main_cmux_ms);
        print_ms("Main: Total (FFT domain)", t_main_cmux0, t_main_cmux1);
    }


    auto end_main = std::chrono::high_resolution_clock::now();
    summary_T_main_ms = std::chrono::duration<double, std::milli>(end_main - start_main).count();

    summary_T_total_ms = summary_T_unpack_ms + summary_T_main_ms;

    
    
    
    const double T_sec = summary_T_total_ms / 1000.0;
    const double payload_bytes_per_query = (double)summary_MM1 / 8.0; 

    const uint64_t record_bytes_effective = N1 * coeff_bytes;  
    

    summary_payload_MBps = (payload_bytes_per_query / T_sec) / 1e6;
    summary_record_MBps  = ((double)record_bytes_effective / T_sec) / 1e6;

    const double assumed_DB_bytes  = (double)summary_n * (double)record_bytes_assumed;
    const double effective_DB_bytes = (double)summary_n * (double)record_bytes_effective;
    summary_throughput_MBps          = (assumed_DB_bytes / T_sec) / 1e6;
    summary_effective_throughput_MBps = (effective_DB_bytes / T_sec) / 1e6;

    
    
    
    const auto t_srv_out0 = Clock::now();
    for (int q = 0; q <= k; ++q) TorusPolynomial64_fft_lvl2(&enc_dataa->a[q], &enc_temp[0].a[q], env);
    TorusPolynomial64_fft_lvl2(enc_dataa->b, enc_temp[0].b, env);
    const auto t_srv_out1 = Clock::now();
    if (profile) print_ms("Server: Final IFFT (FFT->coef) for response", t_srv_out0, t_srv_out1);

    
    const auto t_dec0 = Clock::now();
    Torus64Polynomial* decrypt = new Torus64Polynomial(N);
    tLwe64Phase_lvl2(decrypt, enc_dataa, env);
    const auto t_dec1 = Clock::now();
    if (profile) print_ms("Client: Decrypt packed (sanity)", t_dec0, t_dec1);

    
    const uint64_t showK2 = std::min<uint64_t>(8, N1);
        {
    const uint64_t showBytes_packed = std::min<uint64_t>(8, N1 * coeff_bytes);
    std::cout << "[Packed] coset0 bytes (stride D1), first " << showBytes_packed << ": ";
    uint64_t printed_p = 0;
    for (uint64_t i = 0; i < N1 && printed_p < showBytes_packed; ++i) {
        const uint64_t u = torus_to_uint_round(decrypt->coefs[i * D1], MM1);
        for (uint64_t b = 0; b < coeff_bytes && printed_p < showBytes_packed; ++b) {
            const uint8_t byte = (uint8_t)((u >> (8 * b)) & 0xFF);
            std::cout << (int)byte << " ";
            printed_p++;
        }
    }
}
    std::cout << "\n";

    
    
    
    
    
    std::cout << "\n==================== METHOD A-2 (server extracts w/o knowing j) ====================\n";

    
    
    
    const int logD1 = (int)std::llround(std::log2((double)D1));
    if (((uint64_t)1 << logD1) != D1) {
        std::cerr << "[ERROR] A-2 requires D1 to be a power of two. D1=" << D1 << std::endl;
        return 1;
    }
    std::vector<TGswSample64*> j_ctrl(logD1, nullptr);

    const auto t_a2_q0 = Clock::now();

    
    std::vector<int64_t> bit_j(logD1, 0);
    int_to_bin_digit((uint32_t)index_r, logD1, bit_j.data());

    Torus64Polynomial** msg_j = new Torus64Polynomial*[l];
    TLweSample64** cipher_j = new TLweSample64*[l];
    TLweSample64*** rot_cipher_j = new TLweSample64**[l];
    TLweSample64*** cipher_prime_j = new TLweSample64**[l];

    for (int i = 0; i < l; i++) {
        
        
        
        
        msg_j[i] = new Torus64Polynomial(N);
        for (int u = 0; u < N; u++) msg_j[i]->coefs[u] = 0;

        const int shift = 64 - (i + 1) * env->bgbit; 
        const Torus64 Lp = (shift >= 0 && shift <= 63) ? (Torus64)(UINT64_C(1) << shift) : 0;
        for (int b = 0; b < logD1; b++) {
            const int pos = logD1 - 1 - b;
            const int64_t bj = bit_j[b]; 
            msg_j[i]->coefs[pos] = (bj ? Lp : 0);
        }

        cipher_j[i] = new TLweSample64(N);
        
        tLwe64Encrypt_debug(cipher_j[i], msg_j[i], alpha1, env);

        
        rot_cipher_j[i] = new TLweSample64*[logD1];
        cipher_prime_j[i] = new TLweSample64*[logD1];
        rot_cipher_j[i][0] = new TLweSample64(N);
for (int q = 0; q <= k; ++q) {
    for (int u = 0; u < N; ++u) rot_cipher_j[i][0]->a[q].coefs[u] = cipher_j[i]->a[q].coefs[u];
}
        for (int t = 1; t < logD1; t++) {
            rot_cipher_j[i][t] = new TLweSample64(N);
            left_shift_by_one(rot_cipher_j[i][t], rot_cipher_j[i][t - 1], N);
        }

        
        for (int t = 0; t < logD1; t++) {
            cipher_prime_j[i][t] = new TLweSample64(N);
            
            GlweCipher64* ct_in = new GlweCipher64(N);
            GlweCipher64* ct_out = new GlweCipher64(N);
            for (int u = 0; u < N; u++) {
                ct_in->a[0].coefs[u] = rot_cipher_j[i][t]->a[0].coefs[u];
                ct_in->b->coefs[u]   = rot_cipher_j[i][t]->b->coefs[u];
            }
            
            
            RevHomTrace_Alg5(ct_out, ct_in, aks_list, n_trace, env);
            for (int u = 0; u < N; u++) {
                cipher_prime_j[i][t]->a[0].coefs[u] = ct_out->a[0].coefs[u];
                cipher_prime_j[i][t]->b->coefs[u]   = ct_out->b->coefs[u];
            }
            delete ct_in;
            delete ct_out;
        }
    }

    const auto t_a2_q1 = Clock::now();
    if (profile) print_ms("A-2 Client: Build j query (cipher_prime)", t_a2_q0, t_a2_q1);

    
    const auto t_a2_u0 = Clock::now();
    for (int b = 0; b < logD1; b++) {
        j_ctrl[b] = new TGswSample64(l, N);
        std::vector<const TLweSample64*> tmp_list(l);
        for (int i = 0; i < l; i++) tmp_list[i] = cipher_prime_j[i][b];
        unpacking_algorithm4(j_ctrl[b], tmp_list.data(), convk, env);
    }
    const auto t_a2_u1 = Clock::now();
    if (profile) print_ms("A-2 Server: Unpack j query -> RGSW bits", t_a2_u0, t_a2_u1);


    
const auto t_a2_srv0 = Clock::now();


const auto t_a2_rot0 = Clock::now();
const int D1i = (int)D1;
std::vector<TLweSample64*> cand(D1i, nullptr);
for (int t = 0; t < D1i; t++) {
    cand[t] = new TLweSample64(N);
    
    tlweMulByXNegJ(cand[t], enc_dataa, N, k, t);
}
const auto t_a2_rot1 = Clock::now();
if (profile) print_ms("A-2 Server: build rotated cands", t_a2_rot0, t_a2_rot1);




const auto t_a2_sel0 = Clock::now();
std::vector<TLweSample64*> level = cand; 
for (int b = 0; b < logD1; b++) {
    const int pairs = (int)level.size() / 2;
    std::vector<TLweSample64*> next(pairs, nullptr);
    for (int i = 0; i < pairs; i++) {
        TLweSample64* c0 = level[2 * i];
        TLweSample64* c1 = level[2 * i + 1];
        next[i] = new TLweSample64(N);
        CMux(next[i], j_ctrl[b], c0, c1, env); 
        delete c0;
        delete c1;
    }
    level.swap(next);
}
TLweSample64* cur_a2 = level[0];
const auto t_a2_sel1 = Clock::now();
if (profile) print_ms("A-2 Server: CMux-select (bits)", t_a2_sel0, t_a2_sel1);


const auto t_a2_rs0 = Clock::now();
TLweSample64* outN0 = new TLweSample64(N0);
ringSwitch_N_to_N0(outN0, cur_a2, rswk, N0, env);
const auto t_a2_rs1 = Clock::now();
if (profile) print_ms("A-2 Server: RingSwitch N->N0", t_a2_rs0, t_a2_rs1);


delete cur_a2;

const auto t_a2_srv1 = Clock::now();
if (profile) print_ms("A-2 Server: Total (rot+sel+rs)", t_a2_srv0, t_a2_srv1);

const size_t resp_small_bytes = (size_t)(k + 1) * (size_t)N0 * sizeof(Torus64);
std::cout << "[A-2] Response bytes (R_{N0}) : " << resp_small_bytes << " bytes (" << MB10(resp_small_bytes) << " MB)\n";


Torus64* phase_a2 = new Torus64[N0];
const auto t_a2_ph0 = Clock::now();
tlwePhaseSmall(phase_a2, outN0, sk0, N0);
const auto t_a2_ph1 = Clock::now();
if (profile) print_ms("A-2 Client: Phase (small ring)", t_a2_ph0, t_a2_ph1);

std::cout << "[A-2] extracted record bytes, first " << std::min<uint64_t>(8, (uint64_t)N0) << ": ";
{
    const uint64_t showBytes_a2 = std::min<uint64_t>(8, (uint64_t)N0 * coeff_bytes);
    uint64_t printed_a2 = 0;
    for (uint64_t i = 0; i < (uint64_t)N0 && printed_a2 < showBytes_a2; ++i) {
        const uint64_t u = torus_to_uint_round(phase_a2[i], MM1);
        for (uint64_t b = 0; b < coeff_bytes && printed_a2 < showBytes_a2; ++b) {
            const uint8_t byte = (uint8_t)((u >> (8 * b)) & 0xFF);
            std::cout << (int)byte << " ";
            printed_a2++;
        }
    }
}
std::cout << "\n";


    int bad_a2 = 0;
    int aa_a2 = 0;
    Torus64 maxerr_a2 = 0;
    const uint64_t thr_a2 = (uint64_t)UINT64_C(1) << (63 - (int)logt);
    for (int i = 0; i < N0; i++) {
        const Torus64 expected = DB_rows[index_r]->coefs[index_c * N0 + i];
        const Torus64 diff = phase_a2[i] - expected;
        const Torus64 adiff = (diff < 0) ? (Torus64)(- (int64_t)diff) : diff;
        if ((uint64_t)adiff >= thr_a2) aa_a2++;
        if (adiff > maxerr_a2) maxerr_a2 = adiff;
        if (torus_to_uint_round(phase_a2[i], MM1) != torus_to_uint_round(expected, MM1)) bad_a2++;
    }
    std::cout << "[A-2] record bytes " << (bad_a2 ? "MISMATCH" : "OK")
              << " (bad=" << bad_a2 << ", maxerr=" << (uint64_t)maxerr_a2 << ")\n";



    
    
    
    double bit_ea_a2 = 0.0;
    if ((uint64_t)maxerr_a2 > 0) {
        bit_ea_a2 = std::ceil(std::log2((double)(uint64_t)maxerr_a2));
    }
    summary_output_noise_budget = 64.0 - (double)logt - bit_ea_a2 - 1.0;

    
    summary_decryption_failure = (bad_a2 > 0) || (aa_a2 > 0);

#if 0  

std::cout << "\n==================== APPENDIX: METHOD A-1 (client rotates + RingSwitch) ====================\n";
    const size_t resp_big_bytes = (size_t)(k + 1) * (size_t)N * sizeof(Torus64);
    std::cout << "[A-1] Response bytes (R_N) : " << resp_big_bytes << " bytes (" << MB10(resp_big_bytes) << " MB)\n";

    TLweSample64* ct_big_rot = new TLweSample64(N);
    const auto t_a1_rot0 = Clock::now();
    tlweMulByXNegJ(ct_big_rot, enc_dataa, N, k, (int)index_r);
    const auto t_a1_rot1 = Clock::now();
    if (profile) print_ms("A-1 Client: multiply by X^{-j}", t_a1_rot0, t_a1_rot1);

    TLweSample64* ct_a1_rec = new TLweSample64(N0);
    const auto t_a1_rs0 = Clock::now();
    ringSwitch_N_to_N0(ct_a1_rec, ct_big_rot, rswk, N0, env);
    const auto t_a1_rs1 = Clock::now();
    if (profile) print_ms("A-1 Client: RingSwitch N->N0", t_a1_rs0, t_a1_rs1);

    Torus64* phase_a1 = new Torus64[N0];
    const auto t_a1_ph0 = Clock::now();
    tlwePhaseSmall(phase_a1, ct_a1_rec, sk0, N0);
    const auto t_a1_ph1 = Clock::now();
    if (profile) print_ms("A-1 Client: Phase (small ring)", t_a1_ph0, t_a1_ph1);

    std::cout << "[A-1] extracted record bytes, first " << std::min<uint64_t>(8, (uint64_t)N0) << ": ";
    {
    const uint64_t showBytes_a1 = std::min<uint64_t>(8, (uint64_t)N0 * coeff_bytes);
    uint64_t printed_a1 = 0;
    for (uint64_t i = 0; i < (uint64_t)N0 && printed_a1 < showBytes_a1; ++i) {
        const uint64_t u = torus_to_uint_round(phase_a1[i], MM1);
        for (uint64_t b = 0; b < coeff_bytes && printed_a1 < showBytes_a1; ++b) {
            const uint8_t byte = (uint8_t)((u >> (8 * b)) & 0xFF);
            std::cout << (int)byte << " ";
            printed_a1++;
        }
    }
}
    std::cout << "\n";

    const auto t_a1_chk0 = Clock::now();
    int bad_a1 = 0;
    Torus64 maxerr_a1 = 0;
    for (int i = 0; i < N0; i++) {
        const Torus64 expected = DB_rows[index_r]->coefs[index_c * N0 + i];
        const Torus64 diff = phase_a1[i] - expected;
        const Torus64 adiff = (diff < 0) ? (Torus64)(- (int64_t)diff) : diff;
        if (adiff > maxerr_a1) maxerr_a1 = adiff;
        if (torus_to_uint_round(phase_a1[i], MM1) != torus_to_uint_round(expected, MM1)) bad_a1++;
    }
    const auto t_a1_chk1 = Clock::now();
    if (profile) print_ms("A-1 Client: Verify record bytes", t_a1_chk0, t_a1_chk1);
    std::cout << "[A-1] record bytes " << (bad_a1 ? "MISMATCH" : "OK")
              << " (bad=" << bad_a1 << ", maxerr=" << (uint64_t)maxerr_a1 << ")\n";

    
    
    
    
    
    


#endif  


    
    


    
    
    const auto t_end_no_pre = Clock::now();
    summary_T_total_no_pre_ms = ms_between(t_online_start, t_end_no_pre);

    const double T_no_pre_sec = summary_T_total_no_pre_ms / 1000.0;
    if (T_no_pre_sec > 0) {
        summary_payload_MBps = (payload_bytes_per_query / T_no_pre_sec) / 1e6;
        summary_record_MBps  = ((double)record_bytes_effective / T_no_pre_sec) / 1e6;
        summary_throughput_MBps           = (assumed_DB_bytes / T_no_pre_sec) / 1e6;
        summary_effective_throughput_MBps = (effective_DB_bytes / T_no_pre_sec) / 1e6;
    }


    std::cout << "\n==================== SUMMARY ====================\n";

    std::cout << "[PARAMETERS]\n";
    std::cout << "  - N (poly degree)         : " << summary_N << " (logN=" << summary_logN << ")\n";
    std::cout << "  - plaintext modulus p     : 2^" << summary_MM1 << " (p=" << summary_MM << ")\n";
    std::cout << "  - payload/query (scalar)  : " << payload_bytes_per_query << " bytes\n";
    std::cout << "  - record bytes (effective): " << record_bytes_effective << " bytes (N1=N/D1)\n";
    std::cout << "  - record bytes (assumed)  : " << record_bytes_assumed << " bytes (for DB-throughput reporting)\n";
std::cout << "  - DB #entries (n)         : ";
if (is_power_of_two_u64(summary_n)) {
    std::cout << "2^" << log2_u64_exact(summary_n) << " (" << summary_n << ")\n";
} else {
    std::cout << summary_n << "\n";
}
std::cout << "  - D1 (rows in split)      : " << summary_D1 << "\n";
std::cout << "  - cols = n/D1             : " << summary_cols << "\n";
std::cout << "  - l               : " << l << "\n";
std::cout << "  - bgbit                    : " << Bgbit << "\n";




    std::cout << "\n[OFFLINE / COMM]\n";
    std::cout << "  - Query size (cipher)     : "
              << summary_query_bytes << " bytes (" << MB10(summary_query_bytes) << " MB)\n";
    std::cout << "  - EvalKey size (convk+aks_list+rswk) : "
              << summary_evalkey_bytes << " bytes (" << MB10(summary_evalkey_bytes) << " MB)\n";
    std::cout << "      * convk     : " << summary_convk_bytes << " bytes (" << MB10(summary_convk_bytes) << " MB)\n";
    std::cout << "      * aks_list  : " << summary_aks_bytes   << " bytes (" << MB10(summary_aks_bytes)   << " MB)\n";
    std::cout << "      * rswk      : " << summary_rswk_bytes  << " bytes (" << MB10(summary_rswk_bytes)  << " MB)\n";
    std::cout << "      * aks_count : " << aks_count << "\n";
size_t summary_offline_comm_total_bytes = summary_query_bytes + summary_evalkey_bytes;

std::cout << "  - Offline comm total (query+evalkey) : "
          << summary_offline_comm_total_bytes << " bytes ("
          << MB10(summary_offline_comm_total_bytes) << " MB)\n";

    std::cout << "\n[ONLINE / COMM]\n";
    std::cout << "  - Response size before RingSwitch (R_N)   : "
              << summary_resp_before_bytes << " bytes (" << MB10(summary_resp_before_bytes) << " MB)\n";
    std::cout << "  - Response size after  RingSwitch (R_{N0}): "
              << summary_resp_after_bytes  << " bytes (" << MB10(summary_resp_after_bytes)  << " MB)\n";
    if (summary_resp_after_bytes > 0) {
        const double shrink = (double)summary_resp_before_bytes / (double)summary_resp_after_bytes;
        std::cout << "  - Response shrink factor                 : " << shrink << "x\n";
    }





    std::cout << "\n[ONLINE / COMP]\n";
    std::cout << "  - T_unpack_ms : " << summary_T_unpack_ms << " ms\n";
    std::cout << "  - T_main_ms   : " << summary_T_main_ms   << " ms\n";
    std::cout << "  - T_total_ms  : " << summary_T_total_ms  << " ms\n";
    std::cout << "  - T_total_no_pre_ms (end-to-end) : " << summary_T_total_no_pre_ms << " ms\n";
    std::cout << "  - Throughput (DB assumed n*256B, no-preprocess total) : " << summary_throughput_MBps << " MB/s (10^6 bytes)\n";
    std::cout << "  - Throughput (DB effective n*N1, no-preprocess total) : " << summary_effective_throughput_MBps << " MB/s (10^6 bytes)\n";
    std::cout << "  - Throughput (record retrieved)  : " << summary_record_MBps << " MB/s (10^6 bytes)\n";
    std::cout << "  - Throughput (scalar payload)    : " << summary_payload_MBps << " MB/s (10^6 bytes)\n";

    std::cout << "\n[OUTPUT]\n";
    std::cout << "  - output noise budget : " << summary_output_noise_budget << "\n";
    std::cout << "  - decryption failure? : " << (summary_decryption_failure ? "Yes" : "No") << "\n";

    std::cout << "=================================================\n";


    
    DeleteAutoKsKeyAll(aks_list, env, n_trace);

        if (profile) {
        print_ms("TOTAL PROGRAM (with preprocess)", t_program_start, t_end_no_pre);
        print_ms("TOTAL PROGRAM (no preprocess)", t_online_start, t_end_no_pre);
        print_ms("Preprocess total", t_program_start, t_online_start);
    }
    #if 0  

    {
        const double t_pre_db_ms   = ms_between(t_db0, t_db1);
        const double t_pre_rcols_ms= ms_between(t_rcols0, t_rcols1);
        const double t_pre_fft_ms  = ms_between(t_pre0, t_pre1);
        const double t_pre_total_ms= t_pre_db_ms + t_pre_rcols_ms + t_pre_fft_ms;

        const double t_total_with_pre_ms    = ms_between(t_program_start, t_program_end);
        const double t_total_wo_pre_ms      = ms_between(t_online_start, t_program_end);

        const double t_revhomtrace_ms = ms_between(t_step6_0, t_step6_1);
        const double t_rlwe_to_rgsw_ms = summary_T_unpack_ms;

        const double t_main_total_ms = summary_T_main_ms;

        const double t_a1_rot_ms  = ms_between(t_a1_rot0, t_a1_rot1);
        const double t_a1_rsw_ms  = ms_between(t_a1_rs0, t_a1_rs1);
        const double t_a1_phase_ms= ms_between(t_a1_ph0, t_a1_ph1);

        const double t_a2_buildj_ms = ms_between(t_a2_q0, t_a2_q1);
        const double t_a2_unpj_ms   = ms_between(t_a2_u0, t_a2_u1);
        const double t_a2_rot_ms    = ms_between(t_a2_rot0, t_a2_rot1);
        const double t_a2_sel_ms    = ms_between(t_a2_sel0, t_a2_sel1);
        const double t_a2_rsw_ms    = ms_between(t_a2_rs0, t_a2_rs1);
        const double t_a2_phase_ms  = ms_between(t_a2_ph0, t_a2_ph1);

        const size_t ctN_bytes  = bytes_TLweSample64(k, N);
        const size_t ctN0_bytes = bytes_TLweSample64(k, N0);

        
        const size_t query_row_bytes  = ctN_bytes; 
        const size_t query_col_bytes  = ctN_bytes; 
        const size_t query_base_bytes = query_row_bytes + query_col_bytes;
        const size_t query_j_bytes    = ctN_bytes; 

        const size_t query_A1_bytes = query_base_bytes;
        const size_t query_A2_bytes = query_base_bytes + query_j_bytes;

        
        const size_t evalkey_common_bytes = summary_convk_bytes + summary_aks_bytes; 
        const size_t evalkey_A1_bytes = evalkey_common_bytes; 
        const size_t evalkey_A2_bytes = evalkey_common_bytes + summary_rswk_bytes; 

        std::cout << "\n==================== APPENDIX: A-1 vs A-2 (FULL PACKING) ====================\n";

        std::cout << "[Preprocess (one-time)]\n";
        print_ms_val("Build DB_rows (u8->torus)", t_pre_db_ms);
        print_ms_val("Build R_cols (interleave)", t_pre_rcols_ms);
        print_ms_val("DB preprocess (IFFT cols)", t_pre_fft_ms);
        print_ms_val("Preprocess total", t_pre_total_ms);

        std::cout << "\n[Offline / Comm]\n";
        std::cout << "  - Base query bytes (A-1/A-2) : " << query_A1_bytes << " bytes (" << (query_A1_bytes/1e6) << " MB)\n";
        std::cout << "  - Extra j-query bytes (A-2)  : " << query_j_bytes  << " bytes (" << (query_j_bytes/1e6)  << " MB)\n";
        std::cout << "  - Total query bytes (A-2)    : " << query_A2_bytes << " bytes (" << (query_A2_bytes/1e6) << " MB)\n";
        std::cout << "  - EvalKey bytes (A-1)        : " << evalkey_A1_bytes << " bytes (" << (evalkey_A1_bytes/1e6) << " MB)\n";
        std::cout << "  - EvalKey bytes (A-2)        : " << evalkey_A2_bytes << " bytes (" << (evalkey_A2_bytes/1e6) << " MB)\n";
        std::cout << "  - Offline total (A-1)        : " << (query_A1_bytes + evalkey_A1_bytes) << " bytes (" << ((query_A1_bytes + evalkey_A1_bytes)/1e6) << " MB)\n";
        std::cout << "  - Offline total (A-2)        : " << (query_A2_bytes + evalkey_A2_bytes) << " bytes (" << ((query_A2_bytes + evalkey_A2_bytes)/1e6) << " MB)\n";

        std::cout << "\n[Online / Comm]\n";
        std::cout << "  - Response bytes (A-1, R_N)  : " << ctN_bytes  << " bytes (" << (ctN_bytes/1e6)  << " MB)\n";
        std::cout << "  - Response bytes (A-2, R_N0) : " << ctN0_bytes << " bytes (" << (ctN0_bytes/1e6) << " MB)\n";
        std::cout << "  - Shrink factor              : " << std::fixed << std::setprecision(3) << ((double)ctN_bytes/(double)ctN0_bytes) << "x\n";

        std::cout << "\n[Online / Compute breakdown (exclude preprocess)]\n";
        print_ms_val("Client: RevHomTrace (base query)", t_revhomtrace_ms);
        print_ms_val("Server: RLWE->RGSW (base unpack)", t_rlwe_to_rgsw_ms);
        print_ms_val("Base unpack total (client+server)", t_revhomtrace_ms + t_rlwe_to_rgsw_ms);

        
        print_ms_val("Server: Main total (1D+CMux)", t_main_total_ms);

        std::cout << "\n[A-1 extra work (client side)]\n";
        print_ms_val("Client: multiply by X^{-j}", t_a1_rot_ms);
        print_ms_val("Client: RingSwitch N->N0", t_a1_rsw_ms);
        print_ms_val("Client: Phase (small ring)", t_a1_phase_ms);

        std::cout << "\n[A-2 extra work (client+server)]\n";
        print_ms_val("Client: Build j query (cipher_prime)", t_a2_buildj_ms);
        print_ms_val("Server: Unpack j query -> RGSW bits", t_a2_unpj_ms);
        print_ms_val("Server: build rotated candidates", t_a2_rot_ms);
        print_ms_val("Server: CMux-select (bits)", t_a2_sel_ms);
        print_ms_val("Server: RingSwitch N->N0", t_a2_rsw_ms);
        print_ms_val("Client: Phase (small ring)", t_a2_phase_ms);

        std::cout << "\n[Totals]\n";
        print_ms_val("Total program (with preprocess)", t_total_with_pre_ms);
        print_ms_val("Total program (no preprocess)", t_total_wo_pre_ms);

        std::cout << "==================================================================\n";
    }


#endif

return 0;
}