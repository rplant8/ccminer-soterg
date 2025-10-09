#include <stdio.h>
#include <cstdint>

#include <memory.h>

#include <miner.h>
#include "cuda_helper.h"

using namespace std;

// External reference to RinHash CUDA functions
extern "C" void RinHash(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output
);

extern "C" void RinHash_mine(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash
);

extern "C" void RinHash_mine_persistent(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* target,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash,
    uint32_t* solution_found
);

extern "C" void rinhash_persistent_init(uint32_t max_blocks);
extern "C" void rinhash_persistent_cleanup();

// Thread-local variables
thread_local uint32_t *d_hash = NULL;
thread_local uint8_t *d_rinhash_out = NULL;

// Initialization function for RinHash algorithm
extern "C" void rinhash_init(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);
    rinhash_persistent_init(32768);
}

// Cleanup function for RinHash algorithm
extern "C" void rinhash_free(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);

    rinhash_persistent_cleanup();
    
    cudaFree(d_hash);
    cudaFree(d_rinhash_out);

    d_hash = NULL;
    d_rinhash_out = NULL;
}

// Main scanning function that tries different nonces to find a valid hash
int scanhash_rinhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    uint32_t nonce = first_nonce;
    if (opt_benchmark)
        ptarget[7] = 0xff;

    // 🚀 OPTIMIZED: Tăng batch_size lên 32768 cho GTX 1060 3GB
    uint32_t batch_size = 32768;
    uint32_t found_nonce = 0;
    uint32_t solution_found = 0;
    uint8_t best_hash[32];
    uint8_t target_hash[32];
    uint32_t target[8];

    // Convert target to bytes and uint32_t array
    for (int i = 0; i < 8; i++) {
        target[i] = ptarget[i];
        uint32_t tmp = ptarget[i];
        target_hash[i*4+0] = (tmp >> 24) & 0xff;
        target_hash[i*4+1] = (tmp >> 16) & 0xff;
        target_hash[i*4+2] = (tmp >> 8) & 0xff;
        target_hash[i*4+3] = tmp & 0xff;
    }

    work->valid_nonces = 0;
    cudaSetDevice(device_map[thr_id]);

    do {
        uint32_t current_batch_size = min(batch_size, max_nonce - nonce);
        
        if (current_batch_size <= 0) {
            *hashes_done = nonce - first_nonce;
            return 0;
        }

        // 🚀 OPTIMIZED: Use persistent memory version for 4-6x speedup
        solution_found = 0;
        RinHash_mine_persistent(
            pdata,
            19,
            nonce,
            current_batch_size,
            target,
            &found_nonce,
            target_hash,
            best_hash,
            &solution_found
        );

        *hashes_done = nonce - first_nonce + current_batch_size;

        // Check if we found a valid hash
        if (found_nonce != nonce) {
            // Convert best_hash to vhash for verification
            uint32_t _ALIGN(64) vhash[8];
            for (int i = 0; i < 8; i++) {
                vhash[i] = 0;
                vhash[i] |= best_hash[i*4+0];
                vhash[i] |= best_hash[i*4+1] << 8;
                vhash[i] |= best_hash[i*4+2] << 16;
                vhash[i] |= best_hash[i*4+3] << 24;
            }
            
            const uint32_t Htarg = ptarget[7];
                if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {    
        work->valid_nonces = 1;
                work_set_target_ratio(work, vhash);
                work->nonces[0] = found_nonce;
                pdata[19] = found_nonce;
                return 1;
            } else {
                gpu_increment_reject(thr_id);
            }
        }

        nonce += current_batch_size;
        *hashes_done = nonce - first_nonce; // Update progress continuously

    } while (nonce < max_nonce && !work_restart[thr_id].restart);

    pdata[19] = nonce;
    *hashes_done = nonce - first_nonce;
    return 0;
}

// Empty function to detect algorithm - needed by ccminer
extern "C" void rinhash_hash(const void *output, const void *input)
{
    // Just pass through to the CUDA implementation
    RinHash(
        (const uint32_t*)input,
        (const uint32_t*)((const uint8_t*)input + 4),
        (const uint32_t*)((const uint8_t*)input + 36),
        (const uint32_t*)((const uint8_t*)input + 68),
        (const uint32_t*)((const uint8_t*)input + 72),
        (const uint32_t*)((const uint8_t*)input + 76),
        (uint8_t*)output
    );
}
