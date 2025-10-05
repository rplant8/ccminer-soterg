#include <stdio.h>
#include <cstdint>

#include <memory.h>

#include <miner.h>
#include "cuda_helper.h"

using namespace std;

// MWEB-enhanced hash functions
extern "C" void RinHash_MWEB(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    const uint8_t* mweb_hash,      // MWEB extension block hash (32 bytes)
    uint8_t mweb_present,          // 1 if MWEB data is present, 0 otherwise
    uint8_t* output
);

extern "C" void RinHash_MWEB_mine(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash,
    const uint8_t* mweb_hash,      // MWEB extension block hash
    uint8_t mweb_present           // MWEB presence flag
);

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

// Thread-local variables
thread_local uint32_t *d_hash = NULL;
thread_local uint8_t *d_rinhash_out = NULL;

// Initialization function for RinHash algorithm
extern "C" void rinhash_init(int thr_id)
{
    
}

// Cleanup function for RinHash algorithm
extern "C" void rinhash_free(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);

    // Free allocated memory
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

    // Set up batch mining parameters
    uint32_t batch_size = 4096; // Number of nonces to try in each batch
    uint32_t found_nonce = 0;
    uint8_t best_hash[32];
    uint8_t target_hash[32];

    // Check if MWEB data is present in work
    uint8_t mweb_present = 0;
    uint8_t mweb_hash[32] = {0};
    
    if (work->mweb_data && work->mweb_size > 0) {
        mweb_present = 1;
        // Copy MWEB hash from work data (first 32 bytes of MWEB data is the hash)
        if (work->mweb_size >= 32) {
            memcpy(mweb_hash, work->mweb_data, 32);
        }
    }

    // Convert target to bytes
    for (int i = 0; i < 8; i++) {
        uint32_t tmp = ptarget[i];
        target_hash[i*4+0] = tmp & 0xff;
        target_hash[i*4+1] = (tmp >> 8) & 0xff;
        target_hash[i*4+2] = (tmp >> 16) & 0xff;
        target_hash[i*4+3] = (tmp >> 24) & 0xff;
    }

    work->valid_nonces = 0;
    cudaSetDevice(device_map[thr_id]);

    do {
        uint32_t current_batch_size = min(batch_size, max_nonce - nonce);
        
        if (current_batch_size <= 0) {
            *hashes_done = nonce - first_nonce;
            return 0;
        }

        // Mine a batch of nonces using MWEB-aware hash function
        if (mweb_present) {
            RinHash_MWEB_mine(
                pdata,
                19, // nonce offset in work data
                nonce,
                current_batch_size,
                &found_nonce,
                target_hash,
                best_hash,
                mweb_hash,
                mweb_present
            );
        } else {
            // Fallback to original hash function for non-MWEB blocks
            RinHash_mine(
                pdata,
                19, // nonce offset in work data
                nonce,
                current_batch_size,
                &found_nonce,
                target_hash,
                best_hash
            );
        }

        *hashes_done = nonce - first_nonce + current_batch_size;

        // Check if we found a valid hash
        if (found_nonce != nonce) {
            // Convert best_hash to vhash for verification (FIXED: match target word order)
            uint32_t _ALIGN(64) vhash[8];
            for (int i = 0; i < 8; i++) {
                // GPU outputs little-endian bytes, but we need to match ptarget word order
                // ptarget[7] is most significant word for comparison
                vhash[i] = 0;
                vhash[i] |= best_hash[i*4+0];        // Keep little-endian bytes
                vhash[i] |= best_hash[i*4+1] << 8;
                vhash[i] |= best_hash[i*4+2] << 16;
                vhash[i] |= best_hash[i*4+3] << 24;
            }
            
            const uint32_t Htarg = ptarget[7];
            if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                work->valid_nonces = 1;
                work_set_target_ratio(work, vhash);
                work->nonces[0] = found_nonce;
                pdata[19] = found_nonce + 1;
                return 1;
            } else {
                gpu_increment_reject(thr_id);
            }
        }

        nonce += current_batch_size;

    } while (nonce < max_nonce && !work_restart[thr_id].restart);

    pdata[19] = nonce;
    *hashes_done = nonce - first_nonce;
    return 0;
}

// MWEB-aware hash function for single hash calculation
extern "C" void rinhash_hash_mweb(void *output, const void *input, const uint8_t *mweb_hash, uint8_t mweb_present)
{
    if (mweb_present) {
        RinHash_MWEB(
            (const uint32_t*)input,
            (const uint32_t*)((const uint8_t*)input + 4),
            (const uint32_t*)((const uint8_t*)input + 36),
            (const uint32_t*)((const uint8_t*)input + 68),
            (const uint32_t*)((const uint8_t*)input + 72),
            (const uint32_t*)((const uint8_t*)input + 76),
            mweb_hash,
            mweb_present,
            (uint8_t*)output
        );
    } else {
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
}

// Empty function to detect algorithm - needed by ccminer (kept for compatibility)
extern "C" void rinhash_hash(void *output, const void *input)
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