#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdexcept>

// Include shared device functions (chỉ include .cuh hoặc .h)
#include "rinhash_device.cuh"
#include "argon2d_device.cuh"
#include "sha3-256.cu"
#include "blake3_device.cuh"

// 🚀 GTX 1060 3GB OPTIMIZED: Balance memory usage vs performance
#define MAX_BATCH_BLOCKS 32768

// Kernel đơn: mỗi lần chỉ chạy 1 thread
extern "C" __global__ void rinhash_cuda_kernel(
    const uint8_t* input, 
    size_t input_len, 
    uint8_t* output,
    block* memory,      // bộ nhớ argon2 đã cấp phát trên host, truyền vào
    uint32_t m_cost
) {
    // Chỉ 1 thread xử lý
    if (threadIdx.x == 0) {
        uint8_t blake3_out[32];
        light_hash_device(input, input_len, blake3_out);

        uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
        uint8_t argon2_out[32];
        device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, memory, salt, sizeof(salt));

        uint8_t sha3_out[32];
        sha3_256_device(argon2_out, 32, sha3_out);

        // Copy kết quả ra output
        for (int i = 0; i < 32; i++) output[i] = sha3_out[i];
    }
}

// 🚀 OPTIMIZED Kernel batch with target-aware early termination
extern "C" __global__ void rinhash_cuda_kernel_batch(
    const uint8_t* headers,         // num_blocks * 80 bytes
    size_t header_len,              // = 80
    uint8_t* outputs,               // num_blocks * 32 bytes
    uint32_t num_blocks,
    block* memories,                // num_blocks * m_cost * sizeof(block)
    uint32_t m_cost
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    
    const uint8_t* input = headers + tid * header_len;
    uint8_t* output = outputs + tid * 32;
    block* memory = memories + tid * m_cost;

    uint8_t blake3_out[32];
    light_hash_device(input, header_len, blake3_out);

    uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
    uint8_t argon2_out[32];
    device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, memory, salt, sizeof(salt));

    sha3_256_device(argon2_out, 32, output);
}

// 🚀 NEW: Target-aware kernel with atomic solution detection
extern "C" __global__ void rinhash_cuda_kernel_optimized(
    const uint8_t* headers,
    size_t header_len,
    uint8_t* outputs,
    uint32_t num_blocks,
    block* memories,
    uint32_t m_cost,
    uint32_t* target,           // 8 x uint32_t target
    uint32_t* solution_found,   // atomic flag
    uint32_t* solution_nonce    // winning nonce
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    
    // Early exit if solution already found
    if (atomicAdd(solution_found, 0) > 0) return;
    
    const uint8_t* input = headers + tid * header_len;
    uint8_t* output = outputs + tid * 32;
    block* memory = memories + tid * m_cost;

    uint8_t blake3_out[32];
    light_hash_device(input, header_len, blake3_out);

    uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
    uint8_t argon2_out[32];
    device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, memory, salt, sizeof(salt));

    sha3_256_device(argon2_out, 32, output);
    
    // Quick target check - convert hash to uint32_t array
    uint32_t* hash_words = (uint32_t*)output;
    
    // Check if hash meets target (little-endian comparison from back)
    bool meets_target = true;
    for (int i = 7; i >= 0; i--) {
//        uint32_t* hash_words[i] = ((hash_words[i] & 0xFF) << 24) | 
//                               ((hash_words[i] & 0xFF00) << 8) | 
//                               ((hash_words[i] & 0xFF0000) >> 8) | 
//                               ((hash_words[i] & 0xFF000000) >> 24);
        if (hash_words[i] > target[i]) {
            meets_target = false;
            break;
        } else if (hash_words[i] < target[i]) {
            break; // This hash is better, continue to set solution
        }
    }
    
    if (meets_target) {
        // Atomic solution detection - first thread wins
        if (atomicCAS(solution_found, 0, 1) == 0) {
            // Extract nonce from header (last 4 bytes)
            uint32_t* header_words = (uint32_t*)(input);
            *solution_nonce = header_words[19]; // nonce is at offset 76 bytes = word 19
        }
    }
}


// Helper: kiểm tra lỗi CUDA
inline void check_cuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        throw std::runtime_error("CUDA error");
    }
}

// Cleanup persistent GPU memory (required by rinhash_scanhash.cpp)
extern "C" void rinhash_cuda_cleanup_persistent() {
    // Reset CUDA device to clean up any persistent memory
    cudaDeviceReset();
}

// RinHash CUDA implementation (single)
extern "C" void rinhash_cuda(const uint8_t* input, size_t input_len, uint8_t* output) {
    uint8_t *d_input = nullptr;
    uint8_t *d_output = nullptr;
    block* d_memory = nullptr;
    uint32_t m_cost = 64;

    cudaError_t err;

    // Alloc device memory
    err = cudaMalloc(&d_input, input_len);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc input fail\n"); return; }

    err = cudaMalloc(&d_output, 32);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc output fail\n"); cudaFree(d_input); return; }

    err = cudaMalloc(&d_memory, m_cost * sizeof(block));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc argon2 memory fail\n"); cudaFree(d_input); cudaFree(d_output); return; }

    // Copy input
    err = cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy input fail\n"); cudaFree(d_input); cudaFree(d_output); cudaFree(d_memory); return; }

    // Launch kernel
    rinhash_cuda_kernel<<<1, 1>>>(d_input, input_len, d_output, d_memory, m_cost);
    cudaDeviceSynchronize();
    check_cuda("rinhash_cuda_kernel");

    // Copy result
    err = cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy output fail\n"); }

    // Free
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_memory);
}

// 🚀 OPTIMIZED: Target-aware batch processing for faster mining
extern "C" void rinhash_cuda_batch_optimized(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks,
    uint32_t* target,           // Target for early termination
    uint32_t* solution_found,   // Output: 1 if solution found
    uint32_t* solution_nonce    // Output: winning nonce
) {
    if (num_blocks > MAX_BATCH_BLOCKS) {
        fprintf(stderr, "Batch too large (max %u)\n", MAX_BATCH_BLOCKS);
        return;
    }

    uint8_t *d_headers = nullptr, *d_outputs = nullptr;
    block* d_memories = nullptr;
    uint32_t *d_target = nullptr, *d_solution_found = nullptr, *d_solution_nonce = nullptr;
    uint32_t m_cost = 64;
    
    size_t headers_size = block_header_len * num_blocks;
    size_t outputs_size = 32 * num_blocks;
    size_t memories_size = num_blocks * m_cost * sizeof(block);

    // 🚀 GTX 1060 OPTIMIZED: Define thread configuration first
    const int threads_per_block = 256;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;

    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(&d_headers, headers_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc headers fail\n"); return; }
    err = cudaMalloc(&d_outputs, outputs_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc outputs fail\n"); cudaFree(d_headers); return; }
    err = cudaMalloc(&d_memories, memories_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc argon2 memories fail\n"); cudaFree(d_headers); cudaFree(d_outputs); return; }
    err = cudaMalloc(&d_target, 8 * sizeof(uint32_t));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc target fail\n"); goto cleanup; }
    err = cudaMalloc(&d_solution_found, sizeof(uint32_t));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc solution_found fail\n"); goto cleanup; }
    err = cudaMalloc(&d_solution_nonce, sizeof(uint32_t));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc solution_nonce fail\n"); goto cleanup; }

    // Initialize data
    cudaMemset(d_outputs, 0xee, outputs_size);
    cudaMemset(d_solution_found, 0, sizeof(uint32_t));
    cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    rinhash_cuda_kernel_optimized<<<blocks, threads_per_block>>>(
        d_headers, block_header_len, d_outputs, num_blocks, d_memories, m_cost,
        d_target, d_solution_found, d_solution_nonce
    );
    cudaDeviceSynchronize();
    check_cuda("rinhash_cuda_kernel_optimized");

    // Copy results back
    err = cudaMemcpy(outputs, d_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy output batch fail\n"); }
    
    err = cudaMemcpy(solution_found, d_solution_found, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy solution_found fail\n"); }
    
    err = cudaMemcpy(solution_nonce, d_solution_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy solution_nonce fail\n"); }

cleanup:
    cudaFree(d_headers);
    cudaFree(d_outputs);
    cudaFree(d_memories);
    cudaFree(d_target);
    cudaFree(d_solution_found);
    cudaFree(d_solution_nonce);
}

// Batch processing version for mining (legacy - kept for compatibility)
extern "C" void rinhash_cuda_batch(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks
) {
    if (num_blocks > MAX_BATCH_BLOCKS) {
        fprintf(stderr, "Batch too large (max %u)\n", MAX_BATCH_BLOCKS);
        return;
    }

    uint8_t *d_headers = nullptr, *d_outputs = nullptr;
    block* d_memories = nullptr;
    uint32_t m_cost = 64;
    size_t headers_size = block_header_len * num_blocks;
    size_t outputs_size = 32 * num_blocks;
    size_t memories_size = num_blocks * m_cost * sizeof(block);

    cudaError_t err;
    err = cudaMalloc(&d_headers, headers_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc headers fail\n"); return; }
    err = cudaMalloc(&d_outputs, outputs_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc outputs fail\n"); cudaFree(d_headers); return; }
    err = cudaMalloc(&d_memories, memories_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc argon2 memories fail\n"); cudaFree(d_headers); cudaFree(d_outputs); return; }

    cudaMemset(d_outputs, 0xee, outputs_size);
    cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);

    // 🚀 GTX 1060 OPTIMIZED: 256 threads per block for better GPU utilization
    const int threads_per_block = 256;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    rinhash_cuda_kernel_batch<<<blocks, threads_per_block>>>(
        d_headers, block_header_len, d_outputs, num_blocks, d_memories, m_cost
    );
    cudaDeviceSynchronize();
    check_cuda("rinhash_cuda_kernel_batch");

    err = cudaMemcpy(outputs, d_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy output batch fail\n"); }

    cudaFree(d_headers);
    cudaFree(d_outputs);
    cudaFree(d_memories);
}

// Helper function to convert a block header to bytes
extern "C" void blockheader_to_bytes(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output,
    size_t* output_len
) {
    size_t offset = 0;
    memcpy(output + offset, version, 4); offset += 4;
    memcpy(output + offset, prev_block, 32); offset += 32;
    memcpy(output + offset, merkle_root, 32); offset += 32;
    memcpy(output + offset, timestamp, 4); offset += 4;
    memcpy(output + offset, bits, 4); offset += 4;
    memcpy(output + offset, nonce, 4); offset += 4;
    *output_len = offset;
}

// Main RinHash function that would be called from outside
extern "C" void RinHash(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output
) {
    uint8_t block_header[80]; // Standard block header size
    size_t block_header_len;
    blockheader_to_bytes(
        version,
        prev_block,
        merkle_root,
        timestamp,
        bits,
        nonce,
        block_header,
        &block_header_len
    );
    rinhash_cuda(block_header, block_header_len, output);
}

bool is_better(uint8_t* hash1, uint8_t* hash2) {
    for (int i = 7; i >= 0; i--) {
        uint32_t h1 = ((uint32_t)hash1[i*4 + 0]) |
                      ((uint32_t)hash1[i*4 + 1] << 8) |
                      ((uint32_t)hash1[i*4 + 2] << 16) |
                      ((uint32_t)hash1[i*4 + 3] << 24);
        uint32_t h2 = ((uint32_t)hash2[i*4 + 0]) |
                      ((uint32_t)hash2[i*4 + 1] << 8) |
                      ((uint32_t)hash2[i*4 + 2] << 16) |
                      ((uint32_t)hash2[i*4 + 3] << 24);
        if (h1 < h2) return true;
        if (h1 > h2) return false;
    }
    return false; // equal
}

// 🚀 OPTIMIZED: Enhanced mining function with target-aware early termination
extern "C" void RinHash_mine_optimized(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* target,           // 8 x uint32_t target  
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash,
    uint32_t* solution_found    // 1 if target was met
) {
    const size_t block_header_len = 80;
    if (num_nonces > MAX_BATCH_BLOCKS) {
        fprintf(stderr, "Mining batch too large (max %u)\n", MAX_BATCH_BLOCKS);
        return;
    }
    
    std::vector<uint8_t> block_headers(block_header_len * num_nonces);
    std::vector<uint8_t> hashes(32 * num_nonces);
    uint32_t solution_nonce = 0;

    // Prepare block headers with different nonces
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t current_nonce = start_nonce + i;
        uint32_t work_data_copy[20];
        memcpy(work_data_copy, work_data, 80);
        work_data_copy[nonce_offset] = current_nonce;
        memcpy(&block_headers[i * block_header_len], work_data_copy, 80);
    }

    // Use optimized kernel with target checking
    rinhash_cuda_batch_optimized(
        block_headers.data(), block_header_len, hashes.data(), num_nonces,
        target, solution_found, &solution_nonce
    );

    if (*solution_found) {
        // Solution found! Extract the winning hash
        *found_nonce = solution_nonce;
        uint32_t winner_index = solution_nonce - start_nonce;
        if (winner_index < num_nonces) {
            memcpy(best_hash, hashes.data() + winner_index * 32, 32);
        }
    } else {
        // No solution, find best hash
        memcpy(best_hash, hashes.data(), 32);
        *found_nonce = start_nonce;
        for (uint32_t i = 1; i < num_nonces; i++) {
            uint8_t* current_hash = hashes.data() + i * 32;
            if (is_better(current_hash, best_hash)) {
                memcpy(best_hash, current_hash, 32);
                *found_nonce = start_nonce + i;
            }
        }
    }
}

// Legacy mining function (kept for compatibility)
extern "C" void RinHash_mine(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash
) {
    const size_t block_header_len = 80;
    if (num_nonces > MAX_BATCH_BLOCKS) {
        fprintf(stderr, "Mining batch too large (max %u)\n", MAX_BATCH_BLOCKS);
        return;
    }
    std::vector<uint8_t> block_headers(block_header_len * num_nonces);
    std::vector<uint8_t> hashes(32 * num_nonces);

    // Prepare block headers with different nonces
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t current_nonce = start_nonce + i;
        uint32_t work_data_copy[20];
        memcpy(work_data_copy, work_data, 80);
        work_data_copy[nonce_offset] = current_nonce;
        memcpy(&block_headers[i * block_header_len], work_data_copy, 80);
    }

    // Calculate hashes for all nonces
    rinhash_cuda_batch(block_headers.data(), block_header_len, hashes.data(), num_nonces);

    // Initialize best_hash with the first hash
    memcpy(best_hash, hashes.data(), 32);
    *found_nonce = start_nonce;
    for (uint32_t i = 1; i < num_nonces; i++) {
        uint8_t* current_hash = hashes.data() + i * 32;
        if (is_better(current_hash, best_hash)) {
            memcpy(best_hash, current_hash, 32);
            *found_nonce = start_nonce + i;
        }
    }
}

// MWEB-enhanced hash function
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
) {
    // Use standard RinHash as base
    RinHash(version, prev_block, merkle_root, timestamp, bits, nonce, output);
    
    // If MWEB is present, XOR with MWEB hash for additional mixing
    if (mweb_present && mweb_hash) {
        for (int i = 0; i < 32; i++) {
            output[i] ^= mweb_hash[i % 32];
        }
    }
}

// MWEB-enhanced mining function  
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
) {
    // Use regular mining then post-process if MWEB is present
    RinHash_mine(work_data, nonce_offset, start_nonce, num_nonces, found_nonce, target_hash, best_hash);
    
    // Apply MWEB mixing to best hash if present
    if (mweb_present && mweb_hash) {
        for (int i = 0; i < 32; i++) {
            best_hash[i] ^= mweb_hash[i % 32];
        }
    }
}
