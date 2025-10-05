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

// Include CPU Argon2d reference implementation
extern "C" {
#include "argon2d/argon2ref/argon2.h"
}

// 🚀 GTX 1060 3GB OPTIMIZED: Balance memory usage vs performance
#define MAX_BATCH_BLOCKS 4096

// Kernel chỉ chạy BLAKE3 (GPU)
extern "C" __global__ void blake3_only_kernel(
    const uint8_t* input, 
    size_t input_len, 
    uint8_t* output
) {
    if (threadIdx.x == 0) {
        light_hash_device(input, input_len, output);
    }
}

// Kernel chỉ chạy SHA3-256 (GPU)
extern "C" __global__ void sha3_256_only_kernel(
    const uint8_t* input,
    uint8_t* output
) {
    if (threadIdx.x == 0) {
        sha3_256_device(input, 32, output);
    }
}

// Kernel batch chỉ chạy BLAKE3 (GPU) - cho batch processing
extern "C" __global__ void blake3_batch_kernel(
    const uint8_t* headers,
    size_t header_len,
    uint8_t* outputs,
    uint32_t num_blocks
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    
    const uint8_t* input = headers + tid * header_len;
    uint8_t* output = outputs + tid * 32;
    
    light_hash_device(input, header_len, output);
}

// Kernel batch chỉ chạy SHA3-256 (GPU) - cho batch processing
extern "C" __global__ void sha3_256_batch_kernel(
    const uint8_t* inputs,
    uint8_t* outputs,
    uint32_t num_blocks
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    
    const uint8_t* input = inputs + tid * 32;
    uint8_t* output = outputs + tid * 32;
    
    sha3_256_device(input, 32, output);
}

// Kernel đơn: OLD VERSION - giữ để tương thích ngược
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

// RinHash CUDA implementation (single) - HYBRID: GPU BLAKE3 + CPU Argon2d + GPU SHA3-256
extern "C" void rinhash_cuda(const uint8_t* input, size_t input_len, uint8_t* output) {
    uint8_t *d_input = nullptr;
    uint8_t *d_blake3_out = nullptr;
    uint8_t *d_sha3_out = nullptr;
    
    cudaError_t err;

    // Alloc device memory for BLAKE3
    err = cudaMalloc(&d_input, input_len);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc input fail\n"); return; }

    err = cudaMalloc(&d_blake3_out, 32);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc blake3 output fail\n"); cudaFree(d_input); return; }

    // Copy input
    err = cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy input fail\n"); cudaFree(d_input); cudaFree(d_blake3_out); return; }

    // Step 1: Launch BLAKE3 kernel (GPU)
    blake3_only_kernel<<<1, 1>>>(d_input, input_len, d_blake3_out);
    cudaDeviceSynchronize();
    check_cuda("blake3_only_kernel");

    // Copy BLAKE3 result to CPU
    uint8_t blake3_out[32];
    err = cudaMemcpy(blake3_out, d_blake3_out, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy blake3 output fail\n"); cudaFree(d_input); cudaFree(d_blake3_out); return; }

    // Step 2: Run Argon2d on CPU
    uint8_t argon2_out[32];
    uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
    
    int result = argon2d_hash_raw(
        2,          // t_cost (iterations)
        64,         // m_cost (memory in KB)
        1,          // parallelism (lanes)
        blake3_out, // pwd
        32,         // pwdlen
        salt,       // salt
        sizeof(salt), // saltlen
        argon2_out, // hash output
        32          // hashlen
    );
    
    if (result != ARGON2_OK) {
        fprintf(stderr, "Argon2d CPU error: %s\n", argon2_error_message(result));
        cudaFree(d_input);
        cudaFree(d_blake3_out);
        return;
    }

    // Step 3: Run SHA3-256 on GPU
    err = cudaMalloc(&d_sha3_out, 32);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc sha3 output fail\n"); cudaFree(d_input); cudaFree(d_blake3_out); return; }
    
    // Reuse d_blake3_out buffer for argon2 result
    err = cudaMemcpy(d_blake3_out, argon2_out, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy argon2 output fail\n"); cudaFree(d_input); cudaFree(d_blake3_out); cudaFree(d_sha3_out); return; }
    
    sha3_256_only_kernel<<<1, 1>>>(d_blake3_out, d_sha3_out);
    cudaDeviceSynchronize();
    check_cuda("sha3_256_only_kernel");

    // Copy final result
    err = cudaMemcpy(output, d_sha3_out, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy sha3 output fail\n"); }

    // Free
    cudaFree(d_input);
    cudaFree(d_blake3_out);
    cudaFree(d_sha3_out);
}

// 🚀 OPTIMIZED: Target-aware batch processing - HYBRID: GPU BLAKE3 + CPU Argon2d + GPU SHA3-256
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

    uint8_t *d_headers = nullptr;
    uint8_t *d_blake3_outputs = nullptr;
    uint8_t *d_sha3_outputs = nullptr;
    
    size_t headers_size = block_header_len * num_blocks;
    size_t outputs_size = 32 * num_blocks;

    const int threads_per_block = 256;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;

    cudaError_t err;
    
    // Allocate GPU memory
    err = cudaMalloc(&d_headers, headers_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc headers fail\n"); return; }
    err = cudaMalloc(&d_blake3_outputs, outputs_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc blake3 outputs fail\n"); cudaFree(d_headers); return; }
    err = cudaMalloc(&d_sha3_outputs, outputs_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc sha3 outputs fail\n"); cudaFree(d_headers); cudaFree(d_blake3_outputs); return; }

    // Declare variables before any goto to avoid bypass initialization error
    std::vector<uint8_t> blake3_outputs(outputs_size);
    std::vector<uint8_t> argon2_outputs(outputs_size);
    uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
    
    // Copy headers to GPU
    cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);

    // Step 1: Run BLAKE3 on GPU (batch)
    blake3_batch_kernel<<<blocks, threads_per_block>>>(
        d_headers, block_header_len, d_blake3_outputs, num_blocks
    );
    cudaDeviceSynchronize();
    check_cuda("blake3_batch_kernel");

    // Copy BLAKE3 results to CPU
    err = cudaMemcpy(blake3_outputs.data(), d_blake3_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy blake3 outputs fail\n"); goto cleanup; }

    // Step 2: Run Argon2d on CPU for each block with target checking
    *solution_found = 0;
    
    for (uint32_t i = 0; i < num_blocks; i++) {
        int result = argon2d_hash_raw(
            2,          // t_cost
            64,         // m_cost
            1,          // parallelism
            blake3_outputs.data() + i * 32, // pwd
            32,         // pwdlen
            salt,       // salt
            sizeof(salt), // saltlen
            argon2_outputs.data() + i * 32, // hash output
            32          // hashlen
        );
        
        if (result != ARGON2_OK) {
            fprintf(stderr, "Argon2d CPU error for block %u: %s\n", i, argon2_error_message(result));
            goto cleanup;
        }
    }

    // Copy Argon2d results to GPU
    err = cudaMemcpy(d_blake3_outputs, argon2_outputs.data(), outputs_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy argon2 outputs fail\n"); goto cleanup; }

    // Step 3: Run SHA3-256 on GPU (batch)
    sha3_256_batch_kernel<<<blocks, threads_per_block>>>(
        d_blake3_outputs, d_sha3_outputs, num_blocks
    );
    cudaDeviceSynchronize();
    check_cuda("sha3_256_batch_kernel");

    // Copy final results to host
    err = cudaMemcpy(outputs, d_sha3_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy sha3 outputs fail\n"); }

    // Check for solution on CPU (since we have all hashes)
    for (uint32_t i = 0; i < num_blocks; i++) {
        uint32_t* hash_words = (uint32_t*)(outputs + i * 32);
        bool meets_target = true;
        
        for (int j = 7; j >= 0; j--) {
            if (hash_words[j] > target[j]) {
                meets_target = false;
                break;
            } else if (hash_words[j] < target[j]) {
                break;
            }
        }
        
        if (meets_target) {
            *solution_found = 1;
            // Extract nonce from header (last 4 bytes = offset 76)
            uint32_t* header_words = (uint32_t*)(block_headers + i * block_header_len);
            *solution_nonce = header_words[19];
            break;
        }
    }

cleanup:
    cudaFree(d_headers);
    cudaFree(d_blake3_outputs);
    cudaFree(d_sha3_outputs);
}

// Batch processing version for mining - HYBRID: GPU BLAKE3 + CPU Argon2d + GPU SHA3-256
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

    uint8_t *d_headers = nullptr;
    uint8_t *d_blake3_outputs = nullptr;
    uint8_t *d_sha3_outputs = nullptr;
    
    size_t headers_size = block_header_len * num_blocks;
    size_t outputs_size = 32 * num_blocks;

    cudaError_t err;
    
    // Allocate GPU memory
    err = cudaMalloc(&d_headers, headers_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc headers fail\n"); return; }
    err = cudaMalloc(&d_blake3_outputs, outputs_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc blake3 outputs fail\n"); cudaFree(d_headers); return; }
    err = cudaMalloc(&d_sha3_outputs, outputs_size);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: alloc sha3 outputs fail\n"); cudaFree(d_headers); cudaFree(d_blake3_outputs); return; }

    // Declare variables before any goto to avoid bypass initialization error
    std::vector<uint8_t> blake3_outputs(outputs_size);
    std::vector<uint8_t> argon2_outputs(outputs_size);
    uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
    
    // Copy headers to GPU
    cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);

    // Step 1: Run BLAKE3 on GPU (batch)
    const int threads_per_block = 256;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    
    blake3_batch_kernel<<<blocks, threads_per_block>>>(
        d_headers, block_header_len, d_blake3_outputs, num_blocks
    );
    cudaDeviceSynchronize();
    check_cuda("blake3_batch_kernel");

    // Copy BLAKE3 results to CPU
    err = cudaMemcpy(blake3_outputs.data(), d_blake3_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy blake3 outputs fail\n"); goto cleanup; }

    // Step 2: Run Argon2d on CPU for each block
    
    for (uint32_t i = 0; i < num_blocks; i++) {
        int result = argon2d_hash_raw(
            2,          // t_cost
            64,         // m_cost
            1,          // parallelism
            blake3_outputs.data() + i * 32, // pwd
            32,         // pwdlen
            salt,       // salt
            sizeof(salt), // saltlen
            argon2_outputs.data() + i * 32, // hash output
            32          // hashlen
        );
        
        if (result != ARGON2_OK) {
            fprintf(stderr, "Argon2d CPU error for block %u: %s\n", i, argon2_error_message(result));
            goto cleanup;
        }
    }

    // Copy Argon2d results to GPU
    err = cudaMemcpy(d_blake3_outputs, argon2_outputs.data(), outputs_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy argon2 outputs fail\n"); goto cleanup; }

    // Step 3: Run SHA3-256 on GPU (batch)
    sha3_256_batch_kernel<<<blocks, threads_per_block>>>(
        d_blake3_outputs, d_sha3_outputs, num_blocks
    );
    cudaDeviceSynchronize();
    check_cuda("sha3_256_batch_kernel");

    // Copy final results to host
    err = cudaMemcpy(outputs, d_sha3_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA: copy sha3 outputs fail\n"); }

cleanup:
    cudaFree(d_headers);
    cudaFree(d_blake3_outputs);
    cudaFree(d_sha3_outputs);
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