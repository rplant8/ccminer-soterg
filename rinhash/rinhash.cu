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
#define MAX_BATCH_BLOCKS 65536
#define OPTIMAL_THREADS_PER_BLOCK 512
#define NUM_STREAMS 4

// 🚀 PERSISTENT MEMORY: Tái sử dụng memory giữa các lần mining
struct PersistentMemory {
    uint8_t *d_headers;
    uint8_t *d_outputs;
    block *d_memories;
    uint32_t *d_target;
    uint32_t *d_solution_found;
    uint32_t *d_solution_nonce;
    uint8_t *h_headers_pinned;
    uint8_t *h_outputs_pinned;
    uint32_t *h_target_pinned;
    uint32_t *h_solution_found_pinned;
    uint32_t *h_solution_nonce_pinned;
    cudaStream_t streams[NUM_STREAMS];
    uint32_t allocated_blocks;
    bool initialized;
};

// Thread-local persistent memory
thread_local PersistentMemory g_persistent_mem = {0};
thread_local uint32_t g_m_cost = 64;

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
        uint32_t swapped_hash = ((hash_words[i] & 0xFF) << 24) | 
                               ((hash_words[i] & 0xFF00) << 8) | 
                               ((hash_words[i] & 0xFF0000) >> 8) | 
                               ((hash_words[i] & 0xFF000000) >> 24);
        if (swapped_hash > target[i]) {
            meets_target = false;
            break;
        } else if (swapped_hash < target[i]) {
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

// Forward declaration
extern "C" void rinhash_persistent_cleanup();

// 🚀 PERSISTENT MEMORY MANAGEMENT: Khởi tạo memory 1 lần duy nhất
extern "C" void rinhash_persistent_init(uint32_t max_blocks) {
    if (g_persistent_mem.initialized && g_persistent_mem.allocated_blocks >= max_blocks) {
        return;
    }
    
    if (g_persistent_mem.initialized) {
        rinhash_persistent_cleanup();
    }
    
    size_t headers_size = 80 * max_blocks;
    size_t outputs_size = 32 * max_blocks;
    size_t memories_size = max_blocks * g_m_cost * sizeof(block);
    
    cudaMalloc(&g_persistent_mem.d_headers, headers_size);
    cudaMalloc(&g_persistent_mem.d_outputs, outputs_size);
    cudaMalloc(&g_persistent_mem.d_memories, memories_size);
    cudaMalloc(&g_persistent_mem.d_target, 8 * sizeof(uint32_t));
    cudaMalloc(&g_persistent_mem.d_solution_found, sizeof(uint32_t));
    cudaMalloc(&g_persistent_mem.d_solution_nonce, sizeof(uint32_t));
    
    cudaHostAlloc(&g_persistent_mem.h_headers_pinned, headers_size, cudaHostAllocDefault);
    cudaHostAlloc(&g_persistent_mem.h_outputs_pinned, outputs_size, cudaHostAllocDefault);
    cudaHostAlloc(&g_persistent_mem.h_target_pinned, 8 * sizeof(uint32_t), cudaHostAllocDefault);
    cudaHostAlloc(&g_persistent_mem.h_solution_found_pinned, sizeof(uint32_t), cudaHostAllocDefault);
    cudaHostAlloc(&g_persistent_mem.h_solution_nonce_pinned, sizeof(uint32_t), cudaHostAllocDefault);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&g_persistent_mem.streams[i]);
    }
    
    g_persistent_mem.allocated_blocks = max_blocks;
    g_persistent_mem.initialized = true;
    
    fprintf(stderr, "✅ Persistent memory initialized: %u blocks (%.2f MB)\n", 
            max_blocks, (headers_size + outputs_size + memories_size) / (1024.0 * 1024.0));
}

// 🚀 PERSISTENT MEMORY CLEANUP
extern "C" void rinhash_persistent_cleanup() {
    if (!g_persistent_mem.initialized) return;
    
    cudaFree(g_persistent_mem.d_headers);
    cudaFree(g_persistent_mem.d_outputs);
    cudaFree(g_persistent_mem.d_memories);
    cudaFree(g_persistent_mem.d_target);
    cudaFree(g_persistent_mem.d_solution_found);
    cudaFree(g_persistent_mem.d_solution_nonce);
    
    cudaFreeHost(g_persistent_mem.h_headers_pinned);
    cudaFreeHost(g_persistent_mem.h_outputs_pinned);
    cudaFreeHost(g_persistent_mem.h_target_pinned);
    cudaFreeHost(g_persistent_mem.h_solution_found_pinned);
    cudaFreeHost(g_persistent_mem.h_solution_nonce_pinned);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(g_persistent_mem.streams[i]);
    }
    
    memset(&g_persistent_mem, 0, sizeof(PersistentMemory));
    fprintf(stderr, "🧹 Persistent memory cleaned up\n");
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

// 🚀 NEW: ULTRA-OPTIMIZED with persistent memory + streams + pipeline
extern "C" void rinhash_cuda_batch_persistent(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks,
    uint32_t* target,
    uint32_t* solution_found,
    uint32_t* solution_nonce
) {
    if (num_blocks > MAX_BATCH_BLOCKS) {
        fprintf(stderr, "Batch too large (max %u)\n", MAX_BATCH_BLOCKS);
        return;
    }
    
    rinhash_persistent_init(num_blocks);
    
    size_t headers_size = block_header_len * num_blocks;
    size_t outputs_size = 32 * num_blocks;
    
    const int threads_per_block = OPTIMAL_THREADS_PER_BLOCK;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    
    memcpy(g_persistent_mem.h_headers_pinned, block_headers, headers_size);
    memcpy(g_persistent_mem.h_target_pinned, target, 8 * sizeof(uint32_t));
    
    cudaMemcpyAsync(g_persistent_mem.d_headers, g_persistent_mem.h_headers_pinned, 
                    headers_size, cudaMemcpyHostToDevice, g_persistent_mem.streams[0]);
    cudaMemcpyAsync(g_persistent_mem.d_target, g_persistent_mem.h_target_pinned, 
                    8 * sizeof(uint32_t), cudaMemcpyHostToDevice, g_persistent_mem.streams[0]);
    cudaMemsetAsync(g_persistent_mem.d_solution_found, 0, sizeof(uint32_t), g_persistent_mem.streams[0]);
    
    rinhash_cuda_kernel_optimized<<<blocks, threads_per_block, 0, g_persistent_mem.streams[0]>>>(
        g_persistent_mem.d_headers, block_header_len, g_persistent_mem.d_outputs, 
        num_blocks, g_persistent_mem.d_memories, g_m_cost,
        g_persistent_mem.d_target, g_persistent_mem.d_solution_found, g_persistent_mem.d_solution_nonce
    );
    
    cudaMemcpyAsync(g_persistent_mem.h_outputs_pinned, g_persistent_mem.d_outputs, 
                    outputs_size, cudaMemcpyDeviceToHost, g_persistent_mem.streams[0]);
    cudaMemcpyAsync(g_persistent_mem.h_solution_found_pinned, g_persistent_mem.d_solution_found, 
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, g_persistent_mem.streams[0]);
    cudaMemcpyAsync(g_persistent_mem.h_solution_nonce_pinned, g_persistent_mem.d_solution_nonce, 
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, g_persistent_mem.streams[0]);
    
    cudaStreamSynchronize(g_persistent_mem.streams[0]);
    check_cuda("rinhash_cuda_batch_persistent");
    
    memcpy(outputs, g_persistent_mem.h_outputs_pinned, outputs_size);
    *solution_found = *g_persistent_mem.h_solution_found_pinned;
    *solution_nonce = *g_persistent_mem.h_solution_nonce_pinned;
}

// 🚀 LEGACY: Target-aware batch processing (kept for compatibility)
extern "C" void rinhash_cuda_batch_optimized(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks,
    uint32_t* target,
    uint32_t* solution_found,
    uint32_t* solution_nonce
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

    const int threads_per_block = OPTIMAL_THREADS_PER_BLOCK;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;

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

// 🚀 ULTRA-OPTIMIZED: Persistent memory version for maximum performance
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
) {
    const size_t block_header_len = 80;
    if (num_nonces > MAX_BATCH_BLOCKS) {
        fprintf(stderr, "Mining batch too large (max %u)\n", MAX_BATCH_BLOCKS);
        return;
    }
    
    rinhash_persistent_init(num_nonces);
    
    uint32_t solution_nonce = 0;
    
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t current_nonce = start_nonce + i;
        uint32_t work_data_copy[20];
        memcpy(work_data_copy, work_data, 80);
        work_data_copy[nonce_offset] = current_nonce;
        memcpy(&g_persistent_mem.h_headers_pinned[i * block_header_len], work_data_copy, 80);
    }
    
    rinhash_cuda_batch_persistent(
        g_persistent_mem.h_headers_pinned, block_header_len, 
        g_persistent_mem.h_outputs_pinned, num_nonces,
        target, solution_found, &solution_nonce
    );
    
    if (*solution_found) {
        *found_nonce = solution_nonce;
        uint32_t winner_index = solution_nonce - start_nonce;
        if (winner_index < num_nonces) {
            memcpy(best_hash, g_persistent_mem.h_outputs_pinned + winner_index * 32, 32);
        }
    } else {
        memcpy(best_hash, g_persistent_mem.h_outputs_pinned, 32);
        *found_nonce = start_nonce;
        for (uint32_t i = 1; i < num_nonces; i++) {
            uint8_t* current_hash = g_persistent_mem.h_outputs_pinned + i * 32;
            if (is_better(current_hash, best_hash)) {
                memcpy(best_hash, current_hash, 32);
                *found_nonce = start_nonce + i;
            }
        }
    }
}

// 🚀 LEGACY: Enhanced mining function with target-aware early termination
extern "C" void RinHash_mine_optimized(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* target,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash,
    uint32_t* solution_found
) {
    const size_t block_header_len = 80;
    if (num_nonces > MAX_BATCH_BLOCKS) {
        fprintf(stderr, "Mining batch too large (max %u)\n", MAX_BATCH_BLOCKS);
        return;
    }
    
    std::vector<uint8_t> block_headers(block_header_len * num_nonces);
    std::vector<uint8_t> hashes(32 * num_nonces);
    uint32_t solution_nonce = 0;

    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t current_nonce = start_nonce + i;
        uint32_t work_data_copy[20];
        memcpy(work_data_copy, work_data, 80);
        work_data_copy[nonce_offset] = current_nonce;
        memcpy(&block_headers[i * block_header_len], work_data_copy, 80);
    }

    rinhash_cuda_batch_optimized(
        block_headers.data(), block_header_len, hashes.data(), num_nonces,
        target, solution_found, &solution_nonce
    );

    if (*solution_found) {
        *found_nonce = solution_nonce;
        uint32_t winner_index = solution_nonce - start_nonce;
        if (winner_index < num_nonces) {
            memcpy(best_hash, hashes.data() + winner_index * 32, 32);
        }
    } else {
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
