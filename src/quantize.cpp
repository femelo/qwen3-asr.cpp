#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>


ggml_type ggml_parse_type(const std::string & str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    if (s == "F32")    return GGML_TYPE_F32;
    if (s == "F16")    return GGML_TYPE_F16;
    if (s == "Q4_0")   return GGML_TYPE_Q4_0;
    if (s == "Q4_1")   return GGML_TYPE_Q4_1;
    if (s == "Q4_K" || s == "Q4_K_M" || s == "Q4_K_S") return GGML_TYPE_Q4_K;
    if (s == "Q5_0")   return GGML_TYPE_Q5_0;
    if (s == "Q5_1")   return GGML_TYPE_Q5_1;
    if (s == "Q5_K" || s == "Q5_K_M" || s == "Q5_K_S") return GGML_TYPE_Q5_K;
    if (s == "Q8_0")   return GGML_TYPE_Q8_0;
    return GGML_TYPE_COUNT;
}


// Helper to check alignment
bool is_aligned(struct ggml_tensor * tensor, ggml_type type) {
    // ggml_blck_size returns the required multiple for a given type
    // e.g., 32 for Q4_0, 256 for Q4_K
    int block_size = ggml_blck_size(type);
    return (tensor->ne[0] % block_size == 0);
}


bool should_quantize(const std::string & name, struct ggml_tensor * tensor, ggml_type type) {
    // 1. Skip by name (Standard heuristic)
    if (name.find("bias") != std::string::npos || 
        name.find("norm") != std::string::npos ||
        name.find("token_embd") != std::string::npos) {
        return false;
    }

    // 2. Skip if not aligned (CRITICAL for Conv layers)
    if (!is_aligned(tensor, type)) {
        printf("  Skipping %-30s: Shape [%lld, %lld] not aligned with block size %d\n", 
               name.c_str(), tensor->ne[0], tensor->ne[1], ggml_blck_size(type));
        return false;
    }

    return true;
}


int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s input.gguf output.gguf type\n", argv[0]);
        return 1;
    }

    const char * fname_inp = argv[1];
    const char * fname_out = argv[2];
    ggml_type type = ggml_parse_type(argv[3]);

    if (type == GGML_TYPE_COUNT) {
        fprintf(stderr, "Invalid quantization type: %s\n", argv[3]);
        return 1;
    }

    struct ggml_context * ctx_data = NULL;
    struct gguf_init_params params = { .no_alloc = false, .ctx = &ctx_data };
    struct gguf_context * ctx_inp = gguf_init_from_file(fname_inp, params);
    if (!ctx_inp) return 1;

    struct gguf_context * ctx_out = gguf_init_empty();

    // 1. Correct Metadata Copy
    int n_kv = gguf_get_n_kv(ctx_inp);
    for (int i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx_inp, i);
        gguf_set_kv(ctx_out, ctx_inp); // Added the key argument
    }

    int n_tensors = gguf_get_n_tensors(ctx_inp);
    struct ggml_init_params meta_params = {
        .mem_size   = ggml_tensor_overhead() * n_tensors + (1024 * 1024), // Extra room for metadata
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx_meta = ggml_init(meta_params);
    
    std::vector<void *> allocated_data; // Tracker for cleanup

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_inp, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_data, name);

        if (should_quantize(name, tensor, type) && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16)) {
            printf("Quantizing %-40s to %s\n", name, argv[3]);

            int64_t n_elements = ggml_nelements(tensor);
            
            // --- F16 -> F32 Bridge ---
            std::vector<float> f32_buffer;
            const float * src_data = nullptr;
            if (tensor->type == GGML_TYPE_F16) {
                f32_buffer.resize(n_elements);
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)tensor->data, f32_buffer.data(), n_elements);
                src_data = f32_buffer.data();
            } else {
                src_data = (const float *)tensor->data;
            }

            // Quantize
            size_t size_new = ggml_row_size(type, n_elements);
            void * data_new = malloc(size_new);
            allocated_data.push_back(data_new);

            int64_t n_per_row = tensor->ne[0];
            int64_t n_rows = n_elements / n_per_row;
            
            ggml_quantize_chunk(type, src_data, data_new, 0, n_rows, n_per_row, nullptr);

            struct ggml_tensor * q_tensor = ggml_new_tensor(ctx_meta, type, ggml_n_dims(tensor), tensor->ne);
            ggml_set_name(q_tensor, name);
            q_tensor->data = data_new;

            // CRITICAL: Actually add the quantized tensor to the file!
            gguf_add_tensor(ctx_out, q_tensor); 
        } else {
            printf("Copying   %-40s (No Quant)\n", name);
            gguf_add_tensor(ctx_out, tensor);
        }
    }

    // Write file and clean up
    gguf_write_to_file(ctx_out, fname_out, false);
    for (void * ptr : allocated_data) free(ptr);
    ggml_free(ctx_meta);
    ggml_free(ctx_data);
    gguf_free(ctx_inp);
    gguf_free(ctx_out);
    
    return 0;
}
