#include <md/interactions/NNP_aoti.cuh>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <md/core/State.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/CudaCheck.cuh>
#include <md/cells/Cell.cuh>

#include <sstream>
#include <stdexcept>

// #include <torch_tensorrt/torch_tensorrt.h>

namespace {
    __global__ void count_pairs_kernel(
        dfloat3 pos, 
        int* __restrict__ counts, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        void (*apply_pbc_ptr) (float*, float*, float*, float*), 
        float* lattice, 
        const float cutoff, 
        const int num_atoms, 
        const int max_neighbours
    ) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_atoms) return;

        const float pxi = pos.x[idx];
        const float pyi = pos.y[idx];
        const float pzi = pos.z[idx];

        int valid_pairs = 0;
        for (int c = 0; c < count[idx]; c ++) {
            int j = list[idx * max_neighbours + c];

            if (idx >= j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];
            
            float dx = pxi - pxj;
            float dy = pyi - pyj;
            float dz = pzi - pzj;
        
            apply_pbc_ptr(&dx, &dy, &dz, lattice);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                valid_pairs++;
            }
        }
        counts[idx] = valid_pairs;
    }

    __global__ void build_graph_kernel(
        dfloat3 pos, 
        int64_t* __restrict__ edge_index_ptr, 
        float* __restrict__ edge_weight_ptr, 
        const int* __restrict__ offsets,
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        void (*apply_pbc_ptr) (float*, float*, float*, float*), 
        float* lattice, 
        const float cutoff, 
        const int num_atoms, 
        const int max_neighbours, 
        const int num_edges,
        const int num_pairs
    ) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_atoms) return;

        const float pxi = pos.x[idx];
        const float pyi = pos.y[idx];
        const float pzi = pos.z[idx];

        int write_idx = offsets[idx];

        for (int c = 0; c < count[idx]; c ++) {
            int j = list[idx * max_neighbours + c];
            if (idx >= j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];

            float dx = pxj - pxi;
            float dy = pyj - pyi;
            float dz = pzj - pzi;
        
            apply_pbc_ptr(&dx, &dy, &dz, lattice);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                // i -> j
                edge_index_ptr[write_idx] = idx;
                edge_index_ptr[num_edges + write_idx] = j;

                edge_weight_ptr[write_idx] = dx;
                edge_weight_ptr[num_edges + write_idx] = dy;
                edge_weight_ptr[2 * num_edges + write_idx] = dz;

                // j -> iへコピー
                int rev_idx = write_idx + num_pairs;
                edge_index_ptr[rev_idx] = j;
                edge_index_ptr[num_edges + rev_idx] = idx;

                edge_weight_ptr[rev_idx] = -dx;
                edge_weight_ptr[num_edges + rev_idx] = -dy;
                edge_weight_ptr[2 * num_edges + rev_idx] = -dz;

                write_idx ++;
            }
        }
    }
    void validate_aoti_outputs(const std::vector<torch::Tensor>& outputs, int num_atoms) {
        if (outputs.size() < 2) {
            throw std::runtime_error("NNP_aoti model output must contain energy and forces.");
        }
        if (!outputs[0].is_cuda() || !outputs[1].is_cuda()) {
            throw std::runtime_error("NNP_aoti model outputs must be CUDA tensors.");
        }
        if (outputs[0].numel() < 1) {
            throw std::runtime_error("NNP_aoti energy output must contain at least one value.");
        }
        if (outputs[1].numel() != static_cast<int64_t>(3) * num_atoms) {
            std::ostringstream oss;
            oss << "NNP_aoti force output must contain exactly 3*N values in x/y/z block layout. "
                << "Expected " << (3 * num_atoms) << ", got " << outputs[1].numel() << ".";
            throw std::runtime_error(oss.str());
        }
    }
}

using namespace md::interactions;

NNP_aoti::NNP_aoti(
    State& state, 
    Cell* _cell, 
    NeighbourList* _nl, 
    float _cutoff, 
    int _num_max_edges, 
    const std::string model_path
) : cell(_cell), cutoff(_cutoff), nl(_nl), num_max_edges(_num_max_edges), loader(model_path) {
    auto N = state.n_atoms;

    // メモリの確保
    if (num_max_edges <= 0) {
        throw std::runtime_error("NNP_aoti max_edges must be positive.");
    }

    MD_CUDA_CHECK(cudaMalloc(&x_ptr, N * sizeof(int64_t)));
    MD_CUDA_CHECK(cudaMalloc(&edge_weight_ptr, 3 * num_max_edges * sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&edge_index_ptr, 2 * num_max_edges * sizeof(int64_t)));
    MD_CUDA_CHECK(cudaMalloc(&counts, N * sizeof(int)));
    MD_CUDA_CHECK(cudaMalloc(&offsets, N * sizeof(int)));

    // 原子番号はシミュレーションを通して変わらないため、最初に初期化する
    // int32_t -> int64_t
    thrust::copy(
        thrust::device, 
        state.atomic_numbers, 
        state.atomic_numbers + N, 
        x_ptr
    );
}

NNP_aoti::~NNP_aoti() {
    cudaFree(x_ptr);
    cudaFree(edge_weight_ptr);
    cudaFree(edge_index_ptr);
    cudaFree(counts);
    cudaFree(offsets);
}

void NNP_aoti::create_graph(State& state) {
    int N = state.n_atoms;

    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;

    count_pairs_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        state.pos, 
        counts, 
        nl->get_list(), 
        nl->get_count(), 
        cell->apply_pbc_ptr, 
        cell->d_lattice, 
        cutoff, 
        N, 
        nl->get_max_neighbours()
    );
    MD_CUDA_KERNEL_CHECK();

    // 手前のインデックスまでを加算
    thrust::exclusive_scan(
        thrust::cuda::par.on(state.stream),
        counts,
        counts + N,
        offsets
    );

    int last_count, last_offset;
    MD_CUDA_CHECK(cudaMemcpyAsync(&last_count, counts + N - 1, sizeof(int), cudaMemcpyDeviceToHost, state.stream));
    MD_CUDA_CHECK(cudaMemcpyAsync(&last_offset, offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost, state.stream));

    MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));

    int num_pairs = last_offset + last_count;
    num_edges = 2 * num_pairs;
    if (num_edges > num_max_edges) {
        std::ostringstream oss;
        oss << "NNP_aoti graph has " << num_edges << " edges, exceeding max_edges=" << num_max_edges
            << ". Increase potentials.max_edges or reduce cutoff.";
        throw std::runtime_error(oss.str());
    }

    build_graph_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        state.pos, 
        edge_index_ptr, 
        edge_weight_ptr, 
        offsets, 
        nl->get_list(), 
        nl->get_count(), 
        cell->apply_pbc_ptr, 
        cell->d_lattice, 
        cutoff, 
        N, 
        nl->get_max_neighbours(), 
        num_edges, 
        num_pairs
    );
    MD_CUDA_KERNEL_CHECK();
}

void NNP_aoti::calc_force(State& state) {
    int N = state.n_atoms;

    nl->check(state, cell);
    create_graph(state);

    auto opt = torch::TensorOptions().device(torch::kCUDA);
    inputs = {
        torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32))
    };

    c10::InferenceMode mode;
    int current_device;
    MD_CUDA_CHECK(cudaGetDevice(&current_device));
    c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(state.stream, current_device)
    );
    auto outputs = loader.run(inputs);
    validate_aoti_outputs(outputs, N);

    auto forces = outputs[1].to(torch::kFloat32).contiguous();
    float* force_ptr = forces.data_ptr<float>();

    MD_CUDA_CHECK(cudaMemcpyAsync(state.force.x, force_ptr, 3 * N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream));
}

void NNP_aoti::calc_potential(State& state) {
    nl->check(state, cell);
    create_graph(state);

    int N = state.n_atoms;

    auto opt = torch::TensorOptions().device(torch::kCUDA);
    inputs = {
        torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32))
    };

    c10::InferenceMode mode;
    int current_device;
    MD_CUDA_CHECK(cudaGetDevice(&current_device));
    c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(state.stream, current_device)
    );

    auto outputs = loader.run(inputs);
    validate_aoti_outputs(outputs, N);

    auto energy = outputs[0].to(torch::kFloat32).contiguous();
    float* energy_ptr = energy.data_ptr<float>();

    MD_CUDA_CHECK(cudaMemcpy(&state.potential_energy, energy_ptr, sizeof(float), cudaMemcpyDeviceToHost));
}
