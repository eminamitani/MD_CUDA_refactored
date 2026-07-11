#include <md/interactions/NNP.cuh>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <md/core/State.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/CudaCheck.cuh>
#include <md/cells/Cell.cuh>

#include <sstream>
#include <stdexcept>
#include <utility>
#include <algorithm>

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
    std::pair<torch::Tensor, torch::Tensor> unpack_nnp_output(const c10::IValue& result, int num_atoms, const char* backend_name) {
        if (!result.isTuple()) {
            throw std::runtime_error(std::string(backend_name) + " model must return a tuple: (energy, forces).");
        }

        auto result_tuple = result.toTuple();
        const auto& elements = result_tuple->elements();
        if (elements.size() < 2) {
            throw std::runtime_error(std::string(backend_name) + " model output tuple must contain energy and forces.");
        }
        if (!elements[0].isTensor() || !elements[1].isTensor()) {
            throw std::runtime_error(std::string(backend_name) + " model output elements must be tensors.");
        }

        auto energy = elements[0].toTensor().to(torch::kFloat32).contiguous().detach();
        auto forces = elements[1].toTensor().to(torch::kFloat32).contiguous().detach();

        if (!energy.is_cuda() || !forces.is_cuda()) {
            throw std::runtime_error(std::string(backend_name) + " model outputs must be CUDA tensors.");
        }
        if (energy.numel() < 1) {
            throw std::runtime_error(std::string(backend_name) + " energy output must contain at least one value.");
        }
        if (forces.numel() != static_cast<int64_t>(3) * num_atoms) {
            std::ostringstream oss;
            oss << backend_name << " force output must contain exactly 3*N values in x/y/z block layout. "
                << "Expected " << (3 * num_atoms) << ", got " << forces.numel() << ".";
            throw std::runtime_error(oss.str());
        }

        return {energy, forces};
    }
}

using namespace md::interactions;

NNP::NNP(
    State& state, 
    Cell* _cell, 
    NeighbourList* _nl, 
    float _cutoff, 
    int _num_max_edges, 
    const std::string model_path
) : cell(_cell), cutoff(_cutoff), nl(_nl), num_max_edges(_num_max_edges) {
    // モデルの読み込み
    try {
        model = torch::jit::load(model_path, torch::kCUDA);
        std::cout << "モデルを読み込みました：" << model_path << std::endl;
    }
    catch(c10::Error& e) {
        std::cerr << "モデルの読み込みに失敗しました。" << std::endl
                  << e.what() << std::endl;
        throw;
    }
    model.eval();

    auto N = state.n_atoms;

    // メモリの確保
    if (num_max_edges <= 0) {
        throw std::runtime_error("NNP max_edges must be positive.");
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

NNP::~NNP() {
    if (graph_samples > 0) {
        std::cout << "NNP edge telemetry: samples=" << graph_samples
                  << ", min=" << edge_min
                  << ", max=" << edge_max
                  << ", mean=" << static_cast<double>(edge_sum) / static_cast<double>(graph_samples)
                  << std::endl;
    }
    cudaFree(x_ptr);
    cudaFree(edge_weight_ptr);
    cudaFree(edge_index_ptr);
    cudaFree(counts);
    cudaFree(offsets);
}

void NNP::create_graph(State& state) {
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
        oss << "NNP graph has " << num_edges << " edges, exceeding max_edges=" << num_max_edges
            << ". Increase potentials.max_edges or reduce cutoff.";
        throw std::runtime_error(oss.str());
    }
    if (graph_samples == 0) {
        edge_min = num_edges;
        edge_max = num_edges;
    } else {
        edge_min = std::min(edge_min, num_edges);
        edge_max = std::max(edge_max, num_edges);
    }
    edge_sum += num_edges;
    ++graph_samples;

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

void NNP::calc_force(State& state) {
    int N = state.n_atoms;
    nl->check(state, cell);
    create_graph(state);

    auto opt = torch::TensorOptions().device(torch::kCUDA);

    x = torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64));
    edge_index = torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt64));
    edge_weight = torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32)).set_requires_grad(true);

    // ストリームを指定
    c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromExternal(state.stream, x.device().index());
    c10::cuda::CUDAStreamGuard guard(torch_stream);

    auto result_iv = model.forward({x, edge_index, edge_weight});
    auto [energy, forces] = unpack_nnp_output(result_iv, N, "NNP");

    // libtorch側のポインター
    float* forces_ptr = forces.data_ptr<float>();

    // 値のコピー
    MD_CUDA_CHECK(cudaMemcpyAsync(state.force.x, forces_ptr, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream));
    MD_CUDA_CHECK(cudaMemcpyAsync(state.force.y, forces_ptr + N, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream));
    MD_CUDA_CHECK(cudaMemcpyAsync(state.force.z, forces_ptr + 2 * N, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream));
}

void NNP::calc_potential(State& state) {
    nl->check(state, cell);
    create_graph(state);

    int N = state.n_atoms;

    auto opt = torch::TensorOptions().device(torch::kCUDA);

    x = torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64));
    edge_index = torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt64));
    edge_weight = torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32)).set_requires_grad(true);

    c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromExternal(state.stream, x.device().index());
    c10::cuda::CUDAStreamGuard guard(torch_stream);

    auto result_iv = model.forward({x, edge_index, edge_weight});
    auto [energy, forces] = unpack_nnp_output(result_iv, N, "NNP");

    float* energy_ptr = energy.data_ptr<float>();

    MD_CUDA_CHECK(cudaMemcpy(&state.potential_energy, energy_ptr, sizeof(float), cudaMemcpyDeviceToHost));
}
