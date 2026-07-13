#include <md/interactions/NNP_CSR.cuh>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <md/core/State.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/CudaCheck.cuh>
#include <md/cells/Cell.cuh>

#include <sstream>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <cstdio>

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

            if (idx == j) continue;

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
        const int64_t* __restrict__ offsets, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        void (*apply_pbc_ptr) (float*, float*, float*, float*), 
        float* lattice, 
        const float cutoff, 
        const int num_atoms, 
        const int max_neighbours, 
        const int num_max_edges
    ) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_atoms) return;

        const float pxi = pos.x[idx];
        const float pyi = pos.y[idx];
        const float pzi = pos.z[idx];

        int write_idx = offsets[idx];

        for (int c = 0; c < count[idx]; c ++) {
            int j = list[idx * max_neighbours + c];
            if (idx == j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];

            float dx, dy, dz;
            // 常にインデックスが小さい方を基準に計算
            if (idx < j) {
                dx = pxj - pxi;
                dy = pyj - pyi;
                dz = pzj - pzi;
                apply_pbc_ptr(&dx, &dy, &dz, lattice);
            } else {
                dx = pxi - pxj;
                dy = pyi - pyj;
                dz = pzi - pzj;
                apply_pbc_ptr(&dx, &dy, &dz, lattice);
                // 向きを合わせるために反転
                dx = -dx;
                dy = -dy;
                dz = -dz;
            }
            
            const float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq < cutoff * cutoff) {
                if (write_idx >= num_max_edges) return;
                edge_index_ptr[write_idx] = idx;
                edge_index_ptr[num_max_edges + write_idx] = j;

                edge_weight_ptr[write_idx] = dx;
                edge_weight_ptr[num_max_edges + write_idx] = dy;
                edge_weight_ptr[2 * num_max_edges + write_idx] = dz;

                write_idx ++;
            }
        }
    }

    __global__ void append_total_sum_kernel(
        const int* src_array, 
        int64_t* dst_array, 
        int N
    ) {
        dst_array[N] = dst_array[N - 1] + (int64_t)src_array[N - 1];
    }

    __global__ void padding_kernel(
        int64_t* __restrict__ edge_index_ptr, 
        float* __restrict__ edge_weight_ptr, 
        const int64_t* __restrict__ total_edges, 
        const int num_nodes, 
        const int num_max_edges
    ){
        const int64_t num_edges = total_edges[0];
        const int idx = threadIdx.x + blockDim.x * blockIdx.x;
        const int pad_idx = num_edges + idx;

        if (pad_idx >= num_max_edges) return;

        // Spread zero-contribution self edges across atoms. Sending every
        // padding edge to atom 0 creates a severe scatter_add atomic hotspot
        // in PaiNN even though the cutoff envelope makes each message zero.
        const int padding_node = idx % num_nodes;
        edge_index_ptr[pad_idx] = padding_node;
        edge_index_ptr[num_max_edges + pad_idx] = padding_node;

        edge_weight_ptr[pad_idx] = 1e+5f;
        edge_weight_ptr[num_max_edges + pad_idx] = 0.0f;
        edge_weight_ptr[2 * num_max_edges + pad_idx] = 0.0f;
    }

    __global__ void guard_edge_capacity_kernel(
        const int64_t* total_edges,
        const int num_max_edges
    ) {
        if (threadIdx.x != 0 || blockIdx.x != 0) return;
        if (total_edges[0] > static_cast<int64_t>(num_max_edges)) {
            printf(
                "NNP_fixed edge capacity exceeded: edges=%lld max_edges=%d\n",
                static_cast<long long>(total_edges[0]),
                num_max_edges
            );
            assert(total_edges[0] <= static_cast<int64_t>(num_max_edges));
        }
    }
    std::pair<torch::Tensor, torch::Tensor> unpack_nnp_csr_output(const c10::IValue& result, int num_atoms, const char* backend_name) {
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

NNP_CSR::NNP_CSR(
    State& state, 
    Cell* _cell, 
    NeighbourList* _nl, 
    float _cutoff, 
    int _num_max_edges, 
    const std::string model_path,
    bool _fixed_shape_no_sync
) : cell(_cell), cutoff(_cutoff), nl(_nl), num_max_edges(_num_max_edges), fixed_shape_no_sync(_fixed_shape_no_sync) {
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
        throw std::runtime_error("NNP_CSR max_edges must be positive.");
    }

    MD_CUDA_CHECK(cudaMalloc(&x_ptr, N * sizeof(int64_t)));
    MD_CUDA_CHECK(cudaMalloc(&edge_index_ptr, 2 * num_max_edges * sizeof(int64_t)));
    MD_CUDA_CHECK(cudaMalloc(&edge_weight_ptr, 3 * num_max_edges * sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&offsets_ptr, (N + 1) * sizeof(int64_t)));
    MD_CUDA_CHECK(cudaMalloc(&counts, N * sizeof(int)));

    // 原子番号はシミュレーションを通して変わらないため、最初に初期化する
    // int32_t -> int64_t
    thrust::copy(
        thrust::device, 
        state.atomic_numbers, 
        state.atomic_numbers + N, 
        x_ptr
    );

    // torch::Tensorをメモリのビューとして作成
    int current_device;
    MD_CUDA_CHECK(cudaGetDevice(&current_device));
    auto opt = torch::TensorOptions().device(torch::Device(torch::kCUDA, current_device));

    x = torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64));
    edge_index = torch::from_blob(edge_index_ptr, {2, num_max_edges}, opt.dtype(torch::kInt64));
    edge_weight = torch::from_blob(edge_weight_ptr, {3, num_max_edges}, opt.dtype(torch::kFloat32)).set_requires_grad(true);
    offsets = torch::from_blob(offsets_ptr, {N + 1}, opt.dtype(torch::kInt64));

    // cubのバッファを確保
    MD_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        d_temp_storage, 
        temp_storage_bytes, 
        counts, 
        offsets_ptr, 
        N, 
        state.stream
    ));
    MD_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}

NNP_CSR::~NNP_CSR() {
    cudaFree(x_ptr);
    cudaFree(edge_index_ptr);
    cudaFree(offsets_ptr);
    cudaFree(edge_weight_ptr);
    cudaFree(counts);
    cudaFree(d_temp_storage);
}

void NNP_CSR::create_graph(State& state) {
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
    MD_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        d_temp_storage, 
        temp_storage_bytes, 
        counts, 
        offsets_ptr, 
        N, 
        state.stream
    ));
    // offsets_ptr[N]にnum_edgesを書き込む
    append_total_sum_kernel<<<1, 1, 0, state.stream>>>(
        counts, 
        offsets_ptr, 
        N
    );
    MD_CUDA_KERNEL_CHECK();

    if (fixed_shape_no_sync) {
        if (!initial_edge_count_checked) {
            int64_t h_num_edges = 0;
            MD_CUDA_CHECK(cudaMemcpyAsync(&h_num_edges, offsets_ptr + N, sizeof(int64_t), cudaMemcpyDeviceToHost, state.stream));
            MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
            if (h_num_edges > num_max_edges) {
                std::ostringstream oss;
                oss << "NNP_fixed initial graph has " << h_num_edges << " edges, exceeding max_edges=" << num_max_edges
                    << ". Increase potentials.max_edges or reduce cutoff.";
                throw std::runtime_error(oss.str());
            }
            std::cout << "NNP_fixed initial edges=" << h_num_edges
                      << ", max_edges=" << num_max_edges << std::endl;
            initial_edge_count_checked = true;
        }
        guard_edge_capacity_kernel<<<1, 1, 0, state.stream>>>(offsets_ptr + N, num_max_edges);
        MD_CUDA_KERNEL_CHECK();
        num_edges = num_max_edges;
    } else {
        int64_t h_num_edges = 0;
        int neighbour_overflow_count = 0;
        MD_CUDA_CHECK(cudaMemcpyAsync(&h_num_edges, offsets_ptr + N, sizeof(int64_t), cudaMemcpyDeviceToHost, state.stream));
        nl->enqueue_overflow_count_copy(state, &neighbour_overflow_count);
        MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
        nl->validate_overflow_count(neighbour_overflow_count, "NNP_CSR graph update");
        if (h_num_edges > num_max_edges) {
            std::ostringstream oss;
            oss << "NNP_CSR graph has " << h_num_edges << " edges, exceeding max_edges=" << num_max_edges
                << ". Increase potentials.max_edges or reduce cutoff.";
            throw std::runtime_error(oss.str());
        }
        num_edges = static_cast<int>(h_num_edges);
    }

    build_graph_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        state.pos, 
        edge_index_ptr, 
        edge_weight_ptr, 
        offsets_ptr, 
        nl->get_list(), 
        nl->get_count(), 
        cell->apply_pbc_ptr, 
        cell->d_lattice, 
        cutoff, 
        N, 
        nl->get_max_neighbours(), 
        num_max_edges
    );
    MD_CUDA_KERNEL_CHECK();

    int num_blocks_edges = (num_max_edges + num_threads - 1) / num_threads;
    padding_kernel<<<num_blocks_edges, num_threads, 0, state.stream>>>(
        edge_index_ptr, 
        edge_weight_ptr, 
        offsets_ptr + N, 
        N, 
        num_max_edges
    );
    MD_CUDA_KERNEL_CHECK();
}

void NNP_CSR::calc_force(State& state) {
    int N = state.n_atoms;
    if (fixed_shape_no_sync) nl->check(state, cell);
    else nl->check_deferred(state, cell);
    create_graph(state);

    // ストリームを指定
    c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromExternal(state.stream, x.device().index());
    c10::cuda::CUDAStreamGuard guard(torch_stream);

    auto result_iv = fixed_shape_no_sync
        ? model.forward({x, edge_index, edge_weight})
        : model.forward({x, edge_index, edge_weight, offsets});
    auto [energy, forces] = unpack_nnp_csr_output(result_iv, N, "NNP_CSR");

    // libtorch側のポインター
    float* forces_ptr = forces.data_ptr<float>();
    float* energy_ptr = energy.data_ptr<float>();

    MD_CUDA_CHECK(cudaMemcpyAsync(
        state.cached_potential_energy,
        energy_ptr,
        sizeof(float),
        cudaMemcpyDeviceToDevice,
        state.stream
    ));
    state.cached_potential_energy_valid = true;

    // 値のコピー
    MD_CUDA_CHECK(cudaMemcpyAsync(state.force.x, forces_ptr, 3 * N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream));
}

void NNP_CSR::calc_potential(State& state) {
    if (state.cached_potential_energy_valid) {
        MD_CUDA_CHECK(cudaMemcpyAsync(
            &state.potential_energy,
            state.cached_potential_energy,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            state.stream
        ));
        MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
        return;
    }
    if (fixed_shape_no_sync) nl->check(state, cell);
    else nl->check_deferred(state, cell);
    create_graph(state);

    // ストリームを指定
    c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromExternal(state.stream, x.device().index());
    c10::cuda::CUDAStreamGuard guard(torch_stream);

    auto result_iv = fixed_shape_no_sync
        ? model.forward({x, edge_index, edge_weight})
        : model.forward({x, edge_index, edge_weight, offsets});
    auto [energy, forces] = unpack_nnp_csr_output(result_iv, state.n_atoms, "NNP_CSR");

    float* energy_ptr = energy.data_ptr<float>();

    MD_CUDA_CHECK(cudaMemcpyAsync(
        state.cached_potential_energy,
        energy_ptr,
        sizeof(float),
        cudaMemcpyDeviceToDevice,
        state.stream
    ));
    state.cached_potential_energy_valid = true;
    MD_CUDA_CHECK(cudaMemcpyAsync(
        &state.potential_energy,
        state.cached_potential_energy,
        sizeof(float),
        cudaMemcpyDeviceToHost,
        state.stream
    ));
    MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
}
