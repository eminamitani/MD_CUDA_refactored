#include <md/utils/NeighbourList.cuh>
#include <md/utils/CudaCheck.cuh>
#include <md/cells/Cell.cuh>
#include <md/core/State.cuh>

#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <stdexcept>
#include <string>

using Top2 = md::Top2;

namespace {
    __global__ void generate_nl(
        bool* flag, 
        dfloat3 pos, 
        dfloat3 nl_conf, 
        int num_atoms, 
        int max_neighbours, 
        int* list, 
        int* count, 
        int* overflow_count,
        float cutoff_margin_sq, 
        void (*apply_pbc_ptr) (float*, float*, float*, float*), 
        float* lattice
    ) {
        if (!*flag) return;

        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int warp_id = tid / 32;
        int lane_id = tid % 32;

        int i = warp_id;

        if (i < num_atoms) {
            auto pxi = pos.x[i];
            auto pyi = pos.y[i];
            auto pzi = pos.z[i];

	            int c = 0;
                bool overflow_reported = false;

	            for (int j = 0; j < num_atoms; j += 32) {
                int j_curr = j + lane_id;
                bool is_neighbour = false;

                if (j_curr < num_atoms && i != j_curr) {
                    auto pxj = pos.x[j_curr];
                    auto pyj = pos.y[j_curr];
                    auto pzj = pos.z[j_curr];

                    // 距離の計算
                    auto dx = pxi - pxj;
                    auto dy = pyi - pyj;
                    auto dz = pzi - pzj;
                    
                    apply_pbc_ptr(&dx, &dy, &dz, lattice);

                    const auto dist_sq = dx * dx + dy * dy + dz * dz;

                    if (dist_sq < cutoff_margin_sq) {
                        is_neighbour = true;
                    }
                }
            
	                unsigned int mask = __ballot_sync(0xffffffff, is_neighbour);
                    int matched_count = __popc(mask);

                    if (lane_id == 0 && !overflow_reported && c + matched_count > max_neighbours) {
                        atomicAdd(overflow_count, 1);
                        overflow_reported = true;
                    }

	                if (is_neighbour) {
	                    int offset = __popc(mask & ((1u << lane_id) - 1));
                        int write_idx = c + offset;
                        if (write_idx < max_neighbours) {
	                        list[i * max_neighbours + write_idx] = j_curr;
                        }
	                }

	                c += matched_count;
	            }

	            if (lane_id == 0) {
	                count[i] = c > max_neighbours ? max_neighbours : c;
	                nl_conf.x[i] = pxi;
	                nl_conf.y[i] = pyi;
	                nl_conf.z[i] = pzi;
            }
        }
    }

    __global__ void check_top2(
        Top2* top2, 
        bool* flag, 
        float margin_sq
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            Top2 t = *top2;
            if (t.max1 + t.max2 + 2 * sqrtf(t.max1 * t.max2) > margin_sq) *flag = true;
        }
    }

    struct CalcDist {
        dfloat3 pos;
        dfloat3 nl_conf;

        void (*apply_pbc_ptr) (float*, float*, float*, float*);
        float* lattice;

        CalcDist(
            dfloat3 _pos, 
            dfloat3 _nl_conf, 
            void (*_apply_pbc_ptr) (float*, float*, float*, float*), 
            float* _lattice
        ) : pos(_pos), nl_conf(_nl_conf), apply_pbc_ptr(_apply_pbc_ptr), lattice(_lattice) {}

        __host__ __device__ Top2 operator () (const int idx) const {
            auto dx = pos.x[idx] - nl_conf.x[idx];
            auto dy = pos.y[idx] - nl_conf.y[idx];
            auto dz = pos.z[idx] - nl_conf.z[idx];

            // PBC補正
            apply_pbc_ptr(&dx, &dy, &dz, lattice);

            float dist_sq = dx * dx + dy * dy + dz * dz;

            return Top2(dist_sq);
        }
    };

    // 2つのTop2オブジェクトから新たな一つのTop2オブジェクトを作成
    struct MergeTop2 {
        __host__ __device__ Top2 operator () (const Top2& a, const Top2& b) const {
            float max1 = fmaxf(a.max1, b.max1);
            float max2 = fmaxf(fminf(a.max1, b.max1), fmaxf(a.max2, b.max2));
            return Top2(max1, max2);        
        }
    };
}

using namespace md;

NeighbourList::NeighbourList(State& state, float _cutoff, float _margin, int _max_neighbours)
    : cutoff(_cutoff), margin(_margin), max_neighbours(_max_neighbours) {
    if (max_neighbours <= 0) {
        throw std::runtime_error("max_neighbours must be positive.");
    }

    auto N = state.n_atoms;
    MD_CUDA_CHECK(cudaMalloc(&this->list, (size_t)N * max_neighbours * sizeof(int)));
    MD_CUDA_CHECK(cudaMalloc(&this->count, N * sizeof(int)));
    MD_CUDA_CHECK(cudaMalloc(&this->nl_conf.x, N * sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&this->nl_conf.y, N * sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&this->nl_conf.z, N * sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&this->top2, sizeof(Top2)));
    MD_CUDA_CHECK(cudaMalloc(&this->flag, sizeof(bool)));
    MD_CUDA_CHECK(cudaMalloc(&this->overflow_count, sizeof(int)));
    MD_CUDA_CHECK(cudaMemset(this->flag, 1, sizeof(bool)));
    MD_CUDA_CHECK(cudaMemset(this->overflow_count, 0, sizeof(int)));

    int minGridSize;
    MD_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &generate_nl_num_threads, generate_nl, 0, 0));
}

NeighbourList::~NeighbourList() {
    cudaFree(this->list);
    cudaFree(this->count);
    cudaFree(this->nl_conf.x);
    cudaFree(this->nl_conf.y);
    cudaFree(this->nl_conf.z);
    cudaFree(this->top2);
    cudaFree(this->flag);
    cudaFree(this->overflow_count);
    cudaFree(this->d_temp_storage);
}

void NeighbourList::throw_if_overflow(State& state, const char* context) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    MD_CUDA_CHECK(cudaStreamIsCapturing(state.stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return;
    }

    int h_overflow_count = 0;
    MD_CUDA_CHECK(cudaMemcpyAsync(&h_overflow_count, this->overflow_count, sizeof(int), cudaMemcpyDeviceToHost, state.stream));
    MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
    if (h_overflow_count > 0) {
        throw std::runtime_error(
            std::string("NeighbourList overflow in ") + context +
            ": at least " + std::to_string(h_overflow_count) +
            " atoms exceeded max_neighbours=" + std::to_string(max_neighbours) +
            ". Increase neighbour_list.max_neighbours or reduce cutoff/margin."
        );
    }
}

void NeighbourList::generate(State& state, Cell* cell) {
    auto N = state.n_atoms;
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

    // NLの作成
    int warps_per_block = generate_nl_num_threads / 32;

    int num_blocks = (N + warps_per_block - 1) / warps_per_block;
    MD_CUDA_CHECK(cudaMemsetAsync(this->overflow_count, 0, sizeof(int), state.stream));
    generate_nl<<<num_blocks, generate_nl_num_threads, 0, state.stream>>>(
        this->flag, 
        state.pos, 
        this->nl_conf, 
        N, 
        this->max_neighbours, 
        this->list, 
        this->count, 
        this->overflow_count,
        cutoff_margin_sq, 
        cell->apply_pbc_ptr, 
        cell->d_lattice
    );
    MD_CUDA_KERNEL_CHECK();
    throw_if_overflow(state, "generate");

    MD_CUDA_CHECK(cudaMemsetAsync(this->flag, 0, sizeof(bool), state.stream));

    // バッファの確保
    CalcDist op(
        state.pos, 
        this->nl_conf, 
        cell->apply_pbc_ptr, 
        cell->d_lattice
    );
    thrust::counting_iterator<int> count_itr(0);
    auto trans_itr = thrust::make_transform_iterator(count_itr, op);

    MD_CUDA_CHECK(cub::DeviceReduce::Reduce(
        this->d_temp_storage, 
        this->temp_storage_bytes, 
        trans_itr, 
        this->top2, 
        N, 
        MergeTop2(), 
        Top2()
    ));

    MD_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}

void NeighbourList::check(State& state, Cell* cell) {
    auto N = state.n_atoms;
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

    // 移動距離の大きい順に2粒子の移動距離を表すTop2オブジェクトを計算
    CalcDist op(
        state.pos, 
        this->nl_conf, 
        cell->apply_pbc_ptr, 
        cell->d_lattice
    );
    thrust::counting_iterator<int> count_itr(0);
    auto trans_itr = thrust::make_transform_iterator(count_itr, op);

    MD_CUDA_CHECK(cub::DeviceReduce::Reduce(
        this->d_temp_storage, 
        this->temp_storage_bytes, 
        trans_itr, 
        this->top2, 
        N, 
        MergeTop2(), 
        Top2(), 
        state.stream
    ));

    // Top2オブジェクトの移動距離が閾値を超えているか判定し、超えていたらGenerate()を呼ぶ
    check_top2<<<1, 1, 0, state.stream>>>(
        this->top2, 
        this->flag, 
        margin * margin
    );
    MD_CUDA_KERNEL_CHECK();

    int warps_per_block = generate_nl_num_threads / 32;

    int num_blocks = (N + warps_per_block - 1) / warps_per_block;

    MD_CUDA_CHECK(cudaMemsetAsync(this->overflow_count, 0, sizeof(int), state.stream));
    generate_nl<<<num_blocks, generate_nl_num_threads, 0, state.stream>>>(
        this->flag, 
        state.pos, 
        this->nl_conf, 
        N, 
        this->max_neighbours, 
        this->list, 
        this->count, 
        this->overflow_count,
        cutoff_margin_sq, 
        cell->apply_pbc_ptr, 
        cell->d_lattice
    );
    MD_CUDA_KERNEL_CHECK();
    throw_if_overflow(state, "check");

    MD_CUDA_CHECK(cudaMemsetAsync(this->flag, 0, sizeof(bool), state.stream));
}
