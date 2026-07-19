#include <md/thermostats/NHC1.cuh>

#include <md/core/State.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/thermostats/KinEnergyCalculator.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
    __global__ void update_mass (
        float *mass, 
        float tau, 
        float boltzmann_constant, 
        float dof
    ) {
        *mass = tau * tau * boltzmann_constant * md::c_target_temperature * dof;
    }

    __global__ void calc_scaling_factor (
        float *kinetic_energy, 
        ChainState c_state, 
        float dt, 
        float dof, 
        float boltzmann_constant
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            auto AKIN = 2.0f * *kinetic_energy;
            auto dt_half = 0.5f * dt;
            auto dt_quarter = 0.25f * dt;

            auto f0 = *c_state.force;
            auto v0 = *c_state.vel;
            auto p0 = *c_state.pos;
            auto m0_inv = 1.0f / *c_state.mass;

            auto targ_kin = md::c_target_temperature * dof * boltzmann_constant;

            // 逆順の更新
            f0 = (AKIN - targ_kin) * m0_inv;
            v0 += dt_quarter * f0;

            // スケーリング
            float sf = exp(-dt_half * v0);
            AKIN *= exp(-dt * v0);

            // 変位の更新
            p0 += dt_half * v0;

            // 順方向の更新
            f0 = (AKIN - targ_kin) * m0_inv;
            v0 += dt_quarter * f0;

            // グローバルメモリに書き込み
            *c_state.force = f0;
            *c_state.vel = v0;
            *c_state.pos = p0;
            *c_state.scaling_factor = sf;
        }
    }

    struct Scaling {
        dfloat3 vel;
        float* scaling_factor;
        Scaling(dfloat3 _vel, float* _sf) : vel(_vel), scaling_factor(_sf) {}
        __device__ void operator() (int idx) {
            auto sf = *scaling_factor;
            vel.x[idx] *= sf;
            vel.y[idx] *= sf;
            vel.z[idx] *= sf;
        }
    };
}

using namespace md::thermostats;

NHC1::NHC1(const float _tau, TemperatureScheduler *_scheduler, int _configured_dof)
 :  tau(_tau), configured_dof(_configured_dof), scheduler(_scheduler) {
    cudaMalloc(&c_state.pos, sizeof(float));
    cudaMalloc(&c_state.vel, sizeof(float));
    cudaMalloc(&c_state.force, sizeof(float));
    cudaMalloc(&c_state.mass, sizeof(float));
    cudaMalloc(&c_state.scaling_factor, sizeof(float));
}

NHC1::~NHC1() {
    cudaFree(c_state.pos);
    cudaFree(c_state.vel);
    cudaFree(c_state.force);
    cudaFree(c_state.mass);
    cudaFree(c_state.scaling_factor);
}

void NHC1::init(State& state) {
    this->dof = static_cast<float>(configured_dof > 0 ? configured_dof : state.thermostat_dof);
    if (this->dof <= 0.0f) {
        this->dof = 3.0f * state.n_atoms;
    }
    this->calculator = std::make_unique<KinEnergyCalculator>(state);

    float zero = 0.0f;
    cudaMemcpy(c_state.pos, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.vel, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.force, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.mass, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.scaling_factor, &zero, sizeof(float), cudaMemcpyHostToDevice);
}

void NHC1::stepOne(State& state) {
    op(state);
}

void NHC1::stepTwo(State& state) {
    op(state);
}

void NHC1::op(State& state) {
    auto N = state.n_atoms;
    this->scheduler->get_temperature(state);
    update_mass<<<1, 1, 0, state.stream>>>(
        c_state.mass, 
        tau, 
        boltzmann_constant, 
        dof
    );

    calculator->calc_kinetic_energy(state);
    
    calc_scaling_factor<<<1, 1, 0, state.stream>>>(
        state.kinetic_energy, 
        this->c_state, 
        state.dt, 
        dof, 
        boltzmann_constant
    );

    thrust::for_each(
        thrust::cuda::par_nosync.on(state.stream),  
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Scaling(
            state.vel, 
            c_state.scaling_factor
        )
    );
}

md::CheckpointBytes NHC1::save_checkpoint(State& state) const {
    std::array<float, 6> values{};
    cudaStreamSynchronize(state.stream);
    cudaMemcpy(&values[0], c_state.pos, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&values[1], c_state.vel, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&values[2], c_state.force, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&values[3], c_state.mass, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&values[4], c_state.scaling_factor, sizeof(float), cudaMemcpyDeviceToHost);
    values[5] = dof;
    md::CheckpointBytes data(sizeof(values));
    std::memcpy(data.data(), values.data(), sizeof(values));
    return data;
}

void NHC1::load_checkpoint(State& state, const md::CheckpointBytes& data) {
    std::array<float, 6> values{};
    if (data.size() != sizeof(values)) {
        throw std::runtime_error("Invalid NHC1 checkpoint payload size.");
    }
    std::memcpy(values.data(), data.data(), sizeof(values));
    dof = values[5];
    cudaMemcpy(c_state.pos, &values[0], sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.vel, &values[1], sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.force, &values[2], sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.mass, &values[3], sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.scaling_factor, &values[4], sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(state.stream);
}
