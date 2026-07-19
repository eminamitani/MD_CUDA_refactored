#include <md/integrators/LangevinIntegrator.cuh>

#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/thermostats/Thermostat.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>

#include <thrust/execution_policy.h>

#include <cstring>
#include <stdexcept>

namespace {
    struct InitCurand {
        curandState* states;
        unsigned long long seed;

        InitCurand(curandState* _states, unsigned long long _seed) 
            : states(_states), seed(_seed) {}

        __device__ void operator() (const int idx) {
            curand_init(seed, idx, 0, &states[idx]);
        }
    };

    struct StepOne {
        const float dt_half, dt_half_conv, c1, boltzmann_constant;
        dfloat3 pos, vel;
        const dfloat3 force;
        float* __restrict__ mass; 
        float* __restrict__ mass_inv; 
        curandState* states;

        StepOne(
            dfloat3 _pos, 
            dfloat3 _vel, 
            dfloat3 _force, 
            float* _mass, 
            float* _mass_inv, 
            curandState* _states, 
            float _dt_half, 
            float _dt_half_conv, 
            float _c1, 
            float _boltzmann_constant
        ) : pos(_pos), 
            vel(_vel), 
            force(_force), 
            mass(_mass), 
            mass_inv(_mass_inv), 
            states(_states), 
            dt_half(_dt_half), 
            dt_half_conv(_dt_half_conv), 
            c1(_c1), 
            boltzmann_constant(_boltzmann_constant) {}
        
        __device__ void operator() (const int idx) {
            auto cs = states[idx];
            const auto m = mass[idx];
            const auto mi = mass_inv[idx];
            const float mass_sq_inv = rsqrtf(m);
            const float c3 = sqrtf(boltzmann_constant * md::c_target_temperature * (1.0f - c1 * c1));

            // 速度の更新1
            auto vx = vel.x[idx] + force.x[idx] * mi * dt_half_conv;
            auto vy = vel.y[idx] + force.y[idx] * mi * dt_half_conv;
            auto vz = vel.z[idx] + force.z[idx] * mi * dt_half_conv;

            // 位置の更新1
            pos.x[idx] += vx * dt_half;
            pos.y[idx] += vy * dt_half;
            pos.z[idx] += vz * dt_half;

            // 乱数の生成
            float rx = curand_normal(&cs);
            float ry = curand_normal(&cs);
            float rz = curand_normal(&cs);

            // 速度の更新2
            vx = c1 * vx + c3 * mass_sq_inv * rx;
            vy = c1 * vy + c3 * mass_sq_inv * ry;
            vz = c1 * vz + c3 * mass_sq_inv * rz;
        
            vel.x[idx] = vx;
            vel.y[idx] = vy;
            vel.z[idx] = vz;

            // 位置の更新2
            pos.x[idx] += vx * dt_half;
            pos.y[idx] += vy * dt_half;
            pos.z[idx] += vz * dt_half;

            states[idx] = cs;
        }
    };

    struct StepTwo {
        dfloat3 vel;
        const dfloat3 force;
        const float* mass_inv;

        const float dt_half_conv;

        StepTwo(
            dfloat3 _vel, 
            dfloat3 _force, 
            float* _mass_inv, 
            float _dt_half_conv
        ) : vel(_vel), 
            force(_force), 
            mass_inv(_mass_inv), 
            dt_half_conv(_dt_half_conv) {}

        __host__ __device__ void operator() (const int idx) {
            float mi = mass_inv[idx];
            const auto mi_dt_half_conv = mi * dt_half_conv;

            // 速度の更新3
            vel.x[idx] += force.x[idx] * mi_dt_half_conv;
            vel.y[idx] += force.y[idx] * mi_dt_half_conv;
            vel.z[idx] += force.z[idx] * mi_dt_half_conv; 
        }
    };
}

using namespace md::integrators;

void LangevinIntegrator::init(const State& state, unsigned long long seed) {
    this->dof = 3 * state.n_atoms;
    this->c1 = std::exp(-this->gamma * state.dt);

    this->curand_state.resize(state.n_atoms);

    thrust::for_each(
        thrust::device, 
        thrust::make_counting_iterator<int>(0), 
        thrust::make_counting_iterator<int>(state.n_atoms), 
        InitCurand(thrust::raw_pointer_cast(curand_state.data()), seed)
    );
}

void LangevinIntegrator::integrateStepOne(State& state) {
    // c3の計算
    this->scheduler->get_temperature(state);

    const auto dt_half = state.dt * 0.5;
    const auto dt_half_conv = dt_half * conversion_factor;

    // 更新
    thrust::for_each(
        thrust::cuda::par_nosync.on(state.stream), 
        thrust::make_counting_iterator<int>(0), 
        thrust::make_counting_iterator<int>(state.n_atoms), 
        StepOne(
            state.pos, 
            state.vel, 
            state.force, 
            state.mass, 
            state.mass_inv, 
            thrust::raw_pointer_cast(this->curand_state.data()), 
            dt_half, 
            dt_half_conv, 
            this->c1, 
            boltzmann_constant
        )
    );
}

void LangevinIntegrator::integrateStepTwo(State& state) {
    const auto dt_half_conv = state.dt * 0.5 * conversion_factor;

    thrust::for_each(
        thrust::cuda::par_nosync.on(state.stream), 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(state.n_atoms), 
        StepTwo(
            state.vel, 
            state.force, 
            state.mass_inv, 
            dt_half_conv
        )
    );
}

md::CheckpointBytes LangevinIntegrator::save_checkpoint(State& state) const {
    cudaStreamSynchronize(state.stream);
    const std::uint64_t count = curand_state.size();
    const std::size_t state_bytes = static_cast<std::size_t>(count) * sizeof(curandState);
    md::CheckpointBytes data(sizeof(std::uint64_t) + sizeof(float) * 2 + sizeof(int) + state_bytes);
    std::size_t offset = 0;
    std::memcpy(data.data() + offset, &count, sizeof(count));
    offset += sizeof(count);
    std::memcpy(data.data() + offset, &c1, sizeof(c1));
    offset += sizeof(c1);
    std::memcpy(data.data() + offset, &gamma, sizeof(gamma));
    offset += sizeof(gamma);
    std::memcpy(data.data() + offset, &dof, sizeof(dof));
    offset += sizeof(dof);
    if (state_bytes > 0) {
        cudaMemcpy(
            data.data() + offset,
            thrust::raw_pointer_cast(curand_state.data()),
            state_bytes,
            cudaMemcpyDeviceToHost
        );
    }
    return data;
}

void LangevinIntegrator::load_checkpoint(State& state, const md::CheckpointBytes& data) {
    const std::size_t header_bytes = sizeof(std::uint64_t) + sizeof(float) * 2 + sizeof(int);
    if (data.size() < header_bytes) {
        throw std::runtime_error("Invalid Langevin checkpoint payload.");
    }
    std::size_t offset = 0;
    std::uint64_t count = 0;
    std::memcpy(&count, data.data() + offset, sizeof(count));
    offset += sizeof(count);
    std::memcpy(&c1, data.data() + offset, sizeof(c1));
    offset += sizeof(c1);
    std::memcpy(&gamma, data.data() + offset, sizeof(gamma));
    offset += sizeof(gamma);
    std::memcpy(&dof, data.data() + offset, sizeof(dof));
    offset += sizeof(dof);
    const std::size_t state_bytes = static_cast<std::size_t>(count) * sizeof(curandState);
    if (count != static_cast<std::uint64_t>(state.n_atoms) || data.size() != offset + state_bytes) {
        throw std::runtime_error("Langevin checkpoint atom/RNG state count mismatch.");
    }
    curand_state.resize(static_cast<std::size_t>(count));
    if (state_bytes > 0) {
        cudaMemcpy(
            thrust::raw_pointer_cast(curand_state.data()),
            data.data() + offset,
            state_bytes,
            cudaMemcpyHostToDevice
        );
    }
    cudaStreamSynchronize(state.stream);
}
