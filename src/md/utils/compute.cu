#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cmath>
#include <stdexcept>

namespace {
    struct Multiply {
        float* __restrict__ v;
        float* __restrict__ m;

        Multiply(float* _v, float* _m) : v(_v), m(_m) {}

        __host__ __device__ float operator() (int idx) const {
            return v[idx] * m[idx];
        }
    };

    struct RemoveDrift {
        dfloat3 vel;
        float avg_x, avg_y, avg_z;
        RemoveDrift(
            dfloat3 _vel, 
            float _avg_x, 
            float _avg_y, 
            float _avg_z
        ) : vel(_vel), 
            avg_x(_avg_x), 
            avg_y(_avg_y), 
            avg_z(_avg_z) {}

        __host__ __device__ void operator() (int idx) const {
            vel.x[idx] -= avg_x;
            vel.y[idx] -= avg_y;
            vel.z[idx] -= avg_z;
        }
    };

    struct CalcKinEnergy {
        dfloat3 vel;
        float* __restrict__ mass;

        CalcKinEnergy(
            dfloat3 _vel, 
            float* _mass
        ) : vel(_vel), mass(_mass) {}

        __host__ __device__ float operator() (int idx) {
            auto vx = vel.x[idx];
            auto vy = vel.y[idx];
            auto vz = vel.z[idx];

            return 0.5 * mass[idx] * (vx * vx + vy * vy + vz * vz);
        }
    };

    struct ScaleVelocity {
        dfloat3 vel;
        float scale;

        ScaleVelocity(dfloat3 _vel, float _scale) : vel(_vel), scale(_scale) {}

        __host__ __device__ void operator() (int idx) const {
            vel.x[idx] *= scale;
            vel.y[idx] *= scale;
            vel.z[idx] *= scale;
        }
    };
}

void md::utils::compute::remove_drift(State& state) {
    int N = state.n_atoms;
    auto policy = thrust::cuda::par.on(state.stream);

    // calc drift
    float weighted_sum_x = thrust::transform_reduce(
        policy,
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Multiply(
            state.vel.x, 
            state.mass
        ), 
        0.0f, 
        thrust::plus<float>());
    float weighted_sum_y = thrust::transform_reduce(
        policy,
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Multiply(
            state.vel.y, 
            state.mass
        ), 
        0.0f, 
        thrust::plus<float>());
    float weighted_sum_z = thrust::transform_reduce(
        policy,
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Multiply(
            state.vel.z, 
            state.mass
        ), 
        0.0f, 
        thrust::plus<float>());

    float mass_sum = thrust::reduce(
        policy,
        state.mass, 
        state.mass + N, 
        0.0f, 
        thrust::plus<float>()
    );

    float avg_x = weighted_sum_x / mass_sum;
    float avg_y = weighted_sum_y / mass_sum;
    float avg_z = weighted_sum_z / mass_sum;

    // remove drift
    thrust::for_each(
        policy,
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        RemoveDrift(
            state.vel, 
            avg_x, 
            avg_y, 
            avg_z
        )
    );
}

float md::utils::compute::calc_kinetic_energy(State& state) {
    auto N = state.n_atoms;
    auto policy = thrust::cuda::par.on(state.stream);

    // 運動エネルギーの計算
    auto kinetic_energy = thrust::transform_reduce(
        policy,
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N),  
        CalcKinEnergy(
            state.vel, 
            state.mass
        ), 
        0.0f, 
        thrust::plus<float>()
    );

    return kinetic_energy / conversion_factor;
}

int md::utils::compute::temperature_degrees_of_freedom(const State& state) {
    return state.temperature_dof > 0 ? state.temperature_dof : 3 * state.n_atoms;
}

void md::utils::compute::rescale_temperature(State& state, float temperature, int dof) {
    int effective_dof = dof > 0 ? dof : temperature_degrees_of_freedom(state);
    if (effective_dof <= 0) {
        throw std::runtime_error("温度自由度が正ではありません。");
    }

    float kinetic_energy = calc_kinetic_energy(state);
    if (kinetic_energy <= 0.0f) {
        throw std::runtime_error("速度の再スケールに必要な運動エネルギーが正ではありません。");
    }

    const float target_kinetic_energy = 0.5f * effective_dof * boltzmann_constant * temperature;
    const float scale = std::sqrt(target_kinetic_energy / kinetic_energy);

    thrust::for_each(
        thrust::cuda::par.on(state.stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(state.n_atoms),
        ScaleVelocity(state.vel, scale)
    );
}
