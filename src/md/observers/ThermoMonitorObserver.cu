#include <md/observers/ThermoMonitorObserver.cuh>

#include <md/cells/Cell.cuh>
#include <md/core/State.cuh>
#include <md/core/constant.h>
#include <md/interactions/Interaction.cuh>
#include <md/utils/CudaCheck.cuh>
#include <md/utils/compute.cuh>

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <filesystem>
#include <stdexcept>

using namespace md::observers;

namespace {
    __device__ int species_index(int atomic_number) {
        if (atomic_number == 3) return 0;   // Li
        if (atomic_number == 19) return 1;  // K
        if (atomic_number == 14) return 2;  // Si
        if (atomic_number == 8) return 3;   // O
        return -1;
    }

    __device__ int pair_index(int left, int right) {
        if (left > right) {
            const int temporary = left;
            left = right;
            right = temporary;
        }
        constexpr int table[4][4] = {
            {0, 1, 2, 3},
            {1, 4, 5, 6},
            {2, 5, 7, 8},
            {3, 6, 8, 9},
        };
        return table[left][right];
    }

    __global__ void pair_minimum_kernel(
        dfloat3 position,
        const int* atomic_numbers,
        int n_atoms,
        float lx,
        float ly,
        float lz,
        float* minima
    ) {
        const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        const long long total = static_cast<long long>(n_atoms) * n_atoms;
        if (flat >= total) return;
        const int left_atom = static_cast<int>(flat / n_atoms);
        const int right_atom = static_cast<int>(flat % n_atoms);
        if (left_atom >= right_atom) return;
        const int left = species_index(atomic_numbers[left_atom]);
        const int right = species_index(atomic_numbers[right_atom]);
        if (left < 0 || right < 0) return;
        float dx = position.x[left_atom] - position.x[right_atom];
        float dy = position.y[left_atom] - position.y[right_atom];
        float dz = position.z[left_atom] - position.z[right_atom];
        dx -= nearbyintf(dx / lx) * lx;
        dy -= nearbyintf(dy / ly) * ly;
        dz -= nearbyintf(dz / lz) * lz;
        const float distance = sqrtf(dx * dx + dy * dy + dz * dz);
        atomicMin(
            reinterpret_cast<unsigned int*>(&minima[pair_index(left, right)]),
            __float_as_uint(distance)
        );
    }
}

ThermoMonitorObserver::ThermoMonitorObserver(
    int _interval,
    Interaction* _interaction,
    Cell* _cell,
    const std::string& output_path,
    float _minimum_pair_distance_A
) : interval(_interval), interaction(_interaction), cell(_cell),
    minimum_pair_distance_A(_minimum_pair_distance_A) {
    if (interval <= 0) throw std::runtime_error("thermo_monitor interval must be positive.");
    if (minimum_pair_distance_A < 0.0f) {
        throw std::runtime_error("thermo_monitor minimum_pair_distance_A must be non-negative.");
    }
    const std::filesystem::path path(output_path);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    output_file.open(output_path, std::ios::trunc);
    if (!output_file) throw std::runtime_error("Unable to open thermo_monitor output_path.");
    MD_CUDA_CHECK(cudaMalloc(&device_pair_minima, 10 * sizeof(float)));
    output_file
        << "production_step,workflow_step,time_fs,temperature_K,kinetic_eV,"
        << "potential_eV,total_eV,com_vx_A_fs,com_vy_A_fs,com_vz_A_fs,"
        << "min_Li_Li_A,min_Li_K_A,min_Li_Si_A,min_Li_O_A,min_K_K_A,"
        << "min_K_Si_A,min_K_O_A,min_Si_Si_A,min_Si_O_A,min_O_O_A\n";
}

ThermoMonitorObserver::~ThermoMonitorObserver() {
    if (device_pair_minima) cudaFree(device_pair_minima);
}

void ThermoMonitorObserver::init(State& state) {
    emit(state);
}

void ThermoMonitorObserver::output(State& state) {
    if (state.current_steps % interval == 0) emit(state);
}

void ThermoMonitorObserver::emit(State& state) {
    interaction->calc_potential(state);
    const float kinetic = md::utils::compute::calc_kinetic_energy(state);
    const int dof = md::utils::compute::temperature_degrees_of_freedom(state);
    const float temperature = 2.0f * kinetic / (dof * boltzmann_constant);

    const int n_atoms = state.n_atoms;
    if (host_velocity.size() != static_cast<std::size_t>(3 * n_atoms)) {
        host_velocity.resize(3 * n_atoms);
    }
    if (host_mass.size() != static_cast<std::size_t>(n_atoms)) {
        host_mass.resize(n_atoms);
    }
    MD_CUDA_CHECK(cudaMemcpyAsync(
        host_velocity.data(), state.vel.x, 3 * n_atoms * sizeof(float),
        cudaMemcpyDeviceToHost, state.stream
    ));
    MD_CUDA_CHECK(cudaMemcpyAsync(
        host_mass.data(), state.mass, n_atoms * sizeof(float),
        cudaMemcpyDeviceToHost, state.stream
    ));
    std::array<float, 10> minima;
    minima.fill(std::numeric_limits<float>::infinity());
    MD_CUDA_CHECK(cudaMemcpyAsync(
        device_pair_minima, minima.data(), minima.size() * sizeof(float),
        cudaMemcpyHostToDevice, state.stream
    ));
    const long long pairs_grid = static_cast<long long>(n_atoms) * n_atoms;
    constexpr int block_size = 256;
    const int blocks = static_cast<int>((pairs_grid + block_size - 1) / block_size);
    pair_minimum_kernel<<<blocks, block_size, 0, state.stream>>>(
        state.pos,
        state.atomic_numbers,
        n_atoms,
        cell->lattice[0][0],
        cell->lattice[1][1],
        cell->lattice[2][2],
        device_pair_minima
    );
    MD_CUDA_CHECK(cudaGetLastError());
    MD_CUDA_CHECK(cudaMemcpyAsync(
        minima.data(), device_pair_minima, minima.size() * sizeof(float),
        cudaMemcpyDeviceToHost, state.stream
    ));
    MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));

    double mass_sum = 0.0;
    std::array<double, 3> momentum{0.0, 0.0, 0.0};
    for (int atom = 0; atom < n_atoms; ++atom) {
        const double mass = host_mass[atom];
        mass_sum += mass;
        momentum[0] += mass * host_velocity[atom];
        momentum[1] += mass * host_velocity[n_atoms + atom];
        momentum[2] += mass * host_velocity[2 * n_atoms + atom];
    }
    output_file << std::setprecision(10)
        << state.current_steps << ',' << state.absolute_steps << ','
        << static_cast<double>(state.dt) * state.current_steps << ','
        << temperature << ',' << kinetic << ',' << state.potential_energy << ','
        << kinetic + state.potential_energy << ','
        << momentum[0] / mass_sum << ',' << momentum[1] / mass_sum << ','
        << momentum[2] / mass_sum;
    for (const float minimum : minima) {
        if (std::isfinite(minimum)) output_file << ',' << minimum;
        else output_file << ",nan";
    }
    output_file << '\n';
    output_file.flush();
    if (!output_file) throw std::runtime_error("Unable to write thermo_monitor output.");

    const double com_vx = momentum[0] / mass_sum;
    const double com_vy = momentum[1] / mass_sum;
    const double com_vz = momentum[2] / mass_sum;
    if (!std::isfinite(temperature) || !std::isfinite(kinetic) ||
        !std::isfinite(state.potential_energy) ||
        !std::isfinite(com_vx) || !std::isfinite(com_vy) || !std::isfinite(com_vz)) {
        throw std::runtime_error("thermo_monitor detected a non-finite thermodynamic value.");
    }
    if (minimum_pair_distance_A > 0.0f) {
        for (const float minimum : minima) {
            if (std::isfinite(minimum) && minimum < minimum_pair_distance_A) {
                throw std::runtime_error(
                    "thermo_monitor pair-collapse gate failed: minimum pair distance " +
                    std::to_string(minimum) + " A is below " +
                    std::to_string(minimum_pair_distance_A) + " A."
                );
            }
        }
    }
}

void ThermoMonitorObserver::finalize(State&) {
    output_file.flush();
    if (!output_file) throw std::runtime_error("Unable to flush thermo_monitor output.");
}
