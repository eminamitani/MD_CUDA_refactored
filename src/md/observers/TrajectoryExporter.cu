#include <md/observers/TrajectoryExporter.cuh>

#include <md/cells/Cell.cuh>
#include <md/core/State.cuh>

#include <sstream>

using namespace md::observers;

TrajectoryExporter::TrajectoryExporter(
    State& state,
    const std::string& output_path,
    Cell* _cell,
    const TrajectoryOutputSpec& _spec
) : cell(_cell), spec(_spec), atom_number_map{"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"} {
    if (spec.format != "extxyz") {
        throw std::runtime_error("Only extxyz trajectory output is supported.");
    }
    this->ofs.open(output_path);
    if (!ofs) {
        throw std::runtime_error("出力ファイルが開けませんでした。");
    }

    size_t N = state.n_atoms;

    std::vector<int> h_atomic_numbers(N);
    cudaMemcpy(h_atomic_numbers.data(), state.atomic_numbers, N * sizeof(int), cudaMemcpyDeviceToHost);

    species.resize(N);
    for (size_t i = 0; i < N; i ++) {
        int atomic_number = h_atomic_numbers[i];
        if (atomic_number >= 1 && atomic_number <= static_cast<int>(atom_number_map.size())) {
            species[i] = atom_number_map[atomic_number - 1];
        } else {
            species[i] = "X" + std::to_string(atomic_number);
        }
    }

    if (spec.position) h_pos.resize(3 * N);
    if (spec.velocity) h_velocity.resize(3 * N);
    if (spec.force) h_force.resize(3 * N);
    if (spec.position && spec.unwrap) h_box.resize(3 * N);
}

void TrajectoryExporter::export_trajectory(State& state) {
    export_trajectory(state, "");
}

void TrajectoryExporter::export_trajectory(State& state, const std::string& extra_comment) {
    export_frame(state, extra_comment, spec.unwrap);
}

void TrajectoryExporter::export_trajectory_unwrap(State& state) {
    export_trajectory_unwrap(state, "");
}

void TrajectoryExporter::export_trajectory_unwrap(State& state, const std::string& extra_comment) {
    export_frame(state, extra_comment, true);
}

std::string TrajectoryExporter::frame_comment(
    State& state,
    const std::string& extra_comment,
    bool unwrap
) const {
    std::ostringstream comment;
    bool has_comment = false;
    if (spec.write_field_metadata) {
        has_comment = true;
        comment << "trajectory_mode=" << spec.mode
                << " time_fs=" << std::setprecision(16)
                << static_cast<double>(state.dt) * static_cast<double>(state.current_steps);
        if (spec.position) comment << " position_unit=angstrom";
        if (spec.velocity) comment << " velocity_unit=angstrom_per_fs";
        if (spec.force) comment << " force_unit=eV_per_angstrom";
        if (spec.position) comment << " coordinates=" << (unwrap ? "unwrapped" : "wrapped");
    }
    if (state.trajectory_segment_id >= 0) {
        if (has_comment) comment << " ";
        has_comment = true;
        comment << "production_step_abs=" << state.current_steps
                << " workflow_step_abs=" << state.absolute_steps
                << " time_fs_abs=" << std::setprecision(16)
                << static_cast<double>(state.dt) * static_cast<double>(state.current_steps)
                << " segment_id=" << state.trajectory_segment_id;
        if (!state.checkpoint_parent_id.empty()) {
            comment << " checkpoint_parent=" << state.checkpoint_parent_id;
        }
    }
    if (!extra_comment.empty()) {
        if (has_comment) comment << " ";
        comment << extra_comment;
    }
    return comment.str();
}

void TrajectoryExporter::export_frame(
    State& state,
    const std::string& extra_comment,
    bool unwrap
) {
    auto lattice = cell->lattice;

    size_t N = state.n_atoms;
    if (spec.position) {
        cudaMemcpyAsync(
            h_pos.data(), state.pos.x, 3 * N * sizeof(float), cudaMemcpyDeviceToHost, state.stream
        );
        if (unwrap) {
            if (h_box.size() != 3 * N) h_box.resize(3 * N);
            cudaMemcpyAsync(h_box.data(), state.box.x, N * sizeof(int), cudaMemcpyDeviceToHost, state.stream);
            cudaMemcpyAsync(h_box.data() + N, state.box.y, N * sizeof(int), cudaMemcpyDeviceToHost, state.stream);
            cudaMemcpyAsync(h_box.data() + 2 * N, state.box.z, N * sizeof(int), cudaMemcpyDeviceToHost, state.stream);
        }
    }
    if (spec.velocity) {
        cudaMemcpyAsync(
            h_velocity.data(), state.vel.x, 3 * N * sizeof(float), cudaMemcpyDeviceToHost, state.stream
        );
    }
    if (spec.force) {
        cudaMemcpyAsync(
            h_force.data(), state.force.x, 3 * N * sizeof(float), cudaMemcpyDeviceToHost, state.stream
        );
    }
    if (spec.energy && state.cached_potential_energy_valid) {
        cudaMemcpyAsync(
            &state.potential_energy,
            state.cached_potential_energy,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            state.stream
        );
    }
    cudaStreamSynchronize(state.stream);

    ofs << std::setprecision(7) << std::scientific;
    ofs << N << "\n";
    ofs << "Lattice=\"" << lattice[0][0] << " 0.0 0.0 0.0 "
        << lattice[1][1] << " 0.0 0.0 0.0 " << lattice[2][2] << "\" "
        << "Properties=species:S:1";
    if (spec.position) ofs << ":pos:R:3";
    if (spec.velocity) ofs << ":velocities:R:3";
    if (spec.force) ofs << ":forces:R:3";
    if (spec.energy) ofs << " energy=" << state.potential_energy;
    ofs << " pbc=\"" << (unwrap ? "F F F" : "T T T") << "\"";
    const std::string comment = frame_comment(state, extra_comment, unwrap);
    if (!comment.empty()) ofs << " " << comment;
    ofs << "\n";
    for (size_t i = 0; i < N; i ++) {
        ofs << species[i];
        if (spec.position) {
            const float x = h_pos[i] + (unwrap ? h_box[i] * lattice[0][0] : 0.0f);
            const float y = h_pos[N + i] + (unwrap ? h_box[N + i] * lattice[1][1] : 0.0f);
            const float z = h_pos[2 * N + i] + (unwrap ? h_box[2 * N + i] * lattice[2][2] : 0.0f);
            ofs << " " << x << " " << y << " " << z;
        }
        if (spec.velocity) {
            ofs << " " << h_velocity[i] << " " << h_velocity[N + i] << " " << h_velocity[2 * N + i];
        }
        if (spec.force) {
            ofs << " " << h_force[i] << " " << h_force[N + i] << " " << h_force[2 * N + i];
        }
        ofs << "\n";
    }
}
