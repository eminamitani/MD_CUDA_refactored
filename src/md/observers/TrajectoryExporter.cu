#include <md/observers/TrajectoryExporter.cuh>

#include <md/cells/Cell.cuh>
#include <md/core/State.cuh>

#include <external/nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

using namespace md::observers;

namespace {
    using json = nlohmann::json;
    namespace fs = std::filesystem;

    constexpr char kBinaryMagic[8] = {'M', 'D', 'C', 'B', 'T', '0', '0', '1'};
    constexpr std::uint32_t kBinaryVersion = 1;
    constexpr std::uint64_t kBinaryChunkHeaderBytes = 32;
    constexpr std::uint64_t kBinaryFrameHeaderBytes = 32;

    template <typename T>
    void append_binary(std::vector<char>& destination, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* bytes = reinterpret_cast<const char*>(&value);
        destination.insert(destination.end(), bytes, bytes + sizeof(T));
    }

    void append_binary_block(
        std::vector<char>& destination,
        const void* data,
        std::size_t bytes
    ) {
        const auto* begin = reinterpret_cast<const char*>(data);
        destination.insert(destination.end(), begin, begin + bytes);
    }

    void atomic_write_text(const fs::path& path, const std::string& text) {
        const fs::path temporary = path.string() + ".tmp";
        {
            std::ofstream output(temporary, std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "Unable to open temporary binary trajectory manifest."
                );
            }
            output << text;
            output.flush();
            if (!output) {
                throw std::runtime_error(
                    "Unable to write temporary binary trajectory manifest."
                );
            }
        }
        if (std::rename(temporary.c_str(), path.c_str()) != 0) {
            throw std::runtime_error(
                "Unable to atomically replace binary trajectory manifest."
            );
        }
    }
}

TrajectoryExporter::TrajectoryExporter(
    State& state,
    const std::string& output_path,
    Cell* _cell,
    const TrajectoryOutputSpec& _spec
) : atom_number_map{"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"},
    cell(_cell),
    spec(_spec),
    output_path(output_path) {
    if (spec.format != "extxyz" && spec.format != "transport_binary_v1") {
        throw std::runtime_error(
            "trajectory.format must be extxyz or transport_binary_v1."
        );
    }

    size_t N = state.n_atoms;

    std::vector<int> h_atomic_numbers(N);
    cudaMemcpy(h_atomic_numbers.data(), state.atomic_numbers, N * sizeof(int), cudaMemcpyDeviceToHost);
    atomic_numbers = h_atomic_numbers;

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

    if (spec.format == "extxyz") {
        this->ofs.open(output_path);
        if (!ofs) {
            throw std::runtime_error("出力ファイルが開けませんでした。");
        }
        return;
    }

    if (!spec.position || !spec.velocity || spec.force || spec.unwrap) {
        throw std::runtime_error(
            "transport_binary_v1 requires wrapped position and velocity, "
            "and does not support force output."
        );
    }
    if (spec.binary_chunk_frames <= 0) {
        throw std::runtime_error(
            "transport_binary_v1 chunk_frames must be positive."
        );
    }
    const fs::path manifest_path(output_path);
    if (!manifest_path.parent_path().empty()) {
        fs::create_directories(manifest_path.parent_path());
    }
    if (fs::exists(manifest_path)) {
        throw std::runtime_error(
            "binary trajectory manifest already exists: " + output_path
        );
    }
    binary_lattice = cell->lattice;
    const std::uint64_t record_bytes =
        kBinaryFrameHeaderBytes + 6ULL * N * sizeof(float);
    binary_payload.reserve(
        static_cast<std::size_t>(record_bytes) *
        static_cast<std::size_t>(spec.binary_chunk_frames)
    );
    write_binary_manifest("in_progress");
}

TrajectoryExporter::~TrajectoryExporter() {
    if (!finalized && spec.format == "transport_binary_v1") {
        try {
            finalize();
        } catch (const std::exception& error) {
            std::cerr << "binary trajectory finalization failed during cleanup: "
                      << error.what() << std::endl;
        }
    }
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
    if (spec.format == "transport_binary_v1") {
        if (unwrap) {
            throw std::runtime_error(
                "transport_binary_v1 does not support unwrapped coordinates."
            );
        }
        if (!extra_comment.empty()) {
            throw std::runtime_error(
                "transport_binary_v1 does not support per-frame free-form comments."
            );
        }
        export_binary_frame(state);
        return;
    }
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

void TrajectoryExporter::export_binary_frame(State& state) {
    if (finalized) {
        throw std::runtime_error(
            "cannot append to a finalized binary trajectory."
        );
    }
    const std::size_t N = state.n_atoms;
    cudaMemcpyAsync(
        h_pos.data(), state.pos.x, 3 * N * sizeof(float),
        cudaMemcpyDeviceToHost, state.stream
    );
    cudaMemcpyAsync(
        h_velocity.data(), state.vel.x, 3 * N * sizeof(float),
        cudaMemcpyDeviceToHost, state.stream
    );
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

    const std::int64_t production_step = state.current_steps;
    const std::int64_t workflow_step = state.absolute_steps;
    const double time_fs =
        static_cast<double>(state.dt) * static_cast<double>(state.current_steps);
    const float potential_energy = state.potential_energy;
    const std::uint8_t energy_valid =
        static_cast<std::uint8_t>(
            spec.energy && state.cached_potential_energy_valid ? 1 : 0
        );
    const std::uint8_t padding[3] = {0, 0, 0};
    append_binary(binary_payload, production_step);
    append_binary(binary_payload, workflow_step);
    append_binary(binary_payload, time_fs);
    append_binary(binary_payload, potential_energy);
    append_binary(binary_payload, energy_valid);
    append_binary_block(binary_payload, padding, sizeof(padding));
    append_binary_block(
        binary_payload, h_pos.data(), h_pos.size() * sizeof(float)
    );
    append_binary_block(
        binary_payload, h_velocity.data(), h_velocity.size() * sizeof(float)
    );

    if (binary_chunk_frame_count == 0) {
        binary_first_step = production_step;
        binary_first_time_fs = time_fs;
    }
    binary_last_step = production_step;
    binary_last_time_fs = time_fs;
    ++binary_chunk_frame_count;
    ++binary_total_frame_count;
    if (
        binary_chunk_frame_count >=
        static_cast<std::uint64_t>(spec.binary_chunk_frames)
    ) {
        flush_binary_chunk();
    }
}

void TrajectoryExporter::flush_binary_chunk() {
    if (binary_chunk_frame_count == 0) return;
    const fs::path manifest_path(output_path);
    std::string stem = manifest_path.filename().string();
    if (manifest_path.extension() == ".json") {
        stem = manifest_path.stem().string();
    }
    std::ostringstream filename;
    filename << stem << ".chunk" << std::setw(6) << std::setfill('0')
             << binary_chunk_index << ".bin";
    const fs::path final_path = manifest_path.parent_path() / filename.str();
    const fs::path partial_path = final_path.string() + ".partial";
    if (fs::exists(final_path) || fs::exists(partial_path)) {
        throw std::runtime_error(
            "binary trajectory chunk path already exists: " +
            final_path.string()
        );
    }
    const std::uint64_t record_bytes =
        kBinaryFrameHeaderBytes +
        6ULL * static_cast<std::uint64_t>(atomic_numbers.size()) * sizeof(float);
    const std::uint64_t expected_payload_bytes =
        record_bytes * binary_chunk_frame_count;
    if (binary_payload.size() != expected_payload_bytes) {
        throw std::runtime_error(
            "binary trajectory payload size does not match its frame count."
        );
    }
    {
        std::ofstream output(partial_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Unable to open binary trajectory chunk."
            );
        }
        output.write(kBinaryMagic, sizeof(kBinaryMagic));
        output.write(
            reinterpret_cast<const char*>(&kBinaryVersion),
            sizeof(kBinaryVersion)
        );
        const std::uint32_t n_atoms =
            static_cast<std::uint32_t>(atomic_numbers.size());
        output.write(
            reinterpret_cast<const char*>(&n_atoms), sizeof(n_atoms)
        );
        output.write(
            reinterpret_cast<const char*>(&binary_chunk_frame_count),
            sizeof(binary_chunk_frame_count)
        );
        output.write(
            reinterpret_cast<const char*>(&record_bytes), sizeof(record_bytes)
        );
        output.write(binary_payload.data(), binary_payload.size());
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "Unable to write binary trajectory chunk."
            );
        }
    }
    fs::rename(partial_path, final_path);
    const std::uint64_t bytes =
        kBinaryChunkHeaderBytes + expected_payload_bytes;
    if (fs::file_size(final_path) != bytes) {
        throw std::runtime_error(
            "binary trajectory chunk has an unexpected final size."
        );
    }
    binary_chunks.push_back(
        BinaryChunkRecord{
            filename.str(),
            binary_chunk_frame_count,
            bytes,
            binary_first_step,
            binary_last_step,
            binary_first_time_fs,
            binary_last_time_fs,
        }
    );
    ++binary_chunk_index;
    binary_chunk_frame_count = 0;
    binary_payload.clear();
    write_binary_manifest("in_progress");
}

void TrajectoryExporter::write_binary_manifest(
    const std::string& status
) const {
    if (spec.format != "transport_binary_v1") return;
    json chunks = json::array();
    for (std::size_t index = 0; index < binary_chunks.size(); ++index) {
        const auto& chunk = binary_chunks[index];
        chunks.push_back(
            {
                {"chunk_id", index},
                {"file", chunk.file},
                {"frames", chunk.frames},
                {"bytes", chunk.bytes},
                {
                    "production_step_range",
                    {chunk.first_production_step, chunk.last_production_step}
                },
                {"time_fs_range", {chunk.first_time_fs, chunk.last_time_fs}},
            }
        );
    }
    json manifest = {
        {"format", "mdcuda-transport-binary-v1"},
        {"schema_version", 1},
        {"status", status},
        {"endianness", "little"},
        {"float_dtype", "float32"},
        {"coordinate_layout", "soa_xyz"},
        {"coordinates", "wrapped"},
        {"n_atoms", atomic_numbers.size()},
        {"atomic_numbers", atomic_numbers},
        {"cell_A", binary_lattice},
        {"pbc", {true, true, true}},
        {
            "fields",
            spec.energy
                ? json::array({"position", "velocity", "energy"})
                : json::array({"position", "velocity"})
        },
        {
            "units",
            {
                {"time", "fs"},
                {"position", "angstrom"},
                {"velocity", "angstrom_per_fs"},
                {"energy", "eV"},
            }
        },
        {"chunk_frames", spec.binary_chunk_frames},
        {
            "record_layout",
            {
                {"frame_header_bytes", kBinaryFrameHeaderBytes},
                {"position_values", 3 * atomic_numbers.size()},
                {"velocity_values", 3 * atomic_numbers.size()},
                {"record_bytes",
                    kBinaryFrameHeaderBytes +
                    6ULL * atomic_numbers.size() * sizeof(float)},
            }
        },
        {"frame_count", binary_total_frame_count},
        {"chunks", chunks},
    };
    atomic_write_text(
        fs::path(output_path), manifest.dump(2) + "\n"
    );
}

void TrajectoryExporter::finalize() {
    if (finalized) return;
    if (spec.format == "transport_binary_v1") {
        flush_binary_chunk();
        write_binary_manifest("complete");
    } else if (ofs.is_open()) {
        ofs.flush();
        if (!ofs) {
            throw std::runtime_error("Unable to flush extxyz trajectory.");
        }
    }
    finalized = true;
}
