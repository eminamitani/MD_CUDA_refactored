#include <md/core/CheckpointManager.cuh>

#include <md/core/State.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/observers/Observer.cuh>
#include <md/thermostats/Thermostat.cuh>
#include <md/utils/CudaCheck.cuh>

#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>

#ifndef MD_BUILD_ID
#define MD_BUILD_ID "unknown"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
    constexpr std::uint32_t kCheckpointSchema = 1;
    constexpr std::uint32_t kEndianMarker = 0x01020304U;
    constexpr std::array<char, 8> kMagic{'M', 'D', 'C', 'U', 'D', 'A', 'C', 'K'};

    class Sha256 {
        public:
            Sha256() { reset(); }

            void update(const std::uint8_t* data, std::size_t length) {
                for (std::size_t i = 0; i < length; ++i) {
                    buffer_[buffer_length_++] = data[i];
                    if (buffer_length_ == 64) {
                        transform(buffer_.data());
                        bit_length_ += 512;
                        buffer_length_ = 0;
                    }
                }
            }

            std::string finish() {
                std::uint64_t total_bits = bit_length_ + static_cast<std::uint64_t>(buffer_length_) * 8U;
                buffer_[buffer_length_++] = 0x80U;
                if (buffer_length_ > 56) {
                    while (buffer_length_ < 64) buffer_[buffer_length_++] = 0;
                    transform(buffer_.data());
                    buffer_length_ = 0;
                }
                while (buffer_length_ < 56) buffer_[buffer_length_++] = 0;
                for (int i = 7; i >= 0; --i) {
                    buffer_[buffer_length_++] = static_cast<std::uint8_t>((total_bits >> (i * 8)) & 0xffU);
                }
                transform(buffer_.data());

                std::ostringstream out;
                out << std::hex << std::setfill('0');
                for (std::uint32_t word : state_) out << std::setw(8) << word;
                return out.str();
            }

        private:
            static constexpr std::array<std::uint32_t, 64> k_{
                0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
                0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
                0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
                0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
                0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
                0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
                0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
                0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
            };

            std::array<std::uint32_t, 8> state_{};
            std::array<std::uint8_t, 64> buffer_{};
            std::size_t buffer_length_ = 0;
            std::uint64_t bit_length_ = 0;

            static std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) {
                return (value >> bits) | (value << (32U - bits));
            }

            void reset() {
                state_ = {
                    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
                };
                buffer_length_ = 0;
                bit_length_ = 0;
            }

            void transform(const std::uint8_t* block) {
                std::array<std::uint32_t, 64> words{};
                for (std::size_t i = 0; i < 16; ++i) {
                    words[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24U)
                             | (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16U)
                             | (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8U)
                             | static_cast<std::uint32_t>(block[i * 4 + 3]);
                }
                for (std::size_t i = 16; i < 64; ++i) {
                    const std::uint32_t s0 = rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^ (words[i - 15] >> 3U);
                    const std::uint32_t s1 = rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^ (words[i - 2] >> 10U);
                    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
                }
                std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
                std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
                for (std::size_t i = 0; i < 64; ++i) {
                    const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                    const std::uint32_t choice = (e & f) ^ ((~e) & g);
                    const std::uint32_t temp1 = h + s1 + choice + k_[i] + words[i];
                    const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
                    const std::uint32_t temp2 = s0 + majority;
                    h = g; g = f; f = e; e = d + temp1;
                    d = c; c = b; b = a; a = temp1 + temp2;
                }
                state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
                state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
            }
    };

    std::string sha256_bytes(const std::vector<std::uint8_t>& data) {
        Sha256 hash;
        if (!data.empty()) hash.update(data.data(), data.size());
        return hash.finish();
    }

    template <typename T>
    void append_value(std::vector<std::uint8_t>& out, const T& value) {
        static_assert(std::is_trivially_copyable<T>::value, "checkpoint values must be trivially copyable");
        const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
        out.insert(out.end(), begin, begin + sizeof(T));
    }

    void append_bytes(std::vector<std::uint8_t>& out, const void* data, std::size_t size) {
        const auto* begin = reinterpret_cast<const std::uint8_t*>(data);
        out.insert(out.end(), begin, begin + size);
    }

    void append_blob(std::vector<std::uint8_t>& out, const md::CheckpointBytes& data) {
        append_value(out, static_cast<std::uint64_t>(data.size()));
        if (!data.empty()) append_bytes(out, data.data(), data.size());
    }

    class Reader {
        public:
            explicit Reader(const std::vector<std::uint8_t>& data) : data_(data) {}

            template <typename T>
            T value() {
                static_assert(std::is_trivially_copyable<T>::value, "checkpoint values must be trivially copyable");
                require(sizeof(T));
                T result{};
                std::memcpy(&result, data_.data() + offset_, sizeof(T));
                offset_ += sizeof(T);
                return result;
            }

            void bytes(void* output, std::size_t size) {
                require(size);
                std::memcpy(output, data_.data() + offset_, size);
                offset_ += size;
            }

            md::CheckpointBytes blob() {
                const std::uint64_t size = value<std::uint64_t>();
                if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                    throw std::runtime_error("Checkpoint blob is too large.");
                }
                md::CheckpointBytes result(static_cast<std::size_t>(size));
                if (!result.empty()) bytes(result.data(), result.size());
                return result;
            }

            bool finished() const { return offset_ == data_.size(); }

        private:
            const std::vector<std::uint8_t>& data_;
            std::size_t offset_ = 0;

            void require(std::size_t size) {
                if (size > data_.size() - offset_) {
                    throw std::runtime_error("Truncated checkpoint payload.");
                }
            }
    };

    std::vector<std::uint8_t> read_binary(const fs::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Could not open checkpoint payload: " + path.string());
        input.seekg(0, std::ios::end);
        const auto end = input.tellg();
        if (end < 0) throw std::runtime_error("Could not size checkpoint payload: " + path.string());
        std::vector<std::uint8_t> data(static_cast<std::size_t>(end));
        input.seekg(0);
        if (!data.empty()) input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!input) throw std::runtime_error("Could not read checkpoint payload: " + path.string());
        return data;
    }

    void fsync_path(const fs::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Could not open for fsync: " + path.string());
        if (::fsync(fd) != 0) {
            ::close(fd);
            throw std::runtime_error("fsync failed: " + path.string());
        }
        ::close(fd);
    }

    void atomic_write_binary(const fs::path& path, const std::vector<std::uint8_t>& data) {
        const fs::path temporary = path.string() + ".tmp";
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("Could not write checkpoint temporary: " + temporary.string());
            if (!data.empty()) output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            output.flush();
            if (!output) throw std::runtime_error("Could not flush checkpoint temporary: " + temporary.string());
        }
        fsync_path(temporary);
        fs::rename(temporary, path);
    }

    void atomic_write_text(const fs::path& path, const std::string& text) {
        const fs::path temporary = path.string() + ".tmp";
        {
            std::ofstream output(temporary, std::ios::trunc);
            if (!output) throw std::runtime_error("Could not write checkpoint sidecar: " + temporary.string());
            output << text;
            output.flush();
            if (!output) throw std::runtime_error("Could not flush checkpoint sidecar: " + temporary.string());
        }
        fsync_path(temporary);
        fs::rename(temporary, path);
    }

    std::string executable_path() {
#if defined(__linux__)
        std::array<char, 4096> buffer{};
        const ssize_t size = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
        if (size > 0) return std::string(buffer.data(), static_cast<std::size_t>(size));
#endif
        return {};
    }

    json runtime_compatibility() {
        int cuda_runtime = 0;
        int curand_version = 0;
        int device = 0;
        cudaDeviceProp properties{};
        MD_CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));
        const curandStatus_t curand_status = curandGetVersion(&curand_version);
        if (curand_status != CURAND_STATUS_SUCCESS) {
            throw std::runtime_error("curandGetVersion failed while creating checkpoint metadata.");
        }
        MD_CUDA_CHECK(cudaGetDevice(&device));
        MD_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        return {
            {"cuda_runtime", cuda_runtime},
            {"curand_version", curand_version},
            {"curand_state_size", sizeof(curandState)},
            {"compute_capability", std::to_string(properties.major) + "." + std::to_string(properties.minor)},
            {"build_id", MD_BUILD_ID}
        };
    }

    std::string zero_or_hash(const fs::path& path) {
        return path.empty() ? std::string() : md::checkpoint::sha256_file(path);
    }

    bool is_sidecar_name(const fs::path& path) {
        const std::string name = path.filename().string();
        return name.rfind("checkpoint.", 0) == 0 && path.extension() == ".json";
    }

    md::checkpoint::CheckpointRecord validate_record(const fs::path& sidecar) {
        std::ifstream input(sidecar);
        if (!input) throw std::runtime_error("Could not open checkpoint sidecar: " + sidecar.string());
        json metadata = json::parse(input);
        if (metadata.at("schema_version").get<std::uint32_t>() != kCheckpointSchema) {
            throw std::runtime_error("Unsupported checkpoint schema: " + sidecar.string());
        }
        fs::path payload = sidecar;
        payload.replace_filename(metadata.at("payload_file").get<std::string>());
        if (!fs::is_regular_file(payload)) {
            throw std::runtime_error("Missing checkpoint payload: " + payload.string());
        }
        const std::string observed = md::checkpoint::sha256_file(payload);
        const std::string expected = metadata.at("payload_sha256").get<std::string>();
        if (observed != expected) {
            throw std::runtime_error("Checkpoint SHA-256 mismatch: " + payload.string());
        }
        return {sidecar, payload, std::move(metadata)};
    }

    std::vector<fs::path> candidate_sidecars(const fs::path& directory) {
        std::vector<fs::path> result;
        if (!fs::exists(directory)) return result;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file() && is_sidecar_name(entry.path())) result.push_back(entry.path());
        }
        std::sort(result.begin(), result.end(), std::greater<fs::path>());
        return result;
    }

    std::vector<float> copy_device_floats(const float* device, std::size_t count, cudaStream_t stream) {
        std::vector<float> host(count);
        if (count > 0) {
            MD_CUDA_CHECK(cudaMemcpyAsync(host.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        return host;
    }

    std::vector<int> copy_device_ints(const int* device, std::size_t count, cudaStream_t stream) {
        std::vector<int> host(count);
        if (count > 0) {
            MD_CUDA_CHECK(cudaMemcpyAsync(host.data(), device, count * sizeof(int), cudaMemcpyDeviceToHost, stream));
        }
        return host;
    }

    template <typename T>
    void append_vector(std::vector<std::uint8_t>& output, const std::vector<T>& values) {
        append_value(output, static_cast<std::uint64_t>(values.size()));
        if (!values.empty()) append_bytes(output, values.data(), values.size() * sizeof(T));
    }

    template <typename T>
    std::vector<T> read_vector(Reader& reader, std::size_t expected_size) {
        const std::uint64_t size = reader.value<std::uint64_t>();
        if (size != expected_size) throw std::runtime_error("Checkpoint array size mismatch.");
        std::vector<T> values(static_cast<std::size_t>(size));
        if (!values.empty()) reader.bytes(values.data(), values.size() * sizeof(T));
        return values;
    }
}

namespace md::checkpoint {
    RestartConfig RestartConfig::from_json(const json& setting) {
        RestartConfig result;
        result.mode = setting.value("mode", "off");
        if (result.mode != "off" && result.mode != "auto" && result.mode != "require") {
            throw std::runtime_error("restart.mode must be off, auto, or require.");
        }
        result.directory = setting.value("directory", std::string("checkpoints"));
        result.checkpoint_interval_seconds = setting.value("checkpoint_interval_seconds", 3600.0);
        result.max_walltime_seconds = setting.value("max_walltime_seconds", 244800.0);
        result.poll_interval_steps = setting.value("poll_interval_steps", std::int64_t{1000});
        result.keep_generations = setting.value("keep_generations", 2);
        result.strict_compatibility = setting.value("strict_compatibility", true);
        if (result.checkpoint_interval_seconds <= 0.0 || result.max_walltime_seconds <= 0.0) {
            throw std::runtime_error("restart time intervals must be positive.");
        }
        if (result.poll_interval_steps <= 0 || result.keep_generations < 2) {
            throw std::runtime_error("restart.poll_interval_steps must be positive and keep_generations must be >= 2.");
        }
        return result;
    }

    std::string sha256_file(const fs::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Could not open file for SHA-256: " + path.string());
        Sha256 hash;
        std::array<std::uint8_t, 1 << 16> buffer{};
        while (input) {
            input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = input.gcount();
            if (count > 0) hash.update(buffer.data(), static_cast<std::size_t>(count));
        }
        if (!input.eof()) throw std::runtime_error("Failed while hashing file: " + path.string());
        return hash.finish();
    }

    CheckpointManager::CheckpointManager(
        RestartConfig config,
        fs::path setting_path,
        fs::path model_path,
        std::array<std::array<float, 3>, 3> lattice
    ) : config_(std::move(config)),
        setting_path_(std::move(setting_path)),
        model_path_(std::move(model_path)),
        lattice_(std::move(lattice)),
        config_sha256_(zero_or_hash(setting_path_)),
        model_sha256_(zero_or_hash(model_path_)) {
        const std::string executable = executable_path();
        executable_sha256_ = executable.empty() ? std::string() : sha256_file(executable);
    }

    std::optional<CheckpointRecord> CheckpointManager::discover(const RestartConfig& config) {
        if (!config.enabled()) return std::nullopt;
        const auto candidates = candidate_sidecars(config.directory);
        std::vector<std::string> errors;
        for (const auto& sidecar : candidates) {
            try {
                return validate_record(sidecar);
            } catch (const std::exception& error) {
                errors.push_back(error.what());
            }
        }
        if (!candidates.empty()) {
            std::ostringstream message;
            message << "No valid checkpoint generation found in " << config.directory << ".";
            for (const auto& error : errors) message << "\n  - " << error;
            throw std::runtime_error(message.str());
        }
        return std::nullopt;
    }

    CheckpointRecord CheckpointManager::save(
        State& state,
        Integrator& integrator,
        Thermostat* thermostat,
        Observer& observer,
        int workflow_step_index,
        const std::string& workflow_step_name,
        std::int64_t target_step
    ) {
        MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
        const std::size_t n = static_cast<std::size_t>(state.n_atoms);
        auto pos = copy_device_floats(state.pos.x, 3 * n, state.stream);
        auto vel = copy_device_floats(state.vel.x, 3 * n, state.stream);
        auto force = copy_device_floats(state.force.x, 3 * n, state.stream);
        auto box_x = copy_device_ints(state.box.x, n, state.stream);
        auto box_y = copy_device_ints(state.box.y, n, state.stream);
        auto box_z = copy_device_ints(state.box.z, n, state.stream);
        auto mass = copy_device_floats(state.mass, n, state.stream);
        auto mass_inv = copy_device_floats(state.mass_inv, n, state.stream);
        auto atomic_numbers = copy_device_ints(state.atomic_numbers, n, state.stream);
        float kinetic_energy = 0.0f;
        float cached_potential_energy = 0.0f;
        MD_CUDA_CHECK(cudaMemcpyAsync(&kinetic_energy, state.kinetic_energy, sizeof(float), cudaMemcpyDeviceToHost, state.stream));
        MD_CUDA_CHECK(cudaMemcpyAsync(&cached_potential_energy, state.cached_potential_energy, sizeof(float), cudaMemcpyDeviceToHost, state.stream));
        MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));

        const auto integrator_data = integrator.save_checkpoint(state);
        const auto thermostat_data = thermostat ? thermostat->save_checkpoint(state) : CheckpointBytes{};
        const auto observer_data = observer.save_checkpoint(state);

        std::vector<std::uint8_t> payload;
        payload.reserve((9 * n * sizeof(float)) + (6 * n * sizeof(int)) + 4096);
        append_bytes(payload, kMagic.data(), kMagic.size());
        append_value(payload, kCheckpointSchema);
        append_value(payload, kEndianMarker);
        append_value(payload, state.n_atoms);
        append_value(payload, state.dt);
        append_value(payload, state.current_steps);
        append_value(payload, state.absolute_steps);
        append_value(payload, state.potential_energy);
        append_value(payload, kinetic_energy);
        append_value(payload, cached_potential_energy);
        append_value(payload, static_cast<std::uint8_t>(state.cached_potential_energy_valid ? 1 : 0));
        append_value(payload, state.temperature_dof);
        append_value(payload, state.thermostat_dof);
        append_value(payload, state.com_drift_removal_interval);
        append_vector(payload, pos);
        append_vector(payload, vel);
        append_vector(payload, force);
        append_vector(payload, box_x);
        append_vector(payload, box_y);
        append_vector(payload, box_z);
        append_vector(payload, mass);
        append_vector(payload, mass_inv);
        append_vector(payload, atomic_numbers);
        append_blob(payload, integrator_data);
        append_blob(payload, thermostat_data);
        append_blob(payload, observer_data);

        fs::create_directories(config_.directory);
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const std::int64_t generation = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        std::ostringstream stem;
        stem << "checkpoint.step" << std::setw(14) << std::setfill('0') << state.current_steps
             << ".segment" << std::setw(4) << std::setfill('0') << state.trajectory_segment_id
             << ".g" << generation;
        const fs::path payload_path = config_.directory / (stem.str() + ".bin");
        const fs::path sidecar_path = config_.directory / (stem.str() + ".json");
        atomic_write_binary(payload_path, payload);
        const std::string payload_sha = sha256_file(payload_path);

        const json component_ids = {
            {"integrator", integrator.checkpoint_id()},
            {"thermostat", thermostat ? thermostat->checkpoint_id() : "none"},
            {"observer", observer.checkpoint_id()}
        };
        json metadata = {
            {"schema_version", kCheckpointSchema},
            {"checkpoint_id", stem.str()},
            {"generation", generation},
            {"payload_file", payload_path.filename().string()},
            {"payload_sha256", payload_sha},
            {"payload_size", payload.size()},
            {"workflow_step_index", workflow_step_index},
            {"workflow_step_name", workflow_step_name},
            {"segment_id", state.trajectory_segment_id},
            {"parent_checkpoint_id", state.checkpoint_parent_id},
            {"current_steps", state.current_steps},
            {"absolute_steps", state.absolute_steps},
            {"target_step", target_step},
            {"n_atoms", state.n_atoms},
            {"dt", state.dt},
            {"lattice", lattice_},
            {"config_sha256", config_sha256_},
            {"model_sha256", model_sha256_},
            {"executable_sha256", executable_sha256_},
            {"component_ids", component_ids},
            {"runtime_compatibility", runtime_compatibility()}
        };
        atomic_write_text(sidecar_path, metadata.dump(2) + "\n");
        const json latest = {
            {"schema_version", kCheckpointSchema},
            {"sidecar_file", sidecar_path.filename().string()},
            {"checkpoint_id", stem.str()},
            {"payload_sha256", payload_sha}
        };
        atomic_write_text(config_.directory / "latest.json", latest.dump(2) + "\n");
        fsync_path(config_.directory);
        prune_old_generations();
        return {sidecar_path, payload_path, std::move(metadata)};
    }

    CheckpointLoadResult CheckpointManager::load(
        const CheckpointRecord& record,
        State& state,
        Integrator& integrator,
        Thermostat* thermostat,
        Observer& observer,
        int expected_workflow_step_index,
        std::int64_t expected_target_step
    ) const {
        const json& metadata = record.metadata;
        const auto expect_equal = [&](const char* key, const auto& observed, const auto& expected) {
            if (observed != expected) {
                std::ostringstream message;
                message << "Checkpoint compatibility mismatch for " << key;
                throw std::runtime_error(message.str());
            }
        };
        expect_equal("workflow_step_index", metadata.at("workflow_step_index").get<int>(), expected_workflow_step_index);
        expect_equal("n_atoms", metadata.at("n_atoms").get<int>(), state.n_atoms);
        if (std::fabs(metadata.at("dt").get<float>() - state.dt) > 1.0e-8f) {
            throw std::runtime_error("Checkpoint dt mismatch.");
        }
        if (metadata.at("target_step").get<std::int64_t>() != expected_target_step) {
            throw std::runtime_error("Checkpoint target_step mismatch.");
        }
        if (config_.strict_compatibility) {
            expect_equal("config_sha256", metadata.at("config_sha256").get<std::string>(), config_sha256_);
            expect_equal("model_sha256", metadata.at("model_sha256").get<std::string>(), model_sha256_);
            expect_equal("executable_sha256", metadata.at("executable_sha256").get<std::string>(), executable_sha256_);
            expect_equal("lattice", metadata.at("lattice"), json(lattice_));
            expect_equal("integrator", metadata.at("component_ids").at("integrator").get<std::string>(), integrator.checkpoint_id());
            expect_equal(
                "thermostat",
                metadata.at("component_ids").at("thermostat").get<std::string>(),
                thermostat ? thermostat->checkpoint_id() : std::string("none")
            );
            expect_equal("observer", metadata.at("component_ids").at("observer").get<std::string>(), observer.checkpoint_id());
            expect_equal("runtime_compatibility", metadata.at("runtime_compatibility"), runtime_compatibility());
        }

        const auto payload = read_binary(record.payload_path);
        if (sha256_bytes(payload) != metadata.at("payload_sha256").get<std::string>()) {
            throw std::runtime_error("Checkpoint payload changed after discovery.");
        }
        Reader reader(payload);
        std::array<char, 8> magic{};
        reader.bytes(magic.data(), magic.size());
        if (magic != kMagic) throw std::runtime_error("Invalid checkpoint magic.");
        if (reader.value<std::uint32_t>() != kCheckpointSchema || reader.value<std::uint32_t>() != kEndianMarker) {
            throw std::runtime_error("Unsupported checkpoint schema/endian marker.");
        }
        if (reader.value<int>() != state.n_atoms) throw std::runtime_error("Checkpoint atom count mismatch.");
        state.dt = reader.value<float>();
        state.current_steps = reader.value<std::int64_t>();
        state.absolute_steps = reader.value<std::int64_t>();
        state.potential_energy = reader.value<float>();
        const float kinetic_energy = reader.value<float>();
        const float cached_potential_energy = reader.value<float>();
        state.cached_potential_energy_valid = reader.value<std::uint8_t>() != 0;
        state.temperature_dof = reader.value<int>();
        state.thermostat_dof = reader.value<int>();
        state.com_drift_removal_interval = reader.value<int>();
        const std::size_t n = static_cast<std::size_t>(state.n_atoms);
        const auto pos = read_vector<float>(reader, 3 * n);
        const auto vel = read_vector<float>(reader, 3 * n);
        const auto force = read_vector<float>(reader, 3 * n);
        const auto box_x = read_vector<int>(reader, n);
        const auto box_y = read_vector<int>(reader, n);
        const auto box_z = read_vector<int>(reader, n);
        const auto mass = read_vector<float>(reader, n);
        const auto mass_inv = read_vector<float>(reader, n);
        const auto atomic_numbers = read_vector<int>(reader, n);
        const auto integrator_data = reader.blob();
        const auto thermostat_data = reader.blob();
        const auto observer_data = reader.blob();
        if (!reader.finished()) throw std::runtime_error("Checkpoint payload has trailing data.");

        MD_CUDA_CHECK(cudaMemcpy(state.pos.x, pos.data(), pos.size() * sizeof(float), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.vel.x, vel.data(), vel.size() * sizeof(float), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.force.x, force.data(), force.size() * sizeof(float), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.box.x, box_x.data(), box_x.size() * sizeof(int), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.box.y, box_y.data(), box_y.size() * sizeof(int), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.box.z, box_z.data(), box_z.size() * sizeof(int), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.mass, mass.data(), mass.size() * sizeof(float), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.mass_inv, mass_inv.data(), mass_inv.size() * sizeof(float), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.atomic_numbers, atomic_numbers.data(), atomic_numbers.size() * sizeof(int), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.kinetic_energy, &kinetic_energy, sizeof(float), cudaMemcpyHostToDevice));
        MD_CUDA_CHECK(cudaMemcpy(state.cached_potential_energy, &cached_potential_energy, sizeof(float), cudaMemcpyHostToDevice));
        integrator.load_checkpoint(state, integrator_data);
        if (thermostat) thermostat->load_checkpoint(state, thermostat_data);
        else if (!thermostat_data.empty()) throw std::runtime_error("Checkpoint contains thermostat data but no thermostat is active.");
        observer.load_checkpoint(state, observer_data);
        state.trajectory_segment_id = metadata.at("segment_id").get<int>() + 1;
        state.checkpoint_parent_id = metadata.at("checkpoint_id").get<std::string>();
        MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
        return {record, force};
    }

    void CheckpointManager::verify_recomputed_force(
        State& state,
        const std::vector<float>& saved_force,
        float absolute_tolerance,
        float relative_tolerance
    ) const {
        const std::size_t expected = static_cast<std::size_t>(state.n_atoms) * 3;
        if (saved_force.size() != expected) throw std::runtime_error("Saved force size mismatch.");
        auto recomputed = copy_device_floats(state.force.x, expected, state.stream);
        MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
        float max_abs = 0.0f;
        float max_allowed = 0.0f;
        for (std::size_t i = 0; i < expected; ++i) {
            const float difference = std::fabs(recomputed[i] - saved_force[i]);
            const float allowed = absolute_tolerance + relative_tolerance * std::fabs(saved_force[i]);
            max_abs = std::max(max_abs, difference);
            max_allowed = std::max(max_allowed, allowed);
            if (!std::isfinite(recomputed[i]) || difference > allowed) {
                std::ostringstream message;
                message << "Restart force validation failed at component " << i
                        << ": saved=" << saved_force[i] << ", recomputed=" << recomputed[i]
                        << ", difference=" << difference << ", allowed=" << allowed;
                throw std::runtime_error(message.str());
            }
        }
        // The rebuilt neighbour list may visit otherwise identical neighbours in
        // a different order.  Use the recomputed force only as a compatibility
        // check, then restore the bitwise checkpoint value for the first
        // post-restart half-step.
        MD_CUDA_CHECK(cudaMemcpy(
            state.force.x,
            saved_force.data(),
            saved_force.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));
        MD_CUDA_CHECK(cudaStreamSynchronize(state.stream));
        std::cout << "checkpoint force validation passed: max_abs=" << max_abs
                  << ", max_allowed=" << max_allowed
                  << "; restored saved force for continuation" << std::endl;
    }

    void CheckpointManager::prune_old_generations() const {
        auto candidates = candidate_sidecars(config_.directory);
        if (static_cast<int>(candidates.size()) <= config_.keep_generations) return;
        for (std::size_t i = static_cast<std::size_t>(config_.keep_generations); i < candidates.size(); ++i) {
            try {
                std::ifstream input(candidates[i]);
                json metadata = json::parse(input);
                fs::remove(config_.directory / metadata.at("payload_file").get<std::string>());
                fs::remove(candidates[i]);
            } catch (const std::exception&) {
                // Keep malformed generations for manual diagnosis rather than deleting unknown files.
            }
        }
    }
}
