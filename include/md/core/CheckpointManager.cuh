#pragma once

#include <external/nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace md {
    class State;
    class Integrator;
    class Observer;
    class Thermostat;
}

namespace md::checkpoint {
    struct RestartConfig {
        std::string mode = "off";
        std::filesystem::path directory;
        double checkpoint_interval_seconds = 3600.0;
        double max_walltime_seconds = 244800.0;
        std::int64_t poll_interval_steps = 1000;
        int keep_generations = 2;
        bool strict_compatibility = true;

        bool enabled() const { return mode != "off"; }
        static RestartConfig from_json(const nlohmann::json& setting);
    };

    struct CheckpointRecord {
        std::filesystem::path sidecar_path;
        std::filesystem::path payload_path;
        nlohmann::json metadata;
    };

    struct CheckpointLoadResult {
        CheckpointRecord record;
        std::vector<float> saved_force;
    };

    class CheckpointManager {
        public:
            CheckpointManager(
                RestartConfig config,
                std::filesystem::path setting_path,
                std::filesystem::path model_path,
                std::array<std::array<float, 3>, 3> lattice
            );

            static std::optional<CheckpointRecord> discover(const RestartConfig& config);

            CheckpointRecord save(
                State& state,
                Integrator& integrator,
                Thermostat* thermostat,
                Observer& observer,
                int workflow_step_index,
                const std::string& workflow_step_name,
                std::int64_t target_step
            );

            CheckpointLoadResult load(
                const CheckpointRecord& record,
                State& state,
                Integrator& integrator,
                Thermostat* thermostat,
                Observer& observer,
                int expected_workflow_step_index,
                std::int64_t expected_target_step
            ) const;

            void verify_recomputed_force(
                State& state,
                const std::vector<float>& saved_force,
                float absolute_tolerance = 1.0e-4f,
                float relative_tolerance = 1.0e-5f
            ) const;

            const RestartConfig& config() const { return config_; }
            const std::string& config_sha256() const { return config_sha256_; }
            const std::string& model_sha256() const { return model_sha256_; }

        private:
            RestartConfig config_;
            std::filesystem::path setting_path_;
            std::filesystem::path model_path_;
            std::array<std::array<float, 3>, 3> lattice_;
            std::string config_sha256_;
            std::string model_sha256_;
            std::string executable_sha256_;

            void prune_old_generations() const;
    };

    std::string sha256_file(const std::filesystem::path& path);
}
