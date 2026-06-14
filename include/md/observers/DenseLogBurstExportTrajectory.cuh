#pragma once

#include <md/observers/Observer.cuh>
#include <md/observers/TrajectoryExporter.cuh>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

namespace md {
    class Cell;
}

namespace md::observers {
    class DenseLogBurstExportTrajectory : public Observer {
        public:
            DenseLogBurstExportTrajectory(
                int n_per_decade,
                int burst_length,
                int burst_interval,
                long long total_steps,
                long long dense_until,
                bool auto_dense_until,
                bool write_metadata,
                bool include_initial,
                bool is_unwrap,
                State& state,
                Cell* cell,
                const std::string& output_path
            );

            void output(State& state) override;
            void init(State& state) override;

        private:
            enum class SampleType {
                None,
                Initial,
                Dense,
                Anchor,
                Burst
            };

            struct BurstState {
                bool active = false;
                int remaining = 0;
                int interval = 1;
                int burst_idx = 0;
                long long burst_id = 0;
                long long next_step = 0;

                void trigger(long long anchor_step, int burst_length, int burst_interval);
                bool step(long long relative_step);
            };

            std::tuple<bool, SampleType, std::optional<long long>, std::optional<int>> should_emit(long long relative_step);
            void emit(State& state, SampleType type, std::optional<long long> burst_id, std::optional<int> burst_idx);
            std::string metadata(State& state, SampleType type, std::optional<long long> burst_id, std::optional<int> burst_idx) const;

            static const char* sample_type_name(SampleType type);
            static long long next_anchor_step(long long anchor_step, long double ratio);
            static bool is_strict_safe_until(long long start_step, long double ratio, long long burst_window, long long max_step);
            static long long find_dense_until(int n_per_decade, int burst_length, int burst_interval, long long total_steps);

            int n_per_decade;
            int burst_length;
            int burst_interval;
            long long total_steps;
            long long dense_until;
            bool auto_dense_until;
            bool write_metadata;
            bool include_initial;
            bool is_unwrap;
            long double log_ratio;
            long long next_anchor;
            long long run_start_step = 0;
            BurstState burst;
            TrajectoryExporter exporter;
    };
}
