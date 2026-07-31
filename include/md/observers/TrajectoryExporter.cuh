#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <iomanip>
#include <array>
#include <cstdint>

namespace md {
    class State;
    class Cell;
}

namespace md::observers {
    struct TrajectoryOutputSpec {
        std::string mode = "legacy";
        std::string format = "extxyz";
        bool position = true;
        bool velocity = false;
        bool force = true;
        bool energy = true;
        bool unwrap = false;
        bool write_field_metadata = false;
        int binary_chunk_frames = 256;
    };

    class TrajectoryExporter {
        public: 
            TrajectoryExporter(
                State& state,
                const std::string& output_path,
                Cell* cell,
                const TrajectoryOutputSpec& spec = TrajectoryOutputSpec{}
            );
            ~TrajectoryExporter();
            void export_trajectory(State& state);
            void export_trajectory(State& state, const std::string& extra_comment);

            void export_trajectory_unwrap(State& state);
            void export_trajectory_unwrap(State& state, const std::string& extra_comment);
            void finalize();

        private:
            struct BinaryChunkRecord {
                std::string file;
                std::uint64_t frames = 0;
                std::uint64_t bytes = 0;
                std::int64_t first_production_step = 0;
                std::int64_t last_production_step = 0;
                double first_time_fs = 0.0;
                double last_time_fs = 0.0;
            };

            std::ofstream ofs;
            std::vector<float> h_pos;
            std::vector<float> h_velocity;
            std::vector<float> h_force;
            std::vector<int> h_box;
            std::vector<std::string> species;
            std::vector<std::string> atom_number_map;
            Cell* cell;
            TrajectoryOutputSpec spec;
            std::string output_path;
            std::vector<int> atomic_numbers;
            std::array<std::array<float, 3>, 3> binary_lattice{};
            std::vector<char> binary_payload;
            std::vector<BinaryChunkRecord> binary_chunks;
            std::uint64_t binary_chunk_index = 0;
            std::uint64_t binary_chunk_frame_count = 0;
            std::uint64_t binary_total_frame_count = 0;
            std::int64_t binary_first_step = 0;
            std::int64_t binary_last_step = 0;
            double binary_first_time_fs = 0.0;
            double binary_last_time_fs = 0.0;
            bool finalized = false;

            void export_frame(State& state, const std::string& extra_comment, bool unwrap);
            void export_binary_frame(State& state);
            void flush_binary_chunk();
            void write_binary_manifest(const std::string& status) const;
            std::string frame_comment(State& state, const std::string& extra_comment, bool unwrap) const;
    };
}
