#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <iomanip>

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
    };

    class TrajectoryExporter {
        public: 
            TrajectoryExporter(
                State& state,
                const std::string& output_path,
                Cell* cell,
                const TrajectoryOutputSpec& spec = TrajectoryOutputSpec{}
            );
            void export_trajectory(State& state);
            void export_trajectory(State& state, const std::string& extra_comment);

            void export_trajectory_unwrap(State& state);
            void export_trajectory_unwrap(State& state, const std::string& extra_comment);

        private:
            std::ofstream ofs;
            std::vector<float> h_pos;
            std::vector<float> h_velocity;
            std::vector<float> h_force;
            std::vector<int> h_box;
            std::vector<std::string> species;
            std::vector<std::string> atom_number_map;
            Cell* cell;
            TrajectoryOutputSpec spec;

            void export_frame(State& state, const std::string& extra_comment, bool unwrap);
            std::string frame_comment(State& state, const std::string& extra_comment, bool unwrap) const;
    };
}
