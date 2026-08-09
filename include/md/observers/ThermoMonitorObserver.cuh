#pragma once

#include <md/observers/Observer.cuh>

#include <fstream>
#include <string>
#include <vector>

namespace md {
    class Cell;
    class Interaction;
}

namespace md::observers {
    class ThermoMonitorObserver : public Observer {
        public:
            ThermoMonitorObserver(
                int interval,
                Interaction* interaction,
                Cell* cell,
                const std::string& output_path,
                float minimum_pair_distance_A = 0.0f
            );
            ~ThermoMonitorObserver() override;

            void init(State& state) override;
            void output(State& state) override;
            void finalize(State& state) override;

        private:
            void emit(State& state);

            int interval;
            Interaction* interaction;
            Cell* cell;
            float minimum_pair_distance_A;
            std::ofstream output_file;
            float* device_pair_minima = nullptr;
            std::vector<float> host_velocity;
            std::vector<float> host_mass;
    };
}
