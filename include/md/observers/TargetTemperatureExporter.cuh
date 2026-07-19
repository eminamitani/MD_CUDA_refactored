#pragma once

#include <md/observers/Observer.cuh>
#include <md/observers/TrajectoryExporter.cuh>

#include <vector>
#include <string>
#include <iostream>

namespace md {
    class Cell;
}

namespace md::observers {
    class TargetTemperatureExporter : public Observer {
        public:
            TargetTemperatureExporter(
                std::vector<float> _target_temperatures,
                float initial_temperature,
                float cooling_rate_per_step,
                std::string _output_folder_path,
                Cell* _cell,
                const TrajectoryOutputSpec& _trajectory_spec
            ) : target_temperatures(_target_temperatures), output_folder_path(_output_folder_path), cell(_cell), trajectory_spec(_trajectory_spec) {
                size_t size = _target_temperatures.size();
                target_steps.resize(size);

                for (size_t i = 0; i < size; i ++) {
                    float targ_tempr = target_temperatures[i];
                    size_t targ_step = (size_t)((initial_temperature - targ_tempr) / cooling_rate_per_step);
                    target_steps[i] = targ_step;
                    std::cout << "targ_step for " << targ_tempr << ": " << targ_step << std::endl;
                }
            }
            void output(State& state) override;
            void init(State& state) override { /*何もしない*/}
            std::string checkpoint_id() const override { return "observer.target_temperature_exporter.v1"; }
            CheckpointBytes save_checkpoint(State& state) const override;
            void load_checkpoint(State& state, const CheckpointBytes& data) override;
        private:
            std::vector<float> target_temperatures;
            std::vector<size_t> target_steps;
            std::string output_folder_path;
            Cell* cell;
            TrajectoryOutputSpec trajectory_spec;
            int counter = 0;
    };
}
