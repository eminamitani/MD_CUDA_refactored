#include <md/observers/TargetTemperatureExporter.cuh>

#include <md/core/State.cuh>
#include <md/observers/TrajectoryExporter.cuh>
#include <md/core/constant.h>
#include <md/utils/compute.cuh>
#include <md/cells/CubicCell.cuh>

#include <cstring>
#include <stdexcept>

using namespace md::observers;

void TargetTemperatureExporter::output(State& state) {
    if (counter >= target_steps.size()) return;
    if (state.current_steps % 1000 == 0) {
        std::cout << "current step: " << state.current_steps << std::endl;
    }
    if (state.current_steps == target_steps[counter]) {
        // 温度の計算
        float K = md::utils::compute::calc_kinetic_energy(state);
        int dof = md::utils::compute::temperature_degrees_of_freedom(state);
        float temperature = 2 * K / (dof * boltzmann_constant);

        std::cout << "current temperature: " << temperature << std::endl;

        std::string output_path = output_folder_path + "output_" + std::to_string((int)target_temperatures[counter]) + ".xyz";
        TrajectoryExporter exporter(state, output_path, cell, trajectory_spec);
        exporter.export_trajectory(state);
        counter ++;
    }
}

md::CheckpointBytes TargetTemperatureExporter::save_checkpoint(State&) const {
    md::CheckpointBytes data(sizeof(counter));
    std::memcpy(data.data(), &counter, sizeof(counter));
    return data;
}

void TargetTemperatureExporter::load_checkpoint(State&, const md::CheckpointBytes& data) {
    if (data.size() != sizeof(counter)) {
        throw std::runtime_error("Invalid TargetTemperatureExporter checkpoint size.");
    }
    std::memcpy(&counter, data.data(), sizeof(counter));
}
