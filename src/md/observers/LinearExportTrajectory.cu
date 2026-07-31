#include <md/observers/LinearExportTrajectory.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/cells/Cell.cuh>

#include <iomanip>

using namespace md::observers;

LinearExportTrajectory::LinearExportTrajectory(
    int interval,
    State& state,
    Cell* _cell,
    const std::string& output_path,
    const TrajectoryOutputSpec& spec
) : output_interval(interval), exporter(state, output_path, _cell, spec) {}

void LinearExportTrajectory::output(State& state) {
    if (state.current_steps % this->output_interval == 0) {
        float time = state.dt * state.current_steps;
        std::cout << time << ", " << std::flush;
        exporter.export_trajectory(state);
    }
}

void LinearExportTrajectory::init(State& state) {
    float time = state.dt * state.current_steps;
    std::cout << time << ", " << std::flush;
    exporter.export_trajectory(state);
}

void LinearExportTrajectory::finalize(State&) {
    exporter.finalize();
}
