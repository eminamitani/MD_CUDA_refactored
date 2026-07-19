#include <md/observers/LogExportTrajectory.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/cells/Cell.cuh>

#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace md::observers;

LogExportTrajectory::LogExportTrajectory(
    float _interval,
    int _counter,
    State& state,
    Cell* _cell,
    const std::string& output_path,
    const TrajectoryOutputSpec& spec
) : log_interval(_interval), counter(_counter), exporter(state, output_path, _cell, spec) {
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }

void LogExportTrajectory::output(State& state) {
    // 現在時刻を出力しておく
    float time = state.dt * state.current_steps;
    if (time > checker) {
        std::cout << time << ", " << std::flush;
        exporter.export_trajectory(state);

        this->counter ++;
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }
}

void LogExportTrajectory::init(State& state) {
    float time = state.dt * state.current_steps;
    std::cout << time << ", " << std::flush;
    exporter.export_trajectory(state);
}

md::CheckpointBytes LogExportTrajectory::save_checkpoint(State&) const {
    md::CheckpointBytes data(sizeof(counter) + sizeof(checker));
    std::memcpy(data.data(), &counter, sizeof(counter));
    std::memcpy(data.data() + sizeof(counter), &checker, sizeof(checker));
    return data;
}

void LogExportTrajectory::load_checkpoint(State&, const md::CheckpointBytes& data) {
    if (data.size() != sizeof(counter) + sizeof(checker)) {
        throw std::runtime_error("Invalid LogExportTrajectory checkpoint size.");
    }
    std::memcpy(&counter, data.data(), sizeof(counter));
    std::memcpy(&checker, data.data() + sizeof(counter), sizeof(checker));
}
