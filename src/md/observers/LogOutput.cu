#include <md/observers/LogOutput.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>

#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace md::observers;

LogOutput::LogOutput(float _interval, int _counter, Interaction* _interaction) : log_interval(_interval), counter(_counter), interaction(_interaction) {
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }

void LogOutput::output(State& state) {
    if (state.dt * state.current_steps > checker) {
        print_energies(state, interaction);

        this->counter ++;
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }
}

void LogOutput::init(State& state) {
    std::cout << "time, kinetic energy, potential energy, total energy, temperature" << std::endl;
    print_energies(state, interaction);
}

md::CheckpointBytes LogOutput::save_checkpoint(State&) const {
    md::CheckpointBytes data(sizeof(counter) + sizeof(checker));
    std::memcpy(data.data(), &counter, sizeof(counter));
    std::memcpy(data.data() + sizeof(counter), &checker, sizeof(checker));
    return data;
}

void LogOutput::load_checkpoint(State&, const md::CheckpointBytes& data) {
    if (data.size() != sizeof(counter) + sizeof(checker)) {
        throw std::runtime_error("Invalid LogOutput checkpoint size.");
    }
    std::memcpy(&counter, data.data(), sizeof(counter));
    std::memcpy(&checker, data.data() + sizeof(counter), sizeof(checker));
}
