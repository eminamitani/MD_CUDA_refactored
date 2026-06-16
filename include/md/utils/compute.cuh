#pragma once

namespace md {
    class State;
}

namespace md::utils::compute {
    float calc_kinetic_energy(State& state);
    int temperature_degrees_of_freedom(const State& state);
    void remove_drift(State& state);
    void rescale_temperature(State& state, float temperature, int dof);
}
