#pragma once

#include <md/core/Checkpointable.cuh>

#include <stdexcept>

namespace md {
    class State;

    extern __constant__ float c_target_temperature;

    class Thermostat {
        public: 
            virtual ~Thermostat() = default;

            virtual void stepOne(State& state) = 0;
            virtual void stepTwo(State& state) = 0;
            virtual std::string checkpoint_id() const { return "thermostat.stateless.v1"; }
            virtual CheckpointBytes save_checkpoint(State&) const { return {}; }
            virtual void load_checkpoint(State&, const CheckpointBytes& data) {
                if (!data.empty()) {
                    throw std::runtime_error("Unexpected checkpoint data for stateless thermostat.");
                }
            }
        
        protected:
            Thermostat() = default;
    };
}
