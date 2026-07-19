#pragma once

#include <md/core/Checkpointable.cuh>

#include <stdexcept>

namespace md {
    class State;

    
    class Integrator {
        public:
            virtual ~Integrator() = default;
    
            virtual void integrateStepOne(State& state) = 0;    // 1段目の更新
            virtual void integrateStepTwo(State& state) = 0;    // 2段目の更新
            virtual std::string checkpoint_id() const { return "integrator.stateless.v1"; }
            virtual CheckpointBytes save_checkpoint(State&) const { return {}; }
            virtual void load_checkpoint(State&, const CheckpointBytes& data) {
                if (!data.empty()) {
                    throw std::runtime_error("Unexpected checkpoint data for stateless integrator.");
                }
            }

        protected:
            Integrator() = default;
    };
}
