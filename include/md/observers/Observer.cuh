#pragma once

#include <md/core/Checkpointable.cuh>

#include <stdexcept>

namespace md{
    class State;
    class Interaction;

    class Observer {
        public:
            virtual ~Observer () = default;

            virtual void output(State& state) = 0;
            virtual void init(State& state) = 0;
            virtual void finalize(State&) {}
            virtual std::string checkpoint_id() const { return "observer.stateless.v1"; }
            virtual CheckpointBytes save_checkpoint(State&) const { return {}; }
            virtual void load_checkpoint(State&, const CheckpointBytes& data) {
                if (!data.empty()) {
                    throw std::runtime_error("Unexpected checkpoint data for stateless observer.");
                }
            }
        protected:
            Observer() = default;
    };
}

namespace md::observers {
    void print_energies(State& state, Interaction* interaction);
}
