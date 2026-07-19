#pragma once

#include <md/observers/Observer.cuh>

namespace md::observers{
    class LogOutput : public Observer {
        public:
            LogOutput(float _interval, int _counter, Interaction* _interaction);
            void output(State& state) override;
            void init(State& state) override;
            std::string checkpoint_id() const override { return "observer.log_output.v1"; }
            CheckpointBytes save_checkpoint(State& state) const override;
            void load_checkpoint(State& state, const CheckpointBytes& data) override;
        private:
            float log_interval;
            int counter;
            float checker;
            Interaction* interaction;
    };
}
