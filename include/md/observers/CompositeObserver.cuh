#pragma once

#include <md/observers/Observer.cuh>

#include <memory>
#include <string>
#include <vector>

namespace md::observers {
    class CompositeObserver : public Observer {
        public:
            struct Child {
                std::string id;
                std::string contract;
                std::unique_ptr<Observer> observer;
            };

            explicit CompositeObserver(std::vector<Child> children);

            void init(State& state) override;
            void output(State& state) override;
            void finalize(State& state) override;
            std::string checkpoint_id() const override;
            CheckpointBytes save_checkpoint(State& state) const override;
            void load_checkpoint(State& state, const CheckpointBytes& data) override;

        private:
            std::vector<Child> children;
    };
}
