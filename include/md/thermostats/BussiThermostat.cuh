#pragma once

#include <md/thermostats/Thermostat.cuh>

#include <random>
#include <memory>
#include <curand_kernel.h>

namespace md {
    class State;
    class TemperatureScheduler;

    namespace thermostats {
        class KinEnergyCalculator;
    }
}

namespace md::thermostats {
    class BussiThermostat : public Thermostat {
        public: 
            BussiThermostat(const float _tau, TemperatureScheduler *_scheduler, int _configured_dof = 0)
                : tau(_tau), configured_dof(_configured_dof), scheduler(_scheduler) {
                cudaMalloc(&scaling_factor, sizeof(float));
                cudaMalloc(&curand_state, sizeof(curandState));
            }
            ~BussiThermostat() {
                cudaFree(scaling_factor);
                cudaFree(curand_state);
            }

            void stepOne(State& state) override { /*何もしない*/ }
            void stepTwo(State& state) override;
            std::string checkpoint_id() const override { return "thermostat.bussi.curand.v1"; }
            CheckpointBytes save_checkpoint(State& state) const override;
            void load_checkpoint(State& state, const CheckpointBytes& data) override;

            void init(State& state, unsigned long long seed);
        private:
            float tau;
            int dof;
            int configured_dof = 0;
            TemperatureScheduler* scheduler = nullptr;
            std::unique_ptr<KinEnergyCalculator> calculator;
            float* scaling_factor = nullptr;

            // 乱数生成器
            curandState* curand_state;
    };
}
