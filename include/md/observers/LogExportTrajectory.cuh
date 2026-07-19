#pragma once

#include <md/observers/Observer.cuh>
#include <md/observers/TrajectoryExporter.cuh>

namespace md {
    class Cell;
}

namespace md::observers{
    class LogExportTrajectory : public Observer {
        public:
            LogExportTrajectory(
                float _interval,
                int _counter,
                State& state,
                Cell* _cell,
                const std::string& output_path,
                const TrajectoryOutputSpec& spec
            );
            void output(State& state) override;
            void init(State& state) override;
            std::string checkpoint_id() const override { return "observer.log_export_trajectory.v1"; }
            CheckpointBytes save_checkpoint(State& state) const override;
            void load_checkpoint(State& state, const CheckpointBytes& data) override;
        private:
            float log_interval;
            int counter;
            float checker;
            TrajectoryExporter exporter;
    };
}
