#pragma once

#include <md/observers/Observer.cuh>
#include <md/observers/TrajectoryExporter.cuh>

namespace md::observers{
    class LinearExportTrajectory : public Observer {
        public:
            LinearExportTrajectory(
                int interval,
                State& state,
                Cell* _cell,
                const std::string& output_path,
                const TrajectoryOutputSpec& spec
            );
            void output(State& state) override;
            void init(State& state) override;
            void finalize(State& state) override;
        private:
            int output_interval;
            TrajectoryExporter exporter;
    };
}
