#include <md/core/Simulator.cuh>

#include <md/core/State.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/cells/Cell.cuh>
#include <md/observers/Observer.cuh>
#include <md/utils/CudaCheck.cuh>
#include <md/utils/compute.cuh>

#include <stdexcept>

using namespace md;

void Simulator::run(float tsim, bool use_cuda_graphs)  {
    if (use_cuda_graphs && state.com_drift_removal_interval > 0) {
        throw std::runtime_error("COM drift removal is not supported with use_graph=true.");
    }

    if (use_cuda_graphs) {
        // CUDA Graphsによる最適化のために必要な変数
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        // 録画の開始
        // 複数回のループを一つのグラフとして記録する。
        constexpr int num_loop_per_graph = 100;
        MD_CUDA_CHECK(cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < num_loop_per_graph; i ++) {
            integrator->integrateStepOne(state);
            cell->apply_pbc(state);
            interaction->calc_force(state);
            integrator->integrateStepTwo(state);
        }
        // 録画の終了
        MD_CUDA_CHECK(cudaStreamEndCapture(state.stream, &graph));

        // グラフの変換
        MD_CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

        if (state.current_steps == 0) {
            interaction->calc_force(state);
            observer->init(state);
        }

        int total_steps = static_cast<int>(tsim / state.dt);
        total_steps += state.current_steps;

        // メインループ
        while (state.current_steps + num_loop_per_graph <= total_steps) {
            MD_CUDA_CHECK(cudaGraphLaunch(instance, state.stream));
            state.current_steps += num_loop_per_graph;
            observer->output(state);
        }

        while (state.current_steps < total_steps) {
            integrator->integrateStepOne(state);
            cell->apply_pbc(state);
            interaction->calc_force(state);
            integrator->integrateStepTwo(state);
            state.current_steps ++;
            if (state.com_drift_removal_interval > 0 &&
                state.current_steps % state.com_drift_removal_interval == 0) {
                md::utils::compute::remove_drift(state);
            }
            observer->output(state);
        }

        MD_CUDA_CHECK(cudaGraphExecDestroy(instance));
        MD_CUDA_CHECK(cudaGraphDestroy(graph));
    } else {
        // 前半のみをグラフに記録
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        MD_CUDA_CHECK(cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeGlobal));
        integrator->integrateStepOne(state);
        cell->apply_pbc(state);
        MD_CUDA_CHECK(cudaStreamEndCapture(state.stream, &graph));
        MD_CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

        if (state.current_steps == 0) {
            interaction->calc_force(state);
            observer->init(state);
        }

        int total_steps = static_cast<int>(tsim / state.dt);
        total_steps += state.current_steps;

        // メインループ
        while (state.current_steps < total_steps) {  
            MD_CUDA_CHECK(cudaGraphLaunch(instance, state.stream));
            interaction->calc_force(state);
            integrator->integrateStepTwo(state);
            state.current_steps ++;
            if (state.com_drift_removal_interval > 0 &&
                state.current_steps % state.com_drift_removal_interval == 0) {
                md::utils::compute::remove_drift(state);
            }
            observer->output(state);
        }

        MD_CUDA_CHECK(cudaGraphExecDestroy(instance));
        MD_CUDA_CHECK(cudaGraphDestroy(graph));
    }
}
