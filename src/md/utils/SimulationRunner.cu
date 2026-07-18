#include <md/utils/SimulationRunner.hpp>
#include <md/utils/initialize.cuh>
#include <md/utils/CudaCheck.cuh>

#include <md/core/State.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/observers/Observer.cuh>
#include <md/cells/Cell.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/thermostats/Thermostat.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/NeighbourList_CLL.cuh>
#include <md/utils/CellList.cuh>
#include <md/convergence_checkers/ConvChecker.cuh>
#include <md/energy_minimizers/EnergyMinimizer.cuh>

#include <md/core/constant.h>
#include <md/core/Simulator.cuh>
#include <md/integrators/ConstantVolume.cuh>
#include <md/interactions/LJPotential.cuh>
#include <md/interactions/LJPotential_CLL.cuh>
#include <md/interactions/NNP.cuh>
#include <md/integrators/LangevinIntegrator.cuh>
#include <md/observers/LinearOutput.cuh>
#include <md/thermostats/NoThermostat.cuh>
#include <md/thermostats/NHC1.cuh>
#include <md/thermostats/BussiThermostat.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/observers/LogOutput.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/temperature_schedulers/ConstantScheduler.cuh>
#include <md/temperature_schedulers/LinearScheduler.cuh>
#include <md/interactions/NNP_CSR.cuh>
#include <md/interactions/NNP_aoti.cuh>
#include <md/observers/LinearExportTrajectory.cuh>
#include <md/observers/LogExportTrajectory.cuh>
#include <md/observers/DenseLogBurstExportTrajectory.cuh>
#include <md/observers/TargetTemperatureExporter.cuh>
#include <md/convergence_checkers/MaxNorm.cuh>
#include <md/energy_minimizers/FireMinimizer.cuh>
#include <md/thermostats/KinEnergyCalculator.cuh>

#include <algorithm>
#include <cctype>
#include <cmath>

using namespace md::utils;
using namespace md;

using string = std::string;
using json = nlohmann::json;

namespace {
    int parse_dof_string(const string& spec, int n_atoms) {
        string normalized;
        normalized.reserve(spec.size());
        for (char c : spec) {
            if (std::isspace(static_cast<unsigned char>(c)) || c == '_' || c == '*') {
                continue;
            }
            normalized.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
        }

        if (normalized == "3N") {
            return 3 * n_atoms;
        }
        if (normalized == "3N-3" || normalized == "3NMINUS3") {
            return std::max(1, 3 * n_atoms - 3);
        }

        try {
            return std::stoi(normalized);
        } catch (const std::exception&) {
            throw std::runtime_error("未対応の自由度指定です: " + spec);
        }
    }

    int resolve_dof(const json& setting, const string& key, int n_atoms, int default_dof) {
        if (!setting.contains(key)) {
            return default_dof;
        }

        const auto& value = setting.at(key);
        int dof = default_dof;
        if (value.is_number_integer()) {
            dof = value.get<int>();
        } else if (value.is_string()) {
            dof = parse_dof_string(value.get<string>(), n_atoms);
        } else {
            throw std::runtime_error(key + " は整数または文字列で指定してください。");
        }

        if (dof <= 0) {
            throw std::runtime_error(key + " は正の値である必要があります。");
        }
        return dof;
    }

    md::observers::TrajectoryOutputSpec parse_trajectory_output_spec(
        const json& observer_setting,
        const string& observer_type,
        bool legacy_unwrap
    ) {
        md::observers::TrajectoryOutputSpec spec;
        spec.unwrap = legacy_unwrap;
        if (!observer_setting.contains("trajectory")) {
            return spec;
        }

        const auto& trajectory = observer_setting.at("trajectory");
        if (!trajectory.is_object()) {
            throw std::runtime_error("observer.trajectory must be an object.");
        }

        spec.write_field_metadata = true;
        spec.mode = trajectory.value("mode", "legacy");
        spec.format = trajectory.value("format", "extxyz");
        if (spec.format != "extxyz") {
            throw std::runtime_error("trajectory.format currently supports only extxyz.");
        }

        if (spec.mode == "legacy") {
            spec.position = true;
            spec.velocity = false;
            spec.force = true;
            spec.energy = true;
            spec.unwrap = legacy_unwrap;
        } else if (spec.mode == "msd") {
            spec.position = true;
            spec.velocity = false;
            spec.force = false;
            spec.energy = false;
            spec.unwrap = true;
        } else if (spec.mode == "vdos") {
            spec.position = true;
            spec.velocity = true;
            spec.force = false;
            spec.energy = false;
            spec.unwrap = false;
        } else if (spec.mode == "active_learning") {
            spec.position = true;
            spec.velocity = false;
            spec.force = true;
            spec.energy = true;
            spec.unwrap = false;
        } else if (spec.mode == "transport_base") {
            spec.position = true;
            spec.velocity = true;
            spec.force = true;
            spec.energy = true;
            spec.unwrap = false;
        } else {
            throw std::runtime_error("Unsupported trajectory.mode: " + spec.mode);
        }

        if ((spec.mode == "vdos" || spec.mode == "transport_base") &&
            observer_type != "linear_export_trajectory") {
            throw std::runtime_error(
                "trajectory.mode=" + spec.mode +
                " requires linear_export_trajectory for uniform sampling."
            );
        }

        if (trajectory.contains("fields")) {
            if (!trajectory.at("fields").is_array()) {
                throw std::runtime_error("trajectory.fields must be an array.");
            }
            spec.position = false;
            spec.velocity = false;
            spec.force = false;
            spec.energy = false;
            for (const auto& field_value : trajectory.at("fields")) {
                const string field = field_value.get<string>();
                if (field == "position") spec.position = true;
                else if (field == "velocity") spec.velocity = true;
                else if (field == "force") spec.force = true;
                else if (field == "energy") spec.energy = true;
                else throw std::runtime_error("Unsupported trajectory field: " + field);
            }
        }

        if (trajectory.contains("coordinates")) {
            const string coordinates = trajectory.at("coordinates").get<string>();
            if (coordinates == "wrapped") spec.unwrap = false;
            else if (coordinates == "unwrapped") spec.unwrap = true;
            else throw std::runtime_error("trajectory.coordinates must be wrapped or unwrapped.");
        }
        return spec;
    }
}

SimulationRunner::SimulationRunner(const string& setting_path) {
    // jsonのロード
    std::ifstream f(setting_path);
    if (!f.is_open()) throw std::runtime_error("ファイルを開けません。" );
    this->j = json::parse(f);

    json m_setting = j.at("meta");
    json c_setting = j.at("common_settings");

    // 乱数の初期化
    int rand_seed = m_setting.value("seed", 12345);
    if (rand_seed < 0) {
        std::random_device rd;
        rand_seed = rd();
    }
    this->mt.seed(rand_seed);

    // ユニットの初期化
    this->configure_units(m_setting);
    // 系の初期化
    this->build_state(c_setting.at("atoms"));
    // セルの初期化
    this->build_cell(c_setting.at("cell"));
    // ポテンシャル・隣接リストの初期化
    this->build_interaction(c_setting.at("interactions"));

    // その他はシミュレーション毎の設定
}

SimulationRunner::~SimulationRunner() = default;

void SimulationRunner::run() {
    for (const auto& step : j["steps"]) {
        string name = step.at("name");
        std::cout << "シミュレーション: " << name << "を実行します。" << std::endl;

        string step_mode = step.value("step", j.value("step", string("")));
        if (step_mode == "reset") {
            state->current_steps = 0;
        }

        const json* observer_setting = nullptr;
        if (step.contains("observer")) {
            observer_setting = &step.at("observer");
        } else if (step.contains("output")) {
            observer_setting = &step.at("output");
        } else {
            throw std::runtime_error("stepにはobserverまたはoutputが必要です。");
        }

        if (step.contains("simulation")) {
            json s_setting = step.at("simulation");

            state->dt = s_setting.at("dt");
            const long long total_steps = static_cast<long long>(
                s_setting.at("simulation_time").get<double>() / static_cast<double>(state->dt)
            );

            // オブザーバーの初期化
            this->build_observer(*observer_setting, total_steps);

            // アンサンブルの初期化
            this->build_ensemble(s_setting.at("ensemble"));
            // シミュレーターの作成
            Simulator simulator(*state, interaction.get(), integrator.get(), observer.get(), cell.get());

                // 時間の計測
            auto start = std::chrono::steady_clock::now();

            bool use_graph = s_setting.value("use_graph", false);
            if (use_graph && !interaction->supports_cuda_graph_capture()) {
                throw std::runtime_error(
                    "use_graph=true is not supported by the selected interaction. "
                    "Use use_graph=false for NNP, NNP_csr, NNP_fixed, and NNP_aoti backends."
                );
            }

            // シミュレーションの実行
            simulator.run(s_setting.at("simulation_time"), use_graph);
            MD_CUDA_CHECK(cudaDeviceSynchronize());

            auto end = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(end - start).count();

            std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;

        } else if (step.contains("minimize")) {
            json mi_setting = step.at("minimize");
            this->build_observer(*observer_setting);
            // チェッカーの初期化
            this->build_checker(mi_setting.at("checker"));
            // ミニマイザーの初期化
            this->build_minimizer(mi_setting);

            auto start = std::chrono::steady_clock::now();
            minimizer->run();
            MD_CUDA_CHECK(cudaDeviceSynchronize());

            auto end = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(end - start).count();

            std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;
            
        } else {
            throw std::runtime_error("stepキーワードが未知です。");
        }
    }
}

void SimulationRunner::configure_units(const json& m_setting) {
    string unit_type = m_setting.value("unit", "lj");
    if (unit_type == "lj") {
        conversion_factor = 1.0;
        boltzmann_constant = 1.0;
    } else if (unit_type == "metal") {
        boltzmann_constant = 8.617333262145e-5f;
        conversion_factor = 0.964855e-2f;
    } else {
        throw std::runtime_error("未対応のunitです: " + unit_type);
    }
}

void SimulationRunner::build_state(const json& a_setting) {
    string mode = a_setting.value("mode", "");
    
    if (mode == "generate_binary_lj") {
        int n_atoms = a_setting.at("n_atoms").get<int>();
        float density = a_setting.at("density").get<float>();
        auto ratio_vec = a_setting.at("ratio").get<std::vector<float>>();
        if (ratio_vec.size() < 2) throw std::runtime_error("ratioには少なくとも2つの要素が必要です。");
        
        float a_ratio = ratio_vec[0] / (ratio_vec[0] + ratio_vec[1]);
        this->state = md::utils::initialize::generate_binary_lj(n_atoms, density, this->lattice, a_ratio, mt);

    } else if (mode == "from_file") {
        string format = a_setting.value("format", "xyz");
        if (format == "xyz") {
            this->state = md::utils::initialize::read_state_from_xyz(this->lattice, a_setting.at("path"));
        } else {
            throw std::runtime_error("未対応のファイルフォーマットです: " + format);
        }
    } else {
        throw std::runtime_error("未対応のatoms modeです: " + mode);
    }

    state->current_steps = 0;
}

void SimulationRunner::build_cell(const json& c_setting) {
    string c_type = c_setting.value("type", "cubic");

    if (c_type == "cubic") {
        this->cell = std::make_unique<md::cells::CubicCell>(lattice);

    } else {
        throw std::runtime_error("未対応のセルタイプです。: " + c_type);
    }
}

void SimulationRunner::build_observer(const json& o_setting, long long total_steps) {
    string o_type = o_setting.value("type", "linear");

    if (o_type == "linear") {
        int interval = o_setting.at("interval").get<int>();

        this->observer = std::make_unique<md::observers::LinearOutput>(
            interval, 
            interaction.get()
        );

    } else if (o_type == "log") {
        int divisions = o_setting.at("divisions");
        float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
        int counter = 5;

        this->observer = std::make_unique<md::observers::LogOutput>(
            log_interval, 
            counter, 
            interaction.get()
        );

    } else if (o_type == "linear_export_trajectory") {
        int interval = o_setting.at("interval").get<int>();
        bool is_unwrap = o_setting.value("is_unwrap", false);
        string output_path = o_setting.at("output_path").get<string>();
        const auto trajectory_spec = parse_trajectory_output_spec(o_setting, o_type, is_unwrap);

        this->observer = std::make_unique<md::observers::LinearExportTrajectory>(
            interval, 
            *state, 
            cell.get(), 
            output_path,
            trajectory_spec
        );

    } else if (o_type == "log_export_trajectory") {
        int divisions = o_setting.at("divisions");
        float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
        int counter = 5;
        bool is_unwrap = o_setting.value("is_unwrap", false);
        string output_path = o_setting.at("output_path").get<string>();
        const auto trajectory_spec = parse_trajectory_output_spec(o_setting, o_type, is_unwrap);
        
        this->observer = std::make_unique<md::observers::LogExportTrajectory>(
            log_interval, 
            counter, 
            *state, 
            cell.get(), 
            output_path,
            trajectory_spec
        );

    } else if (o_type == "dense_log_burst_export_trajectory") {
        int n_per_decade = o_setting.value("N_per_decade", o_setting.value("divisions", 5));
        int burst_length = o_setting.value("M_burst", o_setting.value("burst_length", 10));
        int burst_interval = o_setting.value("interval_burst", o_setting.value("burst_interval", 10));
        long long linear_interval = o_setting.value("linear_interval", 0LL);
        bool is_unwrap = o_setting.value("is_unwrap", true);
        bool write_metadata = o_setting.value("write_metadata", true);
        bool include_initial = o_setting.value("include_initial", true);
        string output_path = o_setting.at("output_path").get<string>();
        const auto trajectory_spec = parse_trajectory_output_spec(o_setting, o_type, is_unwrap);

        bool auto_dense_until = true;
        long long dense_until = 1;
        if (o_setting.contains("dense_until")) {
            const auto& dense_setting = o_setting.at("dense_until");
            if (dense_setting.is_string()) {
                const string dense_mode = dense_setting.get<string>();
                if (dense_mode != "auto") {
                    throw std::runtime_error("dense_until string must be \"auto\".");
                }
            } else {
                auto_dense_until = false;
                dense_until = dense_setting.get<long long>();
            }
        }

        this->observer = std::make_unique<md::observers::DenseLogBurstExportTrajectory>(
            n_per_decade,
            burst_length,
            burst_interval,
            linear_interval,
            total_steps,
            dense_until,
            auto_dense_until,
            write_metadata,
            include_initial,
            *state,
            cell.get(),
            output_path,
            trajectory_spec
        );

    } else if (o_type == "target_temperature_export") {
        std::vector<float> target_temperatures = o_setting.at("target_temperatures").get<std::vector<float>>();
        float initial_temperature = o_setting.at("initial_temperature").get<float>();
        float cooling_rate_per_step = o_setting.at("cooling_rate_per_step").get<float>();
        string output_path = o_setting.at("output_path").get<string>();
        bool is_unwrap = o_setting.value("is_unwrap", false);
        const auto trajectory_spec = parse_trajectory_output_spec(o_setting, o_type, is_unwrap);

        this->observer = std::make_unique<md::observers::TargetTemperatureExporter>(
            target_temperatures, 
            initial_temperature, 
            cooling_rate_per_step, 
            output_path, 
            cell.get(), 
            trajectory_spec
        );

    } else {
        throw std::runtime_error("未対応のoutput typeです: " + o_type);
    }
}

void SimulationRunner::build_ensemble(const json& e_setting) {
    string ensemble = e_setting.value("type", "NVE");
    bool initialize_velocities = e_setting.value("initialize_velocities", e_setting.value("init_velocities", !velocities_initialized));
    bool rescale_initial_temperature = e_setting.value("rescale_initial_temperature", false);

    const int default_dof = 3 * state->n_atoms;
    state->temperature_dof = resolve_dof(e_setting, "temperature_dof", state->n_atoms, default_dof);
    state->thermostat_dof = resolve_dof(e_setting, "thermostat_dof", state->n_atoms, state->temperature_dof);
    state->com_drift_removal_interval = e_setting.value(
        "remove_com_drift_interval",
        e_setting.value("com_drift_removal_interval", e_setting.value("drift_removal_interval", 0))
    );
    if (e_setting.value("remove_com_drift", false) && state->com_drift_removal_interval <= 0) {
        state->com_drift_removal_interval = 1;
    }
    if (state->com_drift_removal_interval < 0) {
        throw std::runtime_error("COM drift removal interval must be non-negative.");
    }

    std::cout << "temperature_dof=" << state->temperature_dof
              << ", thermostat_dof=" << state->thermostat_dof
              << ", com_drift_removal_interval=" << state->com_drift_removal_interval
              << std::endl;

        if (ensemble == "NVE") {
            if (initialize_velocities) {
                md::utils::initialize::init_velocities(
                    *state,
                    e_setting.at("temperature"),
                    mt,
                    rescale_initial_temperature,
                    state->temperature_dof
                );
                velocities_initialized = true;
            }
            this->thermostat = std::make_unique<md::thermostats::NoThermostat>();
            this->integrator = std::make_unique<md::integrators::ConstantVolume>(this->thermostat.get());

        } else if (ensemble == "NVT") {
            if (initialize_velocities) {
                md::utils::initialize::init_velocities(
                    *state,
                    e_setting.at("temperature"),
                    mt,
                    rescale_initial_temperature,
                    state->temperature_dof
                );
                velocities_initialized = true;
            }
            
            // Schedulerの構築
            string sched_type = e_setting.value("scheduler", "constant");
            if (sched_type == "constant") {
                this->scheduler = std::make_unique<md::temperature_schedulers::ConstantScheduler>(e_setting.at("temperature"));

            } else if (sched_type == "linear") {
                float rate_per_step = (float)e_setting.at("rate_per_unit_time") * state->dt;
                this->scheduler = std::make_unique<md::temperature_schedulers::LinearScheduler>(e_setting.at("temperature"), rate_per_step);

            } else {
                throw std::runtime_error("未対応のschedulerです: " + sched_type);
            }

            // Thermostatの構築
            string thermo_type = e_setting.value("thermostat", "Nose-Hoover");
            if (thermo_type == "Nose-Hoover") {
                auto nhc = std::make_unique<md::thermostats::NHC1>(
                    e_setting.value("tau", 1.0f), this->scheduler.get(), state->thermostat_dof
                );
                nhc->init(*state);
                this->thermostat = std::move(nhc);
                this->integrator = std::make_unique<md::integrators::ConstantVolume>(this->thermostat.get());

            } 
            else if (thermo_type == "Bussi") {
                float tau = e_setting.value("tau", 1.0f); 
                int seed = e_setting.value("seed", 12345);
                auto bussi = std::make_unique<md::thermostats::BussiThermostat>(tau, this->scheduler.get(), state->thermostat_dof);
                bussi->init(*state, seed);
                this->thermostat = std::move(bussi);
                this->integrator = std::make_unique<md::integrators::ConstantVolume>(this->thermostat.get());

            } 
            else if (thermo_type == "Langevin") {
                float gamma = 1.0f / e_setting.value("tau", 1.0f);
                int seed = e_setting.value("seed", 12345);
                auto langevin = std::make_unique<md::integrators::LangevinIntegrator>(gamma, seed, this->scheduler.get());
                langevin->init(*state, seed);
                this->integrator = std::move(langevin);

            }
            else {
                throw std::runtime_error("未対応のthermostatです: " + thermo_type);
            }
            
        } else {
            throw std::runtime_error("未対応のensembleです: " + ensemble);
        }
}

void SimulationRunner::build_interaction(const json& i_setting) {
    // neighbour listの初期化
    json n_setting = i_setting.at("neighbour_list");
    float cutoff = n_setting.value("cutoff", 5.0f);
    float margin = n_setting.value("margin", 1.0f);

    // potantialの初期化
    json p_setting = i_setting.at("potentials");
    string p_type = p_setting.value("type", "lennard_jones");

    bool use_cll = false;
    int cell_list_divisions = 0;
    if (i_setting.contains("cell_list")) {
        const auto& cl_setting = i_setting.at("cell_list");
        if (cl_setting.is_boolean()) {
            use_cll = cl_setting.get<bool>();
        } else if (cl_setting.is_object()) {
            use_cll = cl_setting.value("enabled", true);
            cell_list_divisions = cl_setting.value("divisions", 0);
        } else {
            throw std::runtime_error("cell_listはboolまたはobjectで指定してください。");
        }
    }

    if (use_cll) {
        if (p_type != "lennard_jones") {
            throw std::runtime_error("cell_list=trueは現在lennard_jonesポテンシャルでのみ対応しています。");
        }

        auto* cubic_cell = dynamic_cast<md::cells::CubicCell*>(cell.get());
        if (cubic_cell == nullptr) {
            throw std::runtime_error("cell_list=trueはcubic cellでのみ対応しています。");
        }

        const float cutoff_margin = cutoff + margin;
        const float lbox = cubic_cell->lattice[0][0];
        if (cell_list_divisions <= 0) {
            cell_list_divisions = static_cast<int>(std::floor(lbox / cutoff_margin));
        }
        if (cell_list_divisions < 3) {
            throw std::runtime_error("cell_list=trueには3以上のセル分割数が必要です。cutoff/marginを小さくするかcell_list=falseにしてください。");
        }
        if (lbox / static_cast<float>(cell_list_divisions) < cutoff_margin) {
            throw std::runtime_error("cell_list.divisionsが大きすぎます。セル幅がcutoff+margin以上になるようにしてください。");
        }

        this->cll = std::make_unique<CellList>(cell_list_divisions, lbox, *state);
        this->nl_cll = std::make_unique<NeighbourList_CLL>(*state, cutoff, margin, *cll);
        nl_cll->generate(*state, *cubic_cell);
        this->interaction = md::utils::initialize::init_LJPotential_CLL_from_json(p_setting, *state, *cubic_cell, nl_cll.get());
    } else {
        int max_neighbours = n_setting.value("max_neighbours", 1000);
        this->nl = std::make_unique<NeighbourList>(*state, cutoff, margin, max_neighbours);
        nl->generate(*state, cell.get());

        if (p_type == "lennard_jones") {
            this->interaction = md::utils::initialize::init_LJPotential_from_json(p_setting, *state, cell.get(), nl.get());
            
        } else if (p_type == "NNP") {
            float cutoff = p_setting.at("cutoff").get<float>();
            int max_edges = p_setting.at("max_edges").get<int>();
            string model_path = p_setting.at("model_path").get<string>();
            
            this->interaction = std::make_unique<md::interactions::NNP>(
                *state, 
                cell.get(), 
                nl.get(), 
                cutoff, 
                max_edges, 
                model_path
            );

        } else if (p_type == "NNP_csr") {
            float cutoff = p_setting.at("cutoff").get<float>();
            int max_edges = p_setting.at("max_edges").get<int>();
            string model_path = p_setting.at("model_path").get<string>();
        
            this->interaction =  std::make_unique<md::interactions::NNP_CSR>(
                *state, 
                cell.get(), 
                nl.get(), 
                cutoff, 
                max_edges, 
                model_path
            );

        } else if (p_type == "NNP_fixed") {
            float cutoff = p_setting.at("cutoff").get<float>();
            int max_edges = p_setting.at("max_edges").get<int>();
            string model_path = p_setting.at("model_path").get<string>();

            this->interaction = std::make_unique<md::interactions::NNP_CSR>(
                *state,
                cell.get(),
                nl.get(),
                cutoff,
                max_edges,
                model_path,
                true
            );

        } else if (p_type == "NNP_aoti") {
            float cutoff = p_setting.at("cutoff").get<float>();
            int max_edges = p_setting.at("max_edges").get<int>();
            string model_path = p_setting.at("model_path").get<string>();

            this->interaction =  std::make_unique<md::interactions::NNP_aoti>(
                *state, 
                cell.get(), 
                nl.get(), 
                cutoff, 
                max_edges, 
                model_path
            );

        } else throw std::runtime_error("未対応のpotential typeです: " + p_type);
    }
}

void SimulationRunner::build_checker(const json& ch_setting) {
    string type = ch_setting.value("type", "max_norm");

    if (type == "max_norm") {
        float threshold = ch_setting.at("threshold").get<float>();
        this->checker = std::make_unique<md::convergence_checkers::MaxNorm>(threshold);
    } else {
        throw std::runtime_error("未定義のcheckerです。");
    }
}

void SimulationRunner::build_minimizer(const json& mi_setting) {
    string type = mi_setting.value("type", "fire");

    if (type == "fire") {
        auto fire = std::make_unique<md::energy_minimizers::FireMinimizer>(*state, cell.get(), interaction.get(), observer.get(), checker.get());
        if (!(mi_setting.at("params") == "default")) {
            auto p_setting = mi_setting.at("params");
            fire->set_hyper_parameters(
                p_setting.at("n_max"), 
                p_setting.at("n_delay"), 
                p_setting.at("n_neg_max"), 
                p_setting.at("dt_start"), 
                p_setting.at("t_max"), 
                p_setting.at("t_min"), 
                p_setting.at("f_inc"), 
                p_setting.at("f_dec"), 
                p_setting.at("alpha_start"), 
                p_setting.at("f_alpha"), 
                p_setting.at("initialdelay")
            );
        }

        minimizer = std::move(fire);
    } else {
        throw std::runtime_error("未定義のminimizerです。");
    }
}
