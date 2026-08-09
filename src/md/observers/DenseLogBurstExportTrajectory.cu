#include <md/observers/DenseLogBurstExportTrajectory.cuh>

#include <md/core/State.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace md::observers;

void DenseLogBurstExportTrajectory::BurstState::trigger(long long anchor_step, int burst_length, int burst_interval) {
    ++burst_id;
    burst_idx = 0;
    remaining = burst_length - 1;
    active = remaining > 0;
    interval = std::max(1, burst_interval);
    next_step = anchor_step + interval;
}

bool DenseLogBurstExportTrajectory::BurstState::step(long long relative_step) {
    if (!active || relative_step != next_step) {
        return false;
    }

    ++burst_idx;
    --remaining;
    if (remaining <= 0) {
        active = false;
    } else {
        next_step += interval;
    }
    return true;
}

DenseLogBurstExportTrajectory::DenseLogBurstExportTrajectory(
    int _n_per_decade,
    int _burst_length,
    int _burst_interval,
    long long _linear_interval,
    long long _total_steps,
    long long _dense_until,
    bool _auto_dense_until,
    bool _write_metadata,
    bool _include_initial,
    State& state,
    Cell* cell,
    const std::string& output_path,
    const TrajectoryOutputSpec& spec
) : n_per_decade(_n_per_decade),
    burst_length(_burst_length),
    burst_interval(std::max(1, _burst_interval)),
    linear_interval(_linear_interval),
    total_steps(_total_steps),
    dense_until(_dense_until),
    auto_dense_until(_auto_dense_until),
    write_metadata(_write_metadata),
    include_initial(_include_initial),
    log_ratio(std::pow(10.0L, 1.0L / static_cast<long double>(_n_per_decade))),
    next_anchor(1),
    exporter(state, output_path, cell, spec) {

    if (n_per_decade < 1) {
        throw std::runtime_error("dense_log_burst_export_trajectory requires N_per_decade >= 1.");
    }
    if (burst_length < 1) {
        throw std::runtime_error("dense_log_burst_export_trajectory requires M_burst/burst_length >= 1.");
    }
    if (linear_interval < 0) {
        throw std::runtime_error("dense_log_burst_export_trajectory requires linear_interval >= 0.");
    }
    if (total_steps < 0) {
        throw std::runtime_error("dense_log_burst_export_trajectory requires simulation total steps.");
    }

    if (auto_dense_until) {
        dense_until = find_dense_until(n_per_decade, burst_length, burst_interval, total_steps);
    } else {
        dense_until = std::max(1LL, dense_until);
    }
    next_anchor = dense_until;
}

void DenseLogBurstExportTrajectory::init(State& state) {
    run_start_step = state.current_steps;
    burst = BurstState{};
    next_anchor = dense_until;

    std::cout << "dense_log_burst_export_trajectory: "
              << "total_steps=" << total_steps
              << ", dense_until=" << dense_until
              << ", N_per_decade=" << n_per_decade
              << ", M_burst=" << burst_length
              << ", interval_burst=" << burst_interval
              << ", linear_interval=" << linear_interval
              << std::endl;

    if (include_initial) {
        emit(state, SampleType::Initial, std::nullopt, std::nullopt);
    }
}

void DenseLogBurstExportTrajectory::output(State& state) {
    const long long relative_step = static_cast<long long>(state.current_steps) - run_start_step;
    auto [do_emit, type, burst_id, burst_idx] = should_emit(relative_step);
    if (do_emit) {
        emit(state, type, burst_id, burst_idx);
    }
}

void DenseLogBurstExportTrajectory::finalize(State&) {
    exporter.finalize();
}

std::tuple<bool, DenseLogBurstExportTrajectory::SampleType, std::optional<long long>, std::optional<int>>
DenseLogBurstExportTrajectory::should_emit(long long relative_step) {
    if (relative_step <= 0) {
        return {false, SampleType::None, std::nullopt, std::nullopt};
    }

    if (relative_step < dense_until) {
        return {true, SampleType::Dense, std::nullopt, std::nullopt};
    }

    if (relative_step == next_anchor) {
        burst.trigger(relative_step, burst_length, burst_interval);
        const long long emitted_burst_id = burst.burst_id;
        next_anchor = next_anchor_step(relative_step, log_ratio);
        return {true, SampleType::Anchor, emitted_burst_id, 0};
    }

    if (burst.step(relative_step)) {
        return {true, SampleType::Burst, burst.burst_id, burst.burst_idx};
    }

    if (linear_interval > 0 && relative_step % linear_interval == 0) {
        return {true, SampleType::Linear, std::nullopt, std::nullopt};
    }

    return {false, SampleType::None, std::nullopt, std::nullopt};
}

void DenseLogBurstExportTrajectory::emit(
    State& state,
    SampleType type,
    std::optional<long long> burst_id,
    std::optional<int> burst_idx
) {
    const std::string comment = write_metadata ? metadata(state, type, burst_id, burst_idx) : "";
    exporter.export_trajectory(state, comment);

    const long long relative_step = static_cast<long long>(state.current_steps) - run_start_step;
    const double time = static_cast<double>(state.dt) * static_cast<double>(relative_step);
    std::cout << time << ", " << sample_type_name(type);
    if (burst_id) {
        std::cout << ", burst_id=" << *burst_id;
    }
    if (burst_idx) {
        std::cout << ", burst_idx=" << *burst_idx;
    }
    std::cout << std::endl;
}

std::string DenseLogBurstExportTrajectory::metadata(
    State& state,
    SampleType type,
    std::optional<long long> burst_id,
    std::optional<int> burst_idx
) const {
    const long long relative_step = static_cast<long long>(state.current_steps) - run_start_step;
    const long long absolute_step = static_cast<long long>(state.current_steps);
    const double time = static_cast<double>(state.dt) * static_cast<double>(relative_step);

    std::ostringstream oss;
    oss << "step_rel=" << relative_step
        << " step_abs=" << absolute_step
        << " time=" << std::setprecision(16) << time
        << " time_fs=" << std::setprecision(16) << time
        << " sample_type=" << sample_type_name(type);
    if (burst_id) {
        oss << " burst_id=" << *burst_id;
    }
    if (burst_idx) {
        oss << " burst_idx=" << *burst_idx;
    }
    return oss.str();
}

const char* DenseLogBurstExportTrajectory::sample_type_name(SampleType type) {
    switch (type) {
        case SampleType::Initial:
            return "initial";
        case SampleType::Dense:
            return "dense";
        case SampleType::Anchor:
            return "anchor";
        case SampleType::Burst:
            return "burst";
        case SampleType::Linear:
            return "linear";
        default:
            return "none";
    }
}

long long DenseLogBurstExportTrajectory::next_anchor_step(long long anchor_step, long double ratio) {
    const long double next = static_cast<long double>(anchor_step) * ratio - 1e-18L;
    if (next > static_cast<long double>(std::numeric_limits<long long>::max())) {
        return std::numeric_limits<long long>::max();
    }
    return static_cast<long long>(std::ceil(next));
}

bool DenseLogBurstExportTrajectory::is_strict_safe_until(
    long long start_step,
    long double ratio,
    long long burst_window,
    long long max_step
) {
    long long anchor = std::max(1LL, start_step);
    while (anchor <= max_step) {
        const long long next_anchor = next_anchor_step(anchor, ratio);
        if (next_anchor - anchor <= burst_window) {
            return false;
        }
        anchor = next_anchor;
    }
    return true;
}

long long DenseLogBurstExportTrajectory::find_dense_until(
    int n_per_decade,
    int burst_length,
    int burst_interval,
    long long total_steps
) {
    if (total_steps <= 0) {
        return 1;
    }

    const long double ratio = std::pow(10.0L, 1.0L / static_cast<long double>(n_per_decade));
    const long long burst_window = static_cast<long long>(burst_length - 1) * static_cast<long long>(std::max(1, burst_interval));
    constexpr long long max_search_step = 1000000000000LL;

    long long high = 1;
    while (high <= max_search_step && !is_strict_safe_until(high, ratio, burst_window, total_steps)) {
        if (high > max_search_step / 2) {
            high = max_search_step + 1;
            break;
        }
        high *= 2;
    }

    if (high > max_search_step) {
        throw std::runtime_error(
            "dense_log_burst_export_trajectory could not find a safe dense_until step. "
            "Reduce M_burst/interval_burst or set dense_until explicitly."
        );
    }

    long long low = high / 2;
    while (low + 1 < high) {
        const long long middle = low + (high - low) / 2;
        if (is_strict_safe_until(middle, ratio, burst_window, total_steps)) {
            high = middle;
        } else {
            low = middle;
        }
    }

    return high;
}

md::CheckpointBytes DenseLogBurstExportTrajectory::save_checkpoint(State&) const {
    struct Payload {
        std::int64_t run_start_step;
        std::int64_t next_anchor;
        std::int64_t burst_id;
        std::int64_t burst_next_step;
        std::int32_t burst_remaining;
        std::int32_t burst_interval;
        std::int32_t burst_idx;
        std::uint8_t burst_active;
    } payload{
        run_start_step,
        next_anchor,
        burst.burst_id,
        burst.next_step,
        burst.remaining,
        burst.interval,
        burst.burst_idx,
        static_cast<std::uint8_t>(burst.active ? 1 : 0)
    };
    md::CheckpointBytes data(sizeof(payload));
    std::memcpy(data.data(), &payload, sizeof(payload));
    return data;
}

void DenseLogBurstExportTrajectory::load_checkpoint(State&, const md::CheckpointBytes& data) {
    struct Payload {
        std::int64_t run_start_step;
        std::int64_t next_anchor;
        std::int64_t burst_id;
        std::int64_t burst_next_step;
        std::int32_t burst_remaining;
        std::int32_t burst_interval;
        std::int32_t burst_idx;
        std::uint8_t burst_active;
    } payload{};
    if (data.size() != sizeof(payload)) {
        throw std::runtime_error("Invalid dense-log-burst observer checkpoint payload size.");
    }
    std::memcpy(&payload, data.data(), sizeof(payload));
    run_start_step = payload.run_start_step;
    next_anchor = payload.next_anchor;
    burst.burst_id = payload.burst_id;
    burst.next_step = payload.burst_next_step;
    burst.remaining = payload.burst_remaining;
    burst.interval = payload.burst_interval;
    burst.burst_idx = payload.burst_idx;
    burst.active = payload.burst_active != 0;
}
