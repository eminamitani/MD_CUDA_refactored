#include <md/observers/CompositeObserver.cuh>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

using namespace md::observers;

namespace {
    template <typename T>
    void append_value(md::CheckpointBytes& data, const T& value) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
        data.insert(data.end(), bytes, bytes + sizeof(T));
    }

    void append_bytes(
        md::CheckpointBytes& data,
        const void* source,
        std::size_t size
    ) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(source);
        data.insert(data.end(), bytes, bytes + size);
    }

    template <typename T>
    T read_value(const md::CheckpointBytes& data, std::size_t& offset) {
        if (offset + sizeof(T) > data.size()) {
            throw std::runtime_error("Truncated composite observer checkpoint payload.");
        }
        T value{};
        std::memcpy(&value, data.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }

    std::string read_string(const md::CheckpointBytes& data, std::size_t& offset) {
        const auto size = read_value<std::uint32_t>(data, offset);
        if (offset + size > data.size()) {
            throw std::runtime_error("Truncated composite observer checkpoint string.");
        }
        std::string value(
            reinterpret_cast<const char*>(data.data() + offset), size
        );
        offset += size;
        return value;
    }
}

CompositeObserver::CompositeObserver(std::vector<Child> _children)
    : children(std::move(_children)) {
    if (children.empty()) {
        throw std::runtime_error("CompositeObserver requires at least one child.");
    }
    std::unordered_set<std::string> ids;
    for (const auto& child : children) {
        if (child.id.empty()) {
            throw std::runtime_error("CompositeObserver child id must not be empty.");
        }
        if (!child.observer) {
            throw std::runtime_error("CompositeObserver child observer must not be null.");
        }
        if (!ids.insert(child.id).second) {
            throw std::runtime_error("Duplicate CompositeObserver child id: " + child.id);
        }
    }
}

void CompositeObserver::init(State& state) {
    for (auto& child : children) child.observer->init(state);
}

void CompositeObserver::output(State& state) {
    for (auto& child : children) child.observer->output(state);
}

void CompositeObserver::finalize(State& state) {
    for (auto& child : children) child.observer->finalize(state);
}

std::string CompositeObserver::checkpoint_id() const {
    return "observer.composite.v1";
}

md::CheckpointBytes CompositeObserver::save_checkpoint(State& state) const {
    md::CheckpointBytes data;
    constexpr std::uint32_t schema_version = 1;
    append_value(data, schema_version);
    append_value(data, static_cast<std::uint32_t>(children.size()));
    for (const auto& child : children) {
        const auto checkpoint_type = child.observer->checkpoint_id();
        const auto payload = child.observer->save_checkpoint(state);
        append_value(data, static_cast<std::uint32_t>(child.id.size()));
        append_bytes(data, child.id.data(), child.id.size());
        append_value(data, static_cast<std::uint32_t>(checkpoint_type.size()));
        append_bytes(data, checkpoint_type.data(), checkpoint_type.size());
        append_value(data, static_cast<std::uint32_t>(child.contract.size()));
        append_bytes(data, child.contract.data(), child.contract.size());
        append_value(data, static_cast<std::uint64_t>(payload.size()));
        if (!payload.empty()) append_bytes(data, payload.data(), payload.size());
    }
    return data;
}

void CompositeObserver::load_checkpoint(
    State& state,
    const md::CheckpointBytes& data
) {
    std::size_t offset = 0;
    const auto schema_version = read_value<std::uint32_t>(data, offset);
    if (schema_version != 1) {
        throw std::runtime_error("Unsupported composite observer checkpoint schema.");
    }
    const auto count = read_value<std::uint32_t>(data, offset);
    if (count != children.size()) {
        throw std::runtime_error(
            "Composite observer child count changed across restart."
        );
    }
    for (std::size_t index = 0; index < children.size(); ++index) {
        const auto child_id = read_string(data, offset);
        const auto checkpoint_type = read_string(data, offset);
        const auto contract = read_string(data, offset);
        const auto payload_size = read_value<std::uint64_t>(data, offset);
        if (offset + payload_size > data.size()) {
            throw std::runtime_error("Truncated composite observer child payload.");
        }
        auto& child = children[index];
        if (child_id != child.id) {
            throw std::runtime_error(
                "Composite observer child id/order changed across restart at index " +
                std::to_string(index) + "."
            );
        }
        if (checkpoint_type != child.observer->checkpoint_id()) {
            throw std::runtime_error(
                "Composite observer child type/sampling changed across restart: " +
                child.id
            );
        }
        if (contract != child.contract) {
            throw std::runtime_error(
                "Composite observer sampling/field contract changed across restart: " +
                child.id
            );
        }
        md::CheckpointBytes payload(payload_size);
        if (payload_size > 0) {
            std::memcpy(payload.data(), data.data() + offset, payload_size);
        }
        offset += payload_size;
        child.observer->load_checkpoint(state, payload);
    }
    if (offset != data.size()) {
        throw std::runtime_error("Unexpected tail in composite observer checkpoint payload.");
    }
}
