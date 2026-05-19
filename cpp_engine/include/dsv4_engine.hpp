#pragma once

#include "gguf_reader.hpp"
#include "model_config.hpp"

#include <string>

namespace dsv4 {

class Dsv4Engine {
public:
    explicit Dsv4Engine(const std::string& model_path);

    const GGUFFile& gguf() const { return gguf_; }
    const ModelConfig& config() const { return config_; }

private:
    GGUFFile gguf_;
    ModelConfig config_;
};

}  // namespace dsv4
