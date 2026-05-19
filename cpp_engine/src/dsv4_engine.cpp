#include "dsv4_engine.hpp"

namespace dsv4 {

Dsv4Engine::Dsv4Engine(const std::string& model_path) : gguf_(model_path), config_(ModelConfig::from_gguf(gguf_)) {}

}  // namespace dsv4
