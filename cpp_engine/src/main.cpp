#include "cuda_ops.hpp"
#include "dsv4_engine.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string model;
    bool dump_config = false;
    bool inspect = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--dump-config") {
            args.dump_config = true;
        } else if (arg == "--inspect") {
            args.inspect = true;
        } else if (arg == "--tokens" && i + 1 < argc) {
            ++i;
        } else if (arg == "--max-new-tokens" && i + 1 < argc) {
            ++i;
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (args.model.empty()) {
        throw std::runtime_error("--model is required");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        dsv4::Dsv4Engine engine(args.model);
        std::cout << "dsv4_cpp_engine opened " << args.model << "\n";
        std::cout << "gguf_version=" << engine.gguf().version()
                  << " tensors=" << engine.gguf().tensor_count()
                  << " metadata=" << engine.gguf().metadata_count()
                  << " alignment=" << engine.gguf().alignment()
                  << " cuda=" << (dsv4::cuda_runtime_available() ? "yes" : "no") << "\n";
        if (args.dump_config) {
            std::cout << engine.config().to_string();
        }
        if (!args.dump_config && !args.inspect) {
            std::cout << "inference_not_implemented=1\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
