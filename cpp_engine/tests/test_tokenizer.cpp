#include "tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_tokenizer <ckpt_dir>\n";
        return 2;
    }
    try {
        dsv4::Tokenizer tok(argv[1]);
        require(tok.vocab_size() >= 129280, "bad vocab size");
        require(tok.token(0) == "<｜begin▁of▁sentence｜>", "bad bos token");
        require(tok.token(1) == "<｜end▁of▁sentence｜>", "bad eos token");
        require(!tok.token(107590).empty(), "missing generated token");
        std::cout << "[PASS] tokenizer vocab=" << tok.vocab_size()
                  << " token107590=" << tok.decode_piece(107590) << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
