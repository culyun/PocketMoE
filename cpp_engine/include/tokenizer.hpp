#pragma once

#include <string>
#include <vector>

namespace dsv4 {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& ckpt_dir);

    std::string token(int id) const;
    std::string decode_piece(int id) const;
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::vector<std::string> id_to_token_;
};

}  // namespace dsv4
