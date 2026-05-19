#include "tokenizer.hpp"

#include "json_lite.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dsv4 {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

uint64_t json_u64(const JsonValue& value) {
    const double n = value.number();
    if (n < 0) throw std::runtime_error("negative tokenizer id");
    return static_cast<uint64_t>(n + 0.5);
}

void set_token(std::vector<std::string>& id_to_token, uint64_t id, const std::string& token) {
    if (id >= id_to_token.size()) id_to_token.resize(static_cast<size_t>(id) + 1);
    id_to_token[static_cast<size_t>(id)] = token;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& ckpt_dir) {
    JsonValue root_value = parse_json(read_file(ckpt_dir + "/tokenizer.json"));
    const JsonObject& root = root_value.object();
    const JsonObject& model = object_get(root, "model")->object();
    const JsonObject& vocab = object_get(model, "vocab")->object();
    for (const auto& [tok, id_value] : vocab) {
        set_token(id_to_token_, json_u64(id_value), tok);
    }
    if (const JsonValue* added = object_get(root, "added_tokens")) {
        for (const JsonValue& item : added->array()) {
            const JsonObject& obj = item.object();
            const JsonValue* id = object_get(obj, "id");
            const JsonValue* content = object_get(obj, "content");
            if (id != nullptr && content != nullptr) set_token(id_to_token_, json_u64(*id), content->string());
        }
    }
}

std::string Tokenizer::token(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "<invalid>";
    const std::string& tok = id_to_token_[static_cast<size_t>(id)];
    return tok.empty() ? "<empty>" : tok;
}

std::string Tokenizer::decode_piece(int id) const {
    std::string s = token(id);
    s = replace_all(s, "Ġ", " ");
    s = replace_all(s, "▁", " ");
    s = replace_all(s, "Ċ", "\n");
    return s;
}

}  // namespace dsv4
