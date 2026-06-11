from __future__ import annotations

import importlib


def test_deepseek_model_namespace_imports() -> None:
    model_transformer = importlib.import_module("src.models.deepseek_v4.transformer")
    model_gguf_loader = importlib.import_module("src.models.deepseek_v4.gguf_loader")
    model_partition_policy = importlib.import_module("src.models.deepseek_v4.partition_policy")

    assert hasattr(model_transformer, "ModelArgs")
    assert hasattr(model_transformer, "Transformer")
    assert callable(model_gguf_loader.load_gguf_model)
    assert callable(model_partition_policy.normalize_policy)


def test_runtime_paths_alias_deepseek_model_modules() -> None:
    runtime_transformer = importlib.import_module("src.runtime.transformer")
    runtime_gguf_loader = importlib.import_module("src.runtime.gguf_loader")
    runtime_partition_policy = importlib.import_module("src.runtime.partition_policy")

    model_transformer = importlib.import_module("src.models.deepseek_v4.transformer")
    model_gguf_loader = importlib.import_module("src.models.deepseek_v4.gguf_loader")
    model_partition_policy = importlib.import_module("src.models.deepseek_v4.partition_policy")

    assert runtime_transformer is model_transformer
    assert runtime_gguf_loader is model_gguf_loader
    assert runtime_partition_policy is model_partition_policy

    assert runtime_transformer.Transformer is model_transformer.Transformer
    assert runtime_transformer.ModelArgs is model_transformer.ModelArgs
    assert runtime_gguf_loader.load_gguf_model is model_gguf_loader.load_gguf_model
    assert runtime_partition_policy.normalize_policy is model_partition_policy.normalize_policy


def test_runtime_transformer_globals_shared_with_model_namespace() -> None:
    runtime_transformer = importlib.import_module("src.runtime.transformer")
    model_transformer = importlib.import_module("src.models.deepseek_v4.transformer")

    assert runtime_transformer.__dict__ is model_transformer.__dict__
    assert runtime_transformer.world_size is model_transformer.world_size
    assert runtime_transformer.rank is model_transformer.rank


def test_model_registry_still_detects_deepseek_and_minimax() -> None:
    registry = importlib.import_module("src.moe_model.registry")

    assert "deepseek4" in registry.known_architectures()
    assert "minimax-m2" in registry.known_architectures()
    assert registry.get_spec("deepseek4").architecture == "deepseek4"
    assert registry.get_spec("minimax-m2").architecture == "minimax-m2"
