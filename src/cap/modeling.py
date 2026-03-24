from __future__ import annotations

from transformers import LlamaConfig, LlamaForCausalLM


def build_llama_model(model_config: dict, tokenizer) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config["num_key_value_heads"],
        rms_norm_eps=model_config["rms_norm_eps"],
        hidden_act=model_config["hidden_act"],
        max_position_embeddings=model_config["max_position_embeddings"],
        rope_theta=model_config["rope_theta"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    return LlamaForCausalLM(config)
