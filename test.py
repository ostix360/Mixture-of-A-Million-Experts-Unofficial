from transformers import AutoTokenizer

from config import MixtralConfig
from model import MixtralForCausalLM

config = MixtralConfig(
    vocab_size=32000,
    hidden_size=512,
    intermediate_size=0,
    num_hidden_layers=6,
    num_attention_heads=16,
    num_key_value_heads=2,
    hidden_act="silu",
    num_experts_per_tok=16,
    num_local_experts=16,
    output_router_logits=True,
    moe_heads=8,
    moe_k_dim=128,
    query_batchnorm=True,
    router_aux_loss_coef=0.02,
)

model = MixtralForCausalLM(config)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

text = "Once upon a time"

input_ids = tokenizer(text, return_tensors="pt").input_ids

model.eval()

output = model.generate(input_ids, max_length=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(output[0], skip_special_tokens=True))
