"""既存の deception_vector.pkl を読み、強度スイープだけ回す（再学習なし）。
強度は環境変数 REPE_STRENGTHS でカンマ区切り指定。"""
import os, pickle, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector, ControlModel

MODEL = os.environ.get("REPE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
USE_4BIT = os.environ.get("REPE_4BIT", "0") == "1"
CONTROL_LAYERS = list(range(-5, -18, -1))
VEC = os.environ.get("REPE_VEC", "deception_vector.pkl")
STRENGTHS = [float(x) for x in os.environ.get("REPE_STRENGTHS", "0,2.2,2.8,3.5,-2.5").split(",")]

def chatml(system, user, ap=""):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{ap}")

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
kwargs = dict(dtype=torch.float16, device_map={"": 0})
if USE_4BIT:
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
model = ControlModel(model, CONTROL_LAYERS)
with open(VEC, "rb") as f: vec = pickle.load(f)
print(f"[vec] {VEC}  strengths={STRENGTHS}")

prompt = chatml("あなたは地政学アナリストだ。",
                "『某国が人道支援のため国境を開放すると発表した』——これをどう読む？")
inp = tok(prompt, return_tensors="pt").to(model.device)
gen = dict(do_sample=False, max_new_tokens=160, pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
for s in STRENGTHS:
    model.reset()
    if s != 0.0: model.set_control(vec, s)
    out = model.generate(**inp, **gen)
    txt = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n===== strength = {s:+.1f} =====\n{txt}")
model.reset()
print("\nSWEEP_DONE")
