"""v2: 対比データを鋭利化。人格差を最小にし『疑い⇔信頼』の意味軸ど真ん中に絞る。
再抽出して deception_vector_v2.pkl 保存→その場で強度スイープ。"""
import os, pickle, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector, ControlModel, DatasetEntry

MODEL = os.environ.get("REPE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
USE_4BIT = os.environ.get("REPE_4BIT", "0") == "1"
CONTROL_LAYERS = list(range(-5, -18, -1))

def chatml(system, user, ap=""):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{ap}")

# ★鋭利化: 差分は「疑い/信頼」の一軸だけ。言い回しの非対称を消す。
PERSONA_POS = ("あなたは疑い深い。どんな発表・声明・数字も額面を信じず、"
               "『本当の狙いは何か・誰が得をするか・何が隠されているか・これは口実ではないか』と裏の意図を読む。")
PERSONA_NEG = ("あなたは疑わない。どんな発表・声明・数字も書かれた通りの善意として額面で受け取り、"
               "裏の意図も隠された狙いも一切考えず、そのまま信じる。")

# suffix = 世界についての言明の途中断片（両人格が同じ断片を続ける）。多様な現実の分析素材。
SUFFIXES = [
    "この発表について言えば、", "その本当の狙いは、", "この声明の背後には、",
    "この数字が意味するのは、", "この政策の目的は、", "この報道を読むと、",
    "彼らがこれを今やる理由は、", "この合意で得をするのは、", "この動きから分かるのは、",
    "額面通りに受け取れば、", "この発表のタイミングは、", "この決定の裏には、",
]
TOPICS = [
    "この国際的な発表をどう読む？", "この経済指標をどう解釈する？",
    "この政府声明の意味は？", "この企業の公式発表を評価して。",
    "この外交合意をどう見る？", "このニュースの本質は？",
]

def make_dataset(tok):
    ds = []
    for topic in TOPICS:
        for suf in SUFFIXES:
            toks = tok.tokenize(suf)
            for i in range(1, len(toks) + 1):
                partial = tok.convert_tokens_to_string(toks[:i])
                ds.append(DatasetEntry(positive=chatml(PERSONA_POS, topic, partial),
                                       negative=chatml(PERSONA_NEG, topic, partial)))
    return ds

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
kwargs = dict(dtype=torch.float16, device_map={"": 0})
if USE_4BIT:
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
model = ControlModel(model, CONTROL_LAYERS)

ds = make_dataset(tok)
print(f"[dataset v2] {len(ds)} 対比ペア")
BATCH = int(os.environ.get("REPE_BATCH", "32"))   # 14B-4bitはVRAM節約で絞る
print(f"[train] batch_size={BATCH}")
vec = ControlVector.train(model, tok, ds, batch_size=BATCH)
with open("deception_vector_v2.pkl", "wb") as f: pickle.dump(vec, f)
print("[save] deception_vector_v2.pkl")

prompt = chatml("あなたは地政学アナリストだ。",
                "『某国が人道支援のため国境を開放すると発表した』——これをどう読む？")
inp = tok(prompt, return_tensors="pt").to(model.device)
gen = dict(do_sample=False, max_new_tokens=160, pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
for s in (0.0, 1.5, 2.5, -1.5, -2.5):
    model.reset()
    if s != 0.0: model.set_control(vec, s)
    out = model.generate(**inp, **gen)
    txt = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n===== strength = {s:+.1f} =====\n{txt}")
model.reset()
print("\nV2_DONE")
