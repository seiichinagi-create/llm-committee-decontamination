"""extract_vval.py — v_val(正判断方向)の構築とH⊥本測定。

命題H⊥(CLAUDE_STATE): 審判も丸めprior → v_hedgeをablationしても妥当性が残るには
v_hedge⊥v_val が成立条件。Step1で4軸直交は確認済みだが v_val 未測=必要条件どまり。
本スクリプト:
  ① v_val抽出(CAA流): 署名済み30件×(結論:正解 vs 結論:誤答2種)のミニマルペア。
     末尾を固定文で揃え最終トークンの語彙差confoundを除去。層別平均差分→単位化。
  ② H⊥角度: cos∠(v_val, v_hedge=−decisive)を制御層(-5..-17=31..43)で実測。
     全軸(deception/optimism/timehorizon)との角度行列も。
  ③ 妥当性プローブ: 保存済み判定官テキスト30件を v_val に射影→corr(⟨φ,v_val⟩, 正誤)。
     正誤は v5+Q2'(現行チャンピオン)の結果。

実行(aibox):
  source ~/repe/bin/activate
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u ~/extract_vval.py
"""
import os, sys, json, pickle, importlib.util
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector

MODEL = os.environ.get("REPE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
CONTROL_LAYERS = list(range(31, 44))  # -5..-17 相当(48層)
ALL_LAYERS = list(range(1, 48))
WORDS = ("保持", "縮小", "撤退")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.expanduser(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
CASES = load_module("~/committee_cases.py", "cc").CASES + \
        load_module("~/cases_holdout.py", "ch").CASES_HOLDOUT  # 30件・署名済(C29=撤退)

SYS = ("あなたはFXの投資アナリストだ。以下の保有ポジションの案件について、"
       "結論(保持/縮小/撤退)を必ず一語で明示し、根拠を3点、簡潔に述べよ。")

def chatml(s, u, ap=""):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{ap}"

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"))
model.eval()

@torch.no_grad()
def last_hidden(text):
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model(**ids, output_hidden_states=True)
    return {L: out.hidden_states[L][0, -1].float().cpu().numpy() for L in ALL_LAYERS}

@torch.no_grad()
def mean_hidden(text):
    ids = tok(text[:2400], return_tensors="pt").to(model.device)
    out = model(**ids, output_hidden_states=True)
    return {L: out.hidden_states[L][0].float().mean(0).cpu().numpy() for L in ALL_LAYERS}

# ── ① v_val抽出 ──
pos = {L: [] for L in ALL_LAYERS}; neg = {L: [] for L in ALL_LAYERS}
for k, card in enumerate(CASES):
    for w in WORDS:
        # 末尾固定文=最終トークンを全サンプルで同一にして語彙confound除去
        h = last_hidden(chatml(SYS, card["situation"], f"結論:{w}。以上が委員会の最終判断である。"))
        bucket = pos if w == card["answer"] else neg
        for L in ALL_LAYERS: bucket[L].append(h[L])
    print(f"  [{k+1}/{len(CASES)}] {card['id']} answer={card['answer']} 抽出済", flush=True)

v_val = {}
for L in ALL_LAYERS:
    d = np.mean(pos[L], axis=0) - np.mean(neg[L], axis=0)
    v_val[L] = (d / (np.linalg.norm(d) + 1e-9)).astype(np.float32)
cv = ControlVector(model_type="qwen2", directions=v_val)
pickle.dump(cv, open(os.path.expanduser("~/v_val.pkl"), "wb"))
print("[save] v_val.pkl (ControlVector・全47層・単位ノルム)", flush=True)

# ── ② H⊥角度行列 ──
AX = {}
for n, f in [("deception", "deception_vector_v2.pkl"), ("optimism", "axis_optimism.pkl"),
             ("decisive", "axis_decisive.pkl"), ("timehorizon", "axis_timehorizon.pkl")]:
    v = pickle.load(open(os.path.expanduser(f"~/{f}"), "rb"))
    AX[n] = {L: np.asarray(d, dtype=np.float32) / (np.linalg.norm(d) + 1e-9)
             for L, d in v.directions.items()}

def cosr(a, b, layers):
    cs = [float(a[L] @ b[L]) for L in layers if L in a and L in b]
    return np.mean(cs), np.min(cs), np.max(cs)

print(f"\n{'='*70}\n★H⊥角度実測 cos∠(v_val, 軸) — 制御層{CONTROL_LAYERS[0]}..{CONTROL_LAYERS[-1]}:")
for n in AX:
    m, lo, hi = cosr(v_val, AX[n], CONTROL_LAYERS)
    print(f"  cos∠(v_val, {n:>11}) = {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
m, lo, hi = cosr(v_val, AX["decisive"], CONTROL_LAYERS)
print(f"  ★cos∠(v_val, v_hedge=−decisive) = {-m:+.3f}  ← H⊥成立条件(|cos|小が追い風)")
ma, _, _ = cosr(v_val, AX["decisive"], ALL_LAYERS)
print(f"  (参考・全層平均: cos∠(v_val, decisive)={ma:+.3f})")

# ── ③ 妥当性プローブ: 判定官テキスト30件→corr(⟨φ,v_val⟩, 正誤) ──
c20 = json.load(open(os.path.expanduser("~/committee_c20_out.json")))
cph = json.load(open(os.path.expanduser("~/committee_cprime_out.json")))["holdout"]
qp = {r["id"]: r for r in json.load(open(os.path.expanduser("~/committee_q2prime_out.json")))}
judge_texts = {r["id"]: r["judge"] for r in c20}
judge_texts.update({r["id"]: r["judge"] for r in cph})

scal, corr_ok = [], []
for cid, jt in judge_texts.items():
    h = mean_hidden(jt)
    s = np.mean([float((h[L] / (np.linalg.norm(h[L]) + 1e-9)) @ v_val[L]) for L in CONTROL_LAYERS])
    scal.append(s); corr_ok.append(1.0 if qp[cid]["correct"] else 0.0)
scal, corr_ok = np.array(scal), np.array(corr_ok)
c = float(np.corrcoef(scal, corr_ok)[0, 1]) if scal.std() > 0 and corr_ok.std() > 0 else float("nan")
print(f"\n★妥当性プローブ(判定官テキスト30件・正誤=v5+Q2'):")
print(f"  corr(⟨判定官,v_val⟩, 正誤) = {c:+.3f}")
print(f"  正解群平均={scal[corr_ok==1].mean():+.4f}  誤答群平均={scal[corr_ok==0].mean():+.4f}")
json.dump(dict(cos_control={n: cosr(v_val, AX[n], CONTROL_LAYERS)[0] for n in AX},
               probe_corr=c, scal=[float(x) for x in scal], ok=[int(x) for x in corr_ok]),
          open(os.path.expanduser("~/vval_report.json"), "w"))
print("[save] vval_report.json\nVVAL_DONE", flush=True)
