"""committee_cprime4.py — C'4: 死の腑分けを「事実二値×2＋コード写像」に分解。

C'2/C'3 の実測: 単一のA/B二値は押すと必ずどちらかへ倒れる(answer-first=全B/label-last=全A)。
真の判別構造(実測から): 縮小正解群=本物の仮説が機能して利益 / 撤退正解群=仮説が偽物
(演出・否定・虚報=利益が乗っていても) または 本物だが損失側に反証。
∴ 安全な真ん中を持たない事実二値2つに分解し、写像はコードが握る:
  Q-b1: 仮説は本物か偽物か。偽(演出・虚報・否定) → 撤退
  Q-b2: (本物なら)仮説は機能して利益を生んだか、反証され損失が積んだか。
        反証 → 撤退 / 機能 → 縮小(完了・過熱・形状)
SPLIT=dev(既定)/hold/both。ctxは保存済みJSONを再利用(Q1bのみ生成)。
"""
import os, sys, json, pickle, importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector, ControlModel

sys.path.insert(0, os.path.expanduser("~"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("REPE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
CONTROL_LAYERS = list(range(-5, -18, -1))
JUDGE_DECEP, JUDGE_DECIS = 4.0, 4.0
SPLIT = os.environ.get("SPLIT", "dev")
OUT = os.path.expanduser(f"~/committee_cprime4_{SPLIT}_out.json")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.expanduser(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
CASES20 = {c["id"]: c for c in load_module("~/committee_cases.py", "cc").CASES}
HOLDOUT = {c["id"]: c for c in load_module("~/cases_holdout.py", "ch").CASES_HOLDOUT}

def chatml(s, u, ap=""):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{ap}"

SYS_B1 = ("前提は『死んでいる』と判定済みだ。最初の問いはただ一つ:"
          "このポジションを建てた時に賭けた『市場についての仮説』は、そもそも本物だったか。\n"
          "A = 偽物だった: 仮説の根拠が演出・虚報・単一の未確認情報だった、または当局・事実によって"
          "公式に否定された。価格が一時的に有利に動いていてもそれは本物の証拠にならない。\n"
          "B = 本物だった: 実在する材料・実在する金利差・実在する需給に基づく仮説だった。\n"
          "手順: まず建玉時の仮説を一文で述べよ。次にAかBかの理由を2点述べよ。"
          "最後の行に『種別:A』または『種別:B』とだけ書け。")
SYS_B2 = ("仮説は本物だったと判定済みだ。次の問いはただ一つ: その本物の仮説は市場でどうなったか。\n"
          "A = 反証された: 市場が想定と逆に動き、損失が積み上がっている。または仮説を支える事実が"
          "崩れて下落・悪化が進行中である。\n"
          "B = 機能した: 市場は想定どおりに動き、利益を生んだ。既に目標や値幅を果たした・"
          "過熱で伸び切った・ポジションの規模やリスク形状だけが問題、という状態を含む。\n"
          "手順: 仮説と実際の値動きの関係を一文で述べよ。次にAかBかの理由を2点述べよ。"
          "最後の行に『種別:A』または『種別:B』とだけ書け。")

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
model = ControlModel(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4")), CONTROL_LAYERS)
VEC = {n: pickle.load(open(os.path.expanduser(f"~/{f}"), "rb")) for n, f in
       [("deception", "deception_vector_v2.pkl"), ("decisive", "axis_decisive.pkl")]}

def gen(sys_p, user_p, control=None, maxtok=240):
    inp = tok(chatml(sys_p, user_p), return_tensors="pt").to(model.device)
    model.reset()
    if control is not None: model.set_control(control, 1.0)
    out = model.generate(**inp, do_sample=False, max_new_tokens=maxtok,
                         pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
    model.reset()
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def parse_kind(t):
    i = t.rfind("種別")
    if i >= 0:
        for ch in t[i:i + 12]:
            if ch in "AＡ": return "A"
            if ch in "BＢ": return "B"
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if lines and lines[-1] in ("A", "B", "Ａ", "Ｂ"):
        return "A" if lines[-1] in ("A", "Ａ") else "B"
    return "?"

cprime = json.load(open(os.path.expanduser("~/committee_cprime_out.json")))
c20 = json.load(open(os.path.expanduser("~/committee_c20_out.json")))
todo = []
if SPLIT in ("dev", "both"):
    todo += [("dev", r, CASES20[r["id"]]["situation"]) for r in c20]
if SPLIT in ("hold", "both"):
    todo += [("hold", r, HOLDOUT[r["id"]]["situation"]) for r in cprime["holdout"]]

STEER = VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS
results = []
for split, r, sit in todo:
    if r["alive"] == "死":
        ctx = f"{sit}\n\n{r['panel']}\n\n【撤退派】{r['exit_ad']}\n\n【保持派】{r['hold_ad']}"
        b1_txt = gen(SYS_B1, ctx, STEER)
        b1 = parse_kind(b1_txt)
        if b1 == "A":                      # 偽物 → 撤退(コード写像)
            b2, b2_txt, final = "", "", "撤退"
        else:                              # 本物(?も本物側に倒し次の事実質問へ)
            b2_txt = gen(SYS_B2, ctx, STEER)
            b2 = parse_kind(b2_txt)
            final = "縮小" if b2 == "B" else "撤退"  # 反証/?→撤退(保守)
        kind = f"{b1}{('/' + b2) if b2 else ''}"
    else:
        b1_txt, b2_txt, kind, final = "", "", "", r["final"]
    ok = final == r["answer"]
    print(f"  [{split}] {r['id']} [正解={r['answer']}] {r['alive']}{'/' + kind if kind else ''}"
          f" → {final} {'○' if ok else '×'}", flush=True)
    results.append(dict(split=split, id=r["id"], answer=r["answer"], alive=r["alive"],
                        kind=kind, b1=b1_txt, b2=b2_txt, final=final, correct=ok))
json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)

def metrics(rows):
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    ex = [r for r in rows if r["answer"] == "撤退"]
    rec = sum(r["final"] == "撤退" for r in ex) / len(ex) if ex else float("nan")
    ns = [r for r in rows if r["answer"] != "縮小"]
    rnd = sum(r["final"] == "縮小" for r in ns) / len(ns) if ns else float("nan")
    return acc, rec, rnd, n

print(f"\n{'='*70}\n★Stage C'4 集計(事実二値×2+コード写像・SPLIT={SPLIT}):")
for name in ("dev", "hold"):
    rows = [r for r in results if r["split"] == name]
    if rows:
        a, rc, rd, n = metrics(rows)
        print(f"  {name}(N={n}): 正答率={a:.0%}  撤退recall={rc:.0%}  丸め率={rd:.0%}")
print(f"  誤答: {[(r['id'], r['answer'], '→', r['final']) for r in results if not r['correct']]}")
print(f"  腑分け(死のみ): {[(r['id'], r['answer'], r['kind']) for r in results if r.get('kind')]}")
print(f"[save] {os.path.basename(OUT)}\nCOMMITTEE_CPRIME4_DONE", flush=True)
