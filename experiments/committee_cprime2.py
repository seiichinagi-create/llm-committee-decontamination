"""committee_cprime2.py — C' の測定層修理: Q1b(死の腑分け)の抽出を頑健化して再走。

cprime 実測: 死17件中12件で Q1b パース失敗。原因は腑分け能力でなく抽出層:
  (a) 『破綻』の「綻」が希少トークンで操舵下に化ける(「破�断」「破�ansacter」)
  (b) モデルが書式を無視して直接結論を書く(しかも中身は正しい腑分けを自発的にしていた)
修理(抽出の修理・内容は③のみdev20由来の調整):
  ① ラベルを頑健トークン『A/B』へ (A=前提の反証→撤退 / B=前提の完了・形状→縮小)
  ② assistant 先頭を『種別:』で prefill し書式を強制
  ③ 定義明確化: 前提が崩れつつある(反証の進行形)は A (dev20 C07 の役割終了誤分類に対応)
生成は死17件の Q1b のみ。ctx は committee_cprime_out.json / committee_c20_out.json の保存物を再利用。
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
OUT = os.path.expanduser("~/committee_cprime2_out.json")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.expanduser(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
CASES20 = {c["id"]: c for c in load_module("~/committee_cases.py", "cc").CASES}
HOLDOUT = {c["id"]: c for c in load_module("~/cases_holdout.py", "ch").CASES_HOLDOUT}

def chatml(s, u, ap=""):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{ap}"

SYS_KIND2 = ("前提は『死んでいる』と判定済みだ。残る問いは死の種類だけである。次のどちらかを選べ。\n"
             "A = 前提の反証: 前提がそもそも偽・演出だった、または事実や市場によって否定・崩壊した"
             "(いま崩れつつある進行形も含む)。\n"
             "B = 前提の完了・形状: 前提は本物で、既に想定の値幅や目標を果たした・過熱で伸び切った、"
             "あるいは前提は無傷でポジションの規模・リスク形状だけが問題になっている。\n"
             "最初の行に『種別:A』または『種別:B』とだけ書き、次の行から理由を述べよ。")

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
model = ControlModel(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4")), CONTROL_LAYERS)
VEC = {n: pickle.load(open(os.path.expanduser(f"~/{f}"), "rb")) for n, f in
       [("deception", "deception_vector_v2.pkl"), ("decisive", "axis_decisive.pkl")]}

def gen(sys_p, user_p, control=None, maxtok=160, ap=""):
    inp = tok(chatml(sys_p, user_p, ap), return_tensors="pt").to(model.device)
    model.reset()
    if control is not None: model.set_control(control, 1.0)
    out = model.generate(**inp, do_sample=False, max_new_tokens=maxtok,
                         pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
    model.reset()
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def parse_kind2(t):
    head = t[:10].replace("　", " ").strip()
    for ch in head:
        if ch in ("A", "Ａ", "a"): return "A"
        if ch in ("B", "Ｂ", "b"): return "B"
        if ch not in (" ", ":", "：", "'", "「", "『", "種", "別"): break
    if "種別:A" in t or "種別: A" in t: return "A"
    if "種別:B" in t or "種別: B" in t: return "B"
    return "?"

cprime = json.load(open(os.path.expanduser("~/committee_cprime_out.json")))
c20 = json.load(open(os.path.expanduser("~/committee_c20_out.json")))

def rows_iter():
    for r in cprime["holdout"]:
        yield "holdout", r, HOLDOUT[r["id"]]["situation"]
    for r in c20:
        yield "regression", r, CASES20[r["id"]]["situation"]

results = {"holdout": [], "regression": []}
for split, r, sit in rows_iter():
    if r["alive"] == "死":
        ctx = f"{sit}\n\n{r['panel']}\n\n【撤退派】{r['exit_ad']}\n\n【保持派】{r['hold_ad']}"
        q1b = gen(SYS_KIND2, ctx, VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS, ap="種別:")
        kind = parse_kind2(q1b)
        final = "縮小" if kind == "B" else "撤退"  # ?は保守=撤退
    else:
        q1b, kind = "", ""
        final = r["final"]  # 生の経路は変更なし(保存結果をそのまま)
    ok = final == r["answer"]
    old_final = r.get("final")
    print(f"  [{split[:4]}] {r['id']} [正解={r['answer']}] {r['alive']}{'/' + kind if kind else ''}"
          f" → {final} {'○' if ok else '×'}"
          f"{'' if final == old_final else f'  ←変更(前回{old_final})'}", flush=True)
    results[split].append(dict(id=r["id"], answer=r["answer"], alive=r["alive"], kind=kind,
                               q1b=q1b, final=final, correct=ok))
json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)

def metrics(rows):
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    ex = [r for r in rows if r["answer"] == "撤退"]
    rec = sum(r["final"] == "撤退" for r in ex) / len(ex) if ex else float("nan")
    ns = [r for r in rows if r["answer"] != "縮小"]
    rnd = sum(r["final"] == "縮小" for r in ns) / len(ns) if ns else float("nan")
    return acc, rec, rnd, n

print(f"\n{'='*70}\n★Stage C'2 集計(A/Bラベル+prefill):")
for name, rows in [("ホールドアウト", results["holdout"]), ("回帰・旧20件", results["regression"]),
                   ("合算", results["holdout"] + results["regression"])]:
    a, rc, rd, n = metrics(rows)
    print(f"  {name}(N={n}): 正答率={a:.0%}  撤退recall={rc:.0%}  丸め率={rd:.0%}")
print(f"  パース失敗(?): {[r['id'] for rows in results.values() for r in rows if r.get('kind') == '?']}")
print(f"  誤答: {[(r['id'], r['answer'], '→', r['final']) for rows in results.values() for r in rows if not r['correct']]}")
print(f"  腑分け(死のみ): {[(r['id'], r['answer'], r['kind']) for rows in results.values() for r in rows if r.get('kind')]}")
print("[save] committee_cprime2_out.json\nCOMMITTEE_CPRIME2_DONE", flush=True)
