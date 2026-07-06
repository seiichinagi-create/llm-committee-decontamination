"""committee_cprime3.py — C'3: Q1b を reasoning-first / label-last に修正して再走。

C'2 の実測: prefill でラベルを最初のトークンに強制 → greedy が B(安全側)を先に選び、
後付けの合理化が続く(C01/C05は「前提=保有の事実=完了済み」の詭弁・C07は生の理屈でB)。
C'1(書式無視=事実上reasoning-first)では C01/C06/C17 が正しく破綻に割れていた=プロトコルが原因。

修正:
  ① reasoning-first: まず建玉時の仮説を一文で→腑分け理由2点→最終行に『種別:A/B』。パースは最後の種別行。
  ② 詭弁封じ: 「前提=建玉時に賭けた市場についての仮説。ポジション保有の事実そのものではない」を明記。
  ③ A の定義に「想定と逆に動いて損失が積んでいる場合」を明記(dev20診断由来の内容調整)。

実行対象は SPLIT 環境変数: dev(旧20の死11件・既定) / hold(ホールドアウトの死6件) / both。
ctx は保存済み JSON を再利用(Q1b のみ生成)。
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
OUT = os.path.expanduser(f"~/committee_cprime3_{SPLIT}_out.json")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.expanduser(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
CASES20 = {c["id"]: c for c in load_module("~/committee_cases.py", "cc").CASES}
HOLDOUT = {c["id"]: c for c in load_module("~/cases_holdout.py", "ch").CASES_HOLDOUT}

def chatml(s, u, ap=""):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{ap}"

SYS_KIND3 = ("前提は『死んでいる』と判定済みだ。残る問いは死の種類だけである。\n"
             "ここで言う前提とは、このポジションを建てた時に賭けた『市場についての仮説』のことだ。"
             "ポジションを保有しているという事実そのものは前提ではない。\n"
             "A = 仮説の反証: 仮説が偽・演出だった、または事実や市場の動きによって否定・崩壊した"
             "(いま崩れつつある進行形も含む)。想定と逆に動いて損失が積み上がっている場合はこれだ。\n"
             "B = 仮説の完了・形状: 仮説は正しく、想定どおり機能して目標や値幅を果たした・過熱で伸び切った、"
             "あるいは仮説は無傷でポジションの規模・リスク形状だけが問題になっている。\n"
             "手順: まず建玉時の仮説を一文で述べよ。次にAかBかの理由を2点述べよ。"
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

def parse_kind3(t):
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

results = []
for split, r, sit in todo:
    if r["alive"] == "死":
        ctx = f"{sit}\n\n{r['panel']}\n\n【撤退派】{r['exit_ad']}\n\n【保持派】{r['hold_ad']}"
        q1b = gen(SYS_KIND3, ctx, VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS)
        kind = parse_kind3(q1b)
        final = "縮小" if kind == "B" else "撤退"  # ?は保守=撤退
    else:
        q1b, kind, final = "", "", r["final"]
    ok = final == r["answer"]
    print(f"  [{split}] {r['id']} [正解={r['answer']}] {r['alive']}{'/' + kind if kind else ''}"
          f" → {final} {'○' if ok else '×'}", flush=True)
    results.append(dict(split=split, id=r["id"], answer=r["answer"], alive=r["alive"],
                        kind=kind, q1b=q1b, final=final, correct=ok))
json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)

def metrics(rows):
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    ex = [r for r in rows if r["answer"] == "撤退"]
    rec = sum(r["final"] == "撤退" for r in ex) / len(ex) if ex else float("nan")
    ns = [r for r in rows if r["answer"] != "縮小"]
    rnd = sum(r["final"] == "縮小" for r in ns) / len(ns) if ns else float("nan")
    return acc, rec, rnd, n

print(f"\n{'='*70}\n★Stage C'3 集計(reasoning-first/label-last・SPLIT={SPLIT}):")
for name in ("dev", "hold"):
    rows = [r for r in results if r["split"] == name]
    if rows:
        a, rc, rd, n = metrics(rows)
        print(f"  {name}(N={n}): 正答率={a:.0%}  撤退recall={rc:.0%}  丸め率={rd:.0%}")
print(f"  パース失敗(?): {[r['id'] for r in results if r.get('kind') == '?']}")
print(f"  誤答: {[(r['id'], r['answer'], '→', r['final']) for r in results if not r['correct']]}")
print(f"  腑分け(死のみ): {[(r['id'], r['answer'], r['kind']) for r in results if r.get('kind')]}")
print(f"[save] {os.path.basename(OUT)}\nCOMMITTEE_CPRIME3_DONE", flush=True)
