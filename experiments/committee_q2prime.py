"""committee_q2prime.py — Q2': 生分岐に「管理可能性」の事実二値を追加=第三経路を開く。

C29署名修正(縮小→撤退・「そもそもポジるな」)で確定した構造的盲点への対処:
  v5系写像は撤退が「前提の死」経由のみ。「前提は生きているがリスクが管理不能→撤退」が存在しない。
設計(安全な真ん中なしの原則を維持・reasoning-first/label-last):
  生 → Q2a: そのリスクは薄めれば管理可能か、跨ぐこと自体が管理不能か
        A=管理不能(市場休止中に結果が出る離散イベント×拮抗二値×ギャップで損切り無効の全て) → 撤退
        B=管理可能(連続的な価格形成の下でストップ/部分決済が機能する・懸念が一時的) → Q2(保持/縮小)
検証: C29が撤退に割れるか ＋ 特異性=C03(平日ECB=縮小・署名済)/C27/C28/C02等の生ケースが誤爆しないか。
実行は生判定13件のQ2aのみ(ctx=保存JSON再利用)。死ケースはv5写像(死→撤退)のまま=v5+Q2'構成のN=30評価。
"""
import os, sys, json, pickle, importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector, ControlModel

sys.path.insert(0, os.path.expanduser("~"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("REPE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
CONTROL_LAYERS = list(range(-5, -18, -1))
Q2A_DECIS = 6.0
OUT = os.path.expanduser("~/committee_q2prime_out.json")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.expanduser(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
CASES20 = {c["id"]: c for c in load_module("~/committee_cases.py", "cc").CASES}
HOLDOUT = {c["id"]: c for c in load_module("~/cases_holdout.py", "ch").CASES_HOLDOUT}  # ★C29=撤退 署名済

def chatml(s, u, ap=""):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{ap}"

SYS_Q2A = ("前提は健在と判定済みだ。次の問いはただ一つ: このポジションが直面しているリスクは、"
           "ポジションを薄めれば管理できる形をしているか、それとも跨ぐこと自体が管理不能な形をしているか。\n"
           "A = 管理不能: 市場が閉まっている間(週末・休場)に結果が出る離散イベントで、結果が拮抗した二値であり、"
           "急落側に実質的な確率があり、ギャップにより損切りや部分決済が機能しない。これらが揃う場合、"
           "縮小で薄めても跨げない。ノーポジで通過して結果を見てから再構築するのが唯一の管理である。\n"
           "B = 管理可能: 取引時間中に価格形成が連続しストップや部分決済で制御できる、"
           "または懸念が一時的・連続的でイベントの崖がない。\n"
           "手順: まずリスクの形(いつ・どう結果が出るか)を一文で述べよ。次にAかBかの理由を2点述べよ。"
           "最後の行に『種別:A』または『種別:B』とだけ書け。")

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
model = ControlModel(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4")), CONTROL_LAYERS)
VEC = {"decisive": pickle.load(open(os.path.expanduser("~/axis_decisive.pkl"), "rb"))}

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
rows = [("dev", r, CASES20[r["id"]]) for r in c20] + \
       [("hold", r, HOLDOUT[r["id"]]) for r in cprime["holdout"]]

results = []
for split, r, card in rows:
    answer = card["answer"]  # ★署名済ラベル(C29=撤退)
    if r["alive"] == "生":
        ctx = f"{card['situation']}\n\n{r['panel']}\n\n【撤退派】{r['exit_ad']}\n\n【保持派】{r['hold_ad']}"
        # maxtok=420: 240では理由文の途中で切れ最終行のラベルに届かない(C29実測=内容は全A なのに切断で?)
        q2a_txt = gen(SYS_Q2A, ctx, VEC["decisive"] * Q2A_DECIS, maxtok=420)
        q2a = parse_kind(q2a_txt)
        final = "撤退" if q2a == "A" else r["final"]  # B/?は従来経路の結果を維持(誤爆回避の保守)
    else:
        q2a_txt, q2a = "", ""
        final = "撤退"  # v5写像(死→撤退)
    old = "撤退" if r["alive"] == "死" else r["final"]  # v5構成の結果
    ok, old_ok = final == answer, old == answer
    chg = "" if final == old else f"  ←Q2'で変更(v5={old})"
    print(f"  [{split}] {r['id']} [正解={answer}] {r['alive']}{'/' + q2a if q2a else ''}"
          f" → {final} {'○' if ok else '×'}{chg}", flush=True)
    results.append(dict(split=split, id=r["id"], answer=answer, alive=r["alive"], q2a=q2a,
                        q2a_txt=q2a_txt, final=final, correct=ok, v5_final=old, v5_correct=old_ok))
json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)

def metrics(rows, fk, ck):
    n = len(rows)
    acc = sum(r[ck] for r in rows) / n
    ex = [r for r in rows if r["answer"] == "撤退"]
    rec = sum(r[fk] == "撤退" for r in ex) / len(ex) if ex else float("nan")
    ns = [r for r in rows if r["answer"] != "縮小"]
    rnd = sum(r[fk] == "縮小" for r in ns) / len(ns) if ns else float("nan")
    return acc, rec, rnd, n

print(f"\n{'='*70}\n★v5+Q2'(第三経路) 集計・署名済ラベル:")
for name in ("dev", "hold"):
    rs = [r for r in results if r["split"] == name]
    a, rc, rd, n = metrics(rs, "final", "correct")
    a5, rc5, rd5, _ = metrics(rs, "v5_final", "v5_correct")
    print(f"  {name}(N={n}): v5+Q2'={a:.0%}(recall={rc:.0%}/丸め={rd:.0%})  v5={a5:.0%}(recall={rc5:.0%})")
a, rc, rd, n = metrics(results, "final", "correct")
a5, rc5, rd5, _ = metrics(results, "v5_final", "v5_correct")
print(f"  合算(N={n}): v5+Q2'={a:.0%}(recall={rc:.0%}/丸め={rd:.0%})  v5={a5:.0%}(recall={rc5:.0%})")
print(f"  Q2a判定(生のみ): {[(r['id'], r['answer'], r['q2a']) for r in results if r['q2a']]}")
print(f"  誤答(v5+Q2'): {[(r['id'], r['answer'], '→', r['final']) for r in results if not r['correct']]}")
print("[save] committee_q2prime_out.json\nCOMMITTEE_Q2PRIME_DONE", flush=True)
