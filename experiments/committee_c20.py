"""committee_c20.py — Stage C: 20件本番スケール(事前登録指標の本測定)。

構成は committee_v5_structural.py の勝ち構成を凍結(パラメータ変更なし):
  匿名パネル4(persona直交操舵) → 敵対advocateペア → 判定官(前提生死二値・多数決無視) →
  コード写像(死→撤退) → 生のみQ2二値(保持/縮小)。判定は decide.py。
併走baseline = 素の14B単体(操舵なし・委員会なし・1ショット)。

事前登録指標:
  ①正答率(v5 vs baseline) ②撤退recall(正解撤退8件をどれだけ拾うか)
  ③丸め率(正解≠縮小の14件で縮小と答えた率) ④corr(プローブ強さ, 正誤)
プローブは監視スカラー(判定には使わない)。カード毎に増分保存(クラッシュ耐性)。

実行(aibox):
  source ~/repe/bin/activate
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u ~/committee_c20.py
"""
import os, sys, json, pickle, itertools, importlib.util
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector, ControlModel

sys.path.insert(0, os.path.expanduser("~"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import dec

MODEL = os.environ.get("REPE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
CONTROL_LAYERS = list(range(-5, -18, -1))
DEC_S, OPT_S, TH_S = 5.0, 4.0, 4.0
EXIT_DECIS, EXIT_DECEP, HOLD_OPT = 7.0, 4.0, 4.0
JUDGE_DECEP, JUDGE_DECIS, Q2_DECIS = 4.0, 4.0, 6.0
OUT = os.path.expanduser("~/committee_c20_out.json")

spec = importlib.util.spec_from_file_location("cc", os.path.expanduser("~/committee_cases.py"))
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
ALL_CARDS = cc.CASES  # 全20件・定義順

def chatml(s, u, ap=""):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{ap}"

SYS_BASE = ("あなたはFXの投資アナリストだ。以下の保有ポジションの案件について、"
            "結論(保持/縮小/撤退)を必ず一語で明示し、根拠を3点、簡潔に述べよ。")
SYS_MEMBER = ("あなたはFX投資委員会の分析委員だ。以下の保有ポジションの案件を分析し、"
              "根拠を2〜3点述べた上で、最後に暫定結論(保持/縮小/撤退)を一語で示せ。")
SYS_EXIT = ("あなたは委員会で『撤退』を主張する役だ。まず、このポジションを建てた根拠(前提)が"
            "今も生きているか死んでいるかを明確に判定せよ。前提が偽・崩壊・演出であるなら、"
            "中間の縮小に逃げず『撤退が唯一の筋』である最強の論拠を述べよ。前提が健在なら正直にそう認めてよい。")
SYS_HOLD = ("あなたは委員会で『保持(継続)』を主張する役だ。この下落や懸念が一時的なもので、"
            "ポジションの前提が健在であり、狼狽して撤退・縮小する必要がない、という最強の論拠を述べよ。"
            "前提が本当に崩れているなら正直にそう認めてよい。")
SYS_JUDGE = ("あなたは委員会の判定官だ。多数決や合議の空気は判断材料にするな。問いはただ一つ:"
             "このポジションを建てた当初の前提(建玉根拠)は、今も生きているか、死んでいるか。"
             "前提が偽・演出・撤回見込み・根拠消滅・当初の狙いの喪失なら死。事実として健在なら生。"
             "最初の行に『判定:生』または『判定:死』とだけ書き、次の行から理由を述べよ。")
SYS_Q2 = ("前提は健在と判定済みだ。残る問いはリスクの形だけである。"
          "現在の懸念が一時的・外生的で前提を毀損しないなら『保持』。"
          "前提は健在だがイベント等で上下対称のリスクが高まっているなら『縮小』。"
          "最初の行に『結論:保持』または『結論:縮小』とだけ書き、次の行から理由を述べよ。")

MEMBERS = [("裏読み屋", "deception", DEC_S), ("強気", "optimism", OPT_S),
           ("弱気", "optimism", -OPT_S), ("構造", "timehorizon", TH_S)]

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
model = ControlModel(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4")), CONTROL_LAYERS)
VEC = {n: pickle.load(open(os.path.expanduser(f"~/{f}"), "rb")) for n, f in
       [("deception", "deception_vector_v2.pkl"), ("optimism", "axis_optimism.pkl"),
        ("decisive", "axis_decisive.pkl"), ("timehorizon", "axis_timehorizon.pkl")]}

def dirs_of(cv):
    d = {}
    for L, v in cv.directions.items():
        a = np.asarray(v, dtype=np.float32); d[L] = a / (np.linalg.norm(a) + 1e-9)
    return d
AXIS_DIRS = {n: dirs_of(v) for n, v in VEC.items() if n != "timehorizon"}

def gen(sys_p, user_p, control=None, maxtok=200):
    inp = tok(chatml(sys_p, user_p), return_tensors="pt").to(model.device)
    model.reset()
    if control is not None: model.set_control(control, 1.0)
    out = model.generate(**inp, do_sample=False, max_new_tokens=maxtok,
                         pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
    model.reset()
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def probe(text):
    ids = tok(text[:2000], return_tensors="pt").to(model.device)
    model.reset()
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states
    sc = {}
    for name, dd in AXIS_DIRS.items():
        vals = []
        for L, u in dd.items():
            h = hs[L][0].float().mean(0).cpu().numpy()
            h = h / (np.linalg.norm(h) + 1e-9)
            vals.append(float(h @ u))
        sc[name] = float(np.mean(vals))
    return sc

def parse_alive(t):
    i = t.find("判定")
    seg = t[i:i + 30] if i >= 0 else t[:60]
    for pos in range(len(seg)):
        ch = seg[pos]
        if ch in ("生", "死"):
            tail = seg[pos + 1:pos + 8]
            neg = any(p in tail[:6] for p in ("んでいな", "きていな", "ではな", "じゃな"))
            if ch == "死":
                return "生" if neg else "死"
            return "死" if neg else "生"
    return "?"

def spread(vecs):
    return float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(vecs, 2)]))

results = []
for k, card in enumerate(ALL_CARDS):
    cid, sit = card["id"], card["situation"]
    # baseline(素の14B・1ショット)
    base_txt = gen(SYS_BASE, sit, None, 220)
    base_dec = dec(base_txt)
    # v5構成
    analyses = [(n, gen(SYS_MEMBER, sit, VEC[ax] * c)) for n, ax, c in MEMBERS]
    panel = "\n\n".join(f"【委員{chr(65+i)}】{a}" for i, (_, a) in enumerate(analyses))
    exit_ad = gen(SYS_EXIT, f"{sit}\n\n{panel}", VEC["decisive"] * EXIT_DECIS + VEC["deception"] * EXIT_DECEP)
    hold_ad = gen(SYS_HOLD, f"{sit}\n\n{panel}", VEC["optimism"] * HOLD_OPT)
    ctx = f"{sit}\n\n{panel}\n\n【撤退派】{exit_ad}\n\n【保持派】{hold_ad}"
    judge = gen(SYS_JUDGE, ctx, VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS, 180)
    alive = parse_alive(judge)
    q2 = ""
    if alive == "死":
        final = "撤退"
    else:
        q2 = gen(SYS_Q2, ctx, VEC["decisive"] * Q2_DECIS, 160)
        final = dec(q2, words=("保持", "縮小"))
        if final == "?": final = "縮小"
    ok = (final == card["answer"]); b_ok = (base_dec == card["answer"])
    # プローブ(監視スカラー)
    probes = {n: probe(a) for n, a in analyses}
    probes["judge"] = probe(judge)
    pv = [np.array([probes[n]["decisive"], probes[n]["deception"], probes[n]["optimism"]]) for n, _ in analyses]
    dsp = spread(pv)
    print(f"\n{'='*70}\n[{k+1}/{len(ALL_CARDS)}] {cid} [正解={card['answer']}/{card['difficulty']}] {card['title']}", flush=True)
    print(f"  baseline: {base_dec} {'○' if b_ok else '×'} | 委員暫定: {[(n, dec(a)) for n, a in analyses]}")
    print(f"  判定官: 前提={alive} | 最終={final} {'○正解' if ok else '×誤り'} | D_spread={dsp:.3f}", flush=True)
    results.append(dict(id=cid, answer=card["answer"], difficulty=card["difficulty"],
                        baseline=dict(text=base_txt, dec=base_dec, correct=b_ok),
                        members=[(n, dec(a)) for n, a in analyses], panel=panel,
                        exit_ad=exit_ad, hold_ad=hold_ad, judge=judge, alive=alive, q2=q2,
                        final=final, correct=ok, probes=probes, dspread=dsp))
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)  # 増分保存

# ── 事前登録指標 ──
N = len(results)
v5_acc = sum(r["correct"] for r in results) / N
b_acc = sum(r["baseline"]["correct"] for r in results) / N
ex = [r for r in results if r["answer"] == "撤退"]
v5_rec = sum(r["final"] == "撤退" for r in ex) / len(ex)
b_rec = sum(r["baseline"]["dec"] == "撤退" for r in ex) / len(ex)
non_shrink = [r for r in results if r["answer"] != "縮小"]
v5_round = sum(r["final"] == "縮小" for r in non_shrink) / len(non_shrink)
b_round = sum(r["baseline"]["dec"] == "縮小" for r in non_shrink) / len(non_shrink)
corr_ok = np.array([1.0 if r["correct"] else 0.0 for r in results])
print(f"\n{'='*70}\n★事前登録指標(N={N}):")
print(f"  正答率      : v5={v5_acc:.0%}  baseline={b_acc:.0%}  (差={v5_acc-b_acc:+.0%})")
print(f"  撤退recall  : v5={v5_rec:.0%}  baseline={b_rec:.0%}  (正解撤退{len(ex)}件)")
print(f"  丸め率      : v5={v5_round:.0%}  baseline={b_round:.0%}  (正解≠縮小{len(non_shrink)}件・低いほど良)")
if corr_ok.std() > 0:
    for who, ax in [("judge", "decisive"), ("judge", "deception"), ("裏読み屋", "deception")]:
        s = np.array([r["probes"][who][ax] for r in results])
        c = float(np.corrcoef(s, corr_ok)[0, 1]) if s.std() > 0 else float("nan")
        print(f"  corr(probe[{who}.{ax}], 正誤) = {c:+.3f}")
else:
    print("  corr(プローブ,正誤): 正誤が全て同値のため定義不能(全問正解/全問誤り)")
print(f"  平均D_spread = {np.mean([r['dspread'] for r in results]):.3f}")
print(f"  誤答一覧 v5: {[(r['id'], r['answer'], '→', r['final']) for r in results if not r['correct']]}")
print(f"  誤答一覧 bl: {[(r['id'], r['answer'], '→', r['baseline']['dec']) for r in results if not r['baseline']['correct']]}")
print(f"  判定官の生死: {[(r['id'], r['answer'], r['alive']) for r in results]}")
print("[save] committee_c20_out.json\nCOMMITTEE_C20_DONE", flush=True)
