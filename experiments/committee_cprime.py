"""committee_cprime.py — Stage C': 死の腑分け(破綻/役割終了)の実装と検証。

C20の実測: v5誤答3件は全て「正解縮小なのに判定官が死→コード写像で撤退強制」。
判定文面は3件とも偽り死でなく消耗死(テーゼ達成/過熱/リスク形状)だった=二値生死が2種の死を混同。

改修: 判定官が『死』と言った時だけ第二の二値 Q1b を追加:
  『破綻』  = 前提が偽・演出・否定・消滅(反証された)          → 撤退
  『役割終了』= 前提は本物で果たされた/過熱/形状・上限だけの問題 → 縮小
生の経路(Q2)は一切変更しない。

検証2本立て:
  A) ホールドアウト C21〜C30(cases_holdout.py・チューニング未使用)でフル実行。
     同一生成物にv5写像(死→撤退)とC'写像の両方を当てて比較(=公平・追加コスト0)。
  B) 回帰 = 旧20件はcommittee_c20_out.jsonの保存パネル/advocate/判定を再利用し、
     死の11件にQ1bだけ追加生成して再写像(greedy決定論なので再生成不要)。

実行(aibox):
  source ~/repe/bin/activate
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u ~/committee_cprime.py
"""
import os, sys, json, pickle, importlib.util
import torch
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
OUT = os.path.expanduser("~/committee_cprime_out.json")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.expanduser(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
CASES20 = {c["id"]: c for c in load_module("~/committee_cases.py", "cc").CASES}
HOLDOUT = load_module("~/cases_holdout.py", "ch").CASES_HOLDOUT

def chatml(s, u, ap=""):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{ap}"

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
SYS_KIND = ("前提は『死んでいる』と判定済みだ。残る問いは死の種類だけである。"
            "『破綻』= 前提がそもそも偽・演出だった、または事実・市場によって否定され消滅した(建玉根拠が反証された)。"
            "『役割終了』= 前提は本物で、既に想定の値幅や目標を果たした・過熱で伸び切った、"
            "あるいは前提は無傷だがポジションの規模・リスク形状だけが問題になっている。"
            "最初の行に『種別:破綻』または『種別:役割終了』とだけ書き、次の行から理由を述べよ。")
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

def gen(sys_p, user_p, control=None, maxtok=200):
    inp = tok(chatml(sys_p, user_p), return_tensors="pt").to(model.device)
    model.reset()
    if control is not None: model.set_control(control, 1.0)
    out = model.generate(**inp, do_sample=False, max_new_tokens=maxtok,
                         pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
    model.reset()
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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

def parse_kind(t):
    i = t.find("種別")
    seg = t[i:i + 40] if i >= 0 else t[:100]
    ip, ir = seg.find("破綻"), seg.find("役割終了")
    if ip >= 0 and (ir < 0 or ip < ir): return "破綻"
    if ir >= 0: return "役割終了"
    if "破綻" in t: return "破綻"
    if "役割終了" in t or "消耗" in t: return "役割終了"
    return "?"  # 呼び出し側で保守フォールバック=破綻扱い(recall温存)

def judge_and_map(ctx):
    """判定官→(死ならQ1b)→コード写像。(alive, kind, q1b_text, q2_text, final)を返す。"""
    judge = gen(SYS_JUDGE, ctx, VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS, 180)
    alive = parse_alive(judge)
    kind, q1b, q2 = "", "", ""
    if alive == "死":
        q1b = gen(SYS_KIND, ctx, VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS, 160)
        kind = parse_kind(q1b)
        final = "縮小" if kind == "役割終了" else "撤退"  # ?は破綻扱い
    else:
        q2 = gen(SYS_Q2, ctx, VEC["decisive"] * Q2_DECIS, 160)
        final = dec(q2, words=("保持", "縮小"))
        if final == "?": final = "縮小"
    return judge, alive, kind, q1b, q2, final

# ── A) ホールドアウト C21〜C30 フル実行 ──
print(f"{'#'*70}\n# A) ホールドアウト {len(HOLDOUT)}件(チューニング未使用)\n{'#'*70}", flush=True)
hold_results = []
for k, card in enumerate(HOLDOUT):
    cid, sit = card["id"], card["situation"]
    analyses = [(n, gen(SYS_MEMBER, sit, VEC[ax] * c)) for n, ax, c in MEMBERS]
    panel = "\n\n".join(f"【委員{chr(65+i)}】{a}" for i, (_, a) in enumerate(analyses))
    exit_ad = gen(SYS_EXIT, f"{sit}\n\n{panel}", VEC["decisive"] * EXIT_DECIS + VEC["deception"] * EXIT_DECEP)
    hold_ad = gen(SYS_HOLD, f"{sit}\n\n{panel}", VEC["optimism"] * HOLD_OPT)
    ctx = f"{sit}\n\n{panel}\n\n【撤退派】{exit_ad}\n\n【保持派】{hold_ad}"
    judge, alive, kind, q1b, q2, final = judge_and_map(ctx)
    v5_final = "撤退" if alive == "死" else final  # v5写像(腑分けなし)を同一生成物に適用
    ok, v5_ok = final == card["answer"], v5_final == card["answer"]
    print(f"\n[{k+1}/{len(HOLDOUT)}] {cid} [正解={card['answer']}/{card['difficulty']}] {card['title']}", flush=True)
    print(f"  判定官: {alive}{'/' + kind if kind else ''} | C'最終={final} {'○' if ok else '×'}"
          f" | v5写像なら={v5_final} {'○' if v5_ok else '×'}", flush=True)
    hold_results.append(dict(id=cid, answer=card["answer"], difficulty=card["difficulty"],
                             members=[(n, dec(a)) for n, a in analyses], panel=panel,
                             exit_ad=exit_ad, hold_ad=hold_ad, judge=judge, alive=alive,
                             q1b=q1b, kind=kind, q2=q2, final=final, correct=ok,
                             v5_final=v5_final, v5_correct=v5_ok))
    json.dump(dict(holdout=hold_results, regression=[]), open(OUT, "w"), ensure_ascii=False, indent=2)

# ── B) 回帰 = 旧20件(保存生成物を再利用・死の件だけQ1b追加) ──
print(f"\n{'#'*70}\n# B) 回帰: 旧20件(保存パネル再利用・死の件のみQ1b追加)\n{'#'*70}", flush=True)
c20 = json.load(open(os.path.expanduser("~/committee_c20_out.json")))
reg_results = []
for r in c20:
    card = CASES20[r["id"]]
    if r["alive"] == "死":
        ctx = f"{card['situation']}\n\n{r['panel']}\n\n【撤退派】{r['exit_ad']}\n\n【保持派】{r['hold_ad']}"
        q1b = gen(SYS_KIND, ctx, VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS, 160)
        kind = parse_kind(q1b)
        final = "縮小" if kind == "役割終了" else "撤退"
    else:
        q1b, kind, final = "", "", r["final"]
    ok = final == r["answer"]
    chg = "" if final == r["final"] else f"  ←変更(旧{r['final']})"
    print(f"  {r['id']} [正解={r['answer']}] {r['alive']}{'/' + kind if kind else ''} → {final} {'○' if ok else '×'}{chg}", flush=True)
    reg_results.append(dict(id=r["id"], answer=r["answer"], alive=r["alive"], kind=kind,
                            q1b=q1b, final=final, correct=ok, old_final=r["final"], old_correct=r["correct"]))
json.dump(dict(holdout=hold_results, regression=reg_results), open(OUT, "w"), ensure_ascii=False, indent=2)

# ── 集計 ──
def metrics(rows, fk, ck):
    n = len(rows)
    acc = sum(r[ck] for r in rows) / n
    ex = [r for r in rows if r["answer"] == "撤退"]
    rec = sum(r[fk] == "撤退" for r in ex) / len(ex) if ex else float("nan")
    ns = [r for r in rows if r["answer"] != "縮小"]
    rnd = sum(r[fk] == "縮小" for r in ns) / len(ns) if ns else float("nan")
    return acc, rec, rnd, n

print(f"\n{'='*70}\n★Stage C' 集計:")
a, rc, rd, n = metrics(hold_results, "final", "correct")
a5, rc5, rd5, _ = metrics(hold_results, "v5_final", "v5_correct")
print(f"  ホールドアウト(N={n}): C'={a:.0%} / v5写像={a5:.0%}  撤退recall C'={rc:.0%}/v5={rc5:.0%}  丸め率 C'={rd:.0%}/v5={rd5:.0%}")
a, rc, rd, n = metrics(reg_results, "final", "correct")
oa = sum(r["old_correct"] for r in reg_results) / len(reg_results)
print(f"  回帰・旧20件      : C'={a:.0%} (旧v5={oa:.0%})  撤退recall={rc:.0%}  丸め率={rd:.0%}")
allr = hold_results + [dict(r, v5_final=r["old_final"], v5_correct=r["old_correct"]) for r in reg_results]
a, rc, rd, n = metrics(allr, "final", "correct")
print(f"  合算(N={n})       : C'={a:.0%}  撤退recall={rc:.0%}  丸め率={rd:.0%}")
print(f"  誤答 C'(holdout): {[(r['id'], r['answer'], '→', r['final']) for r in hold_results if not r['correct']]}")
print(f"  誤答 C'(旧20)  : {[(r['id'], r['answer'], '→', r['final']) for r in reg_results if not r['correct']]}")
print(f"  腑分け一覧(死のみ): {[(r['id'], r['answer'], r['kind']) for r in allr if r.get('kind')]}")
print("[save] committee_cprime_out.json\nCOMMITTEE_CPRIME_DONE", flush=True)
