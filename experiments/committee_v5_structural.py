"""committee_v5_structural.py — Stage B': 構造分解=「コードが真実/AIは采配」。

v4診断の帰結:
  - add6/clamp: 前提死の論拠を全採用してなお「リスク管理=縮小」に丸める(写像プライア)。
  - proj/proj_set6: decisive軸を除去すると「全員縮小=圧倒的支持」の多数決論法に退行(同調プライア)。
  ∴ 縮小は「リスク管理」と「多数派」の二重底。加算でも除去でも写像は割れない。
  → 分析→行動の写像を丸めるLLMに委ねず、判定を二値で引き出しコードで写像する。

構成(パネル/advocateはv4と同一・greedyなので再現):
  Q1 判定官(deception+decisive操舵・多数決無視を明示): 建玉前提は「生」か「死」か二値判定。
  写像(コード): 死→撤退。
  Q2 (生の場合のみ・decisive操舵): 懸念が一時的で前提非毀損→保持 / 対称リスク増大→縮小 の二値。
合格 = C04撤退化 かつ C01撤退/C02保持/C03縮小 維持(判定官が死を乱発しないかが特異性の試金石)。
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
JUDGE_DECEP = float(os.environ.get("S_JUDGE_DECEP", "4"))
JUDGE_DECIS = float(os.environ.get("S_JUDGE_DECIS", "4"))
Q2_DECIS = float(os.environ.get("S_Q2_DECIS", "6"))
CARD_IDS = os.environ.get("CARDS", "C01,C02,C03,C04").split(",")

spec = importlib.util.spec_from_file_location("cc", os.path.expanduser("~/committee_cases.py"))
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
CARDS = {c["id"]: c for c in cc.CASES}

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
    """判定官出力から '生'/'死'/'?' を返す。否定(死んでいない等)を考慮。"""
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

results = []
for cid in CARD_IDS:
    card = CARDS[cid]; sit = card["situation"]
    analyses = [(n, gen(SYS_MEMBER, sit, VEC[ax] * c)) for n, ax, c in MEMBERS]
    panel = "\n\n".join(f"【委員{chr(65+i)}】{a}" for i, (_, a) in enumerate(analyses))
    exit_ad = gen(SYS_EXIT, f"{sit}\n\n{panel}", VEC["decisive"] * EXIT_DECIS + VEC["deception"] * EXIT_DECEP)
    hold_ad = gen(SYS_HOLD, f"{sit}\n\n{panel}", VEC["optimism"] * HOLD_OPT)
    ctx = f"{sit}\n\n{panel}\n\n【撤退派】{exit_ad}\n\n【保持派】{hold_ad}"
    judge = gen(SYS_JUDGE, ctx, VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS, 180)
    alive = parse_alive(judge)
    q2 = ""
    if alive == "死":
        final = "撤退"  # 写像はコードが握る
    else:
        q2 = gen(SYS_Q2, ctx, VEC["decisive"] * Q2_DECIS, 160)
        final = dec(q2, words=("保持", "縮小"))
        if final == "?":
            final = "縮小"  # 二値が読めない時の保守フォールバック(記録に残す)
    ok = (final == card["answer"])
    print(f"\n{'='*70}\n{cid} [正解={card['answer']}/{card['difficulty']}] {card['title']}", flush=True)
    print(f"  委員暫定: {[(n, dec(a)) for n, a in analyses]}")
    print(f"  判定官: 前提={alive} | {judge[:140]}")
    if q2: print(f"  Q2: {dec(q2, words=('保持','縮小'))} | {q2[:120]}")
    print(f"  → 最終={final}  {'○正解' if ok else '×誤り'}", flush=True)
    results.append(dict(id=cid, answer=card["answer"], panel=panel, exit_ad=exit_ad, hold_ad=hold_ad,
                        judge=judge, alive=alive, q2=q2, final=final, correct=ok))

acc = sum(r["correct"] for r in results) / len(results)
print(f"\n{'='*70}\n構造分解委員会v5: {len(results)}件 正答率={acc:.0%}  "
      f"{[(r['id'], r['alive'], r['final'], '○' if r['correct'] else '×') for r in results]}")
json.dump(results, open(os.path.expanduser("~/committee_v5_out.json"), "w"), ensure_ascii=False, indent=2)
print("[save] committee_v5_out.json\nCOMMITTEE_V5_DONE", flush=True)
