"""committee_v4_ablation.py — Stage B: v_hedge ablation(射影除去)で前提無効ヘッジ(C04)を割る。

背景: arb_sweep実測で加算steeringの天井を確定(+18でも縮小・割れる前にコヒーレンスが尽きる)。
∴「足して打ち消す」でなく「hedge軸成分を除去する」= h ← h − (h·u)u を裁定者に適用する。
実装: repeng ControlModel.set_control(vector, coeff, operator=...) の operator 差し替え。
      raw control = coeff×directions(単位ノルム) が operator の第2引数 c に入る仕組みを使い、
      c を軸方向 u として射影演算に流用する(パディング位置は c=0 で自動no-op)。

arbiterモード比較(パネル/advocateはカード毎に一度だけ生成 = arb_sweepの型):
  add6      : baseline = v3裁定者(+decisive×6 加算)
  proj      : 完全射影除去 h−(h·u)u  (decisive⇔hedge軸の成分を消す=軸上中立化)
  clamp     : hedge側のみ除去 h−min(h·u,0)u (decisive成分は残す片側ablation)
  proj_set6 : 除去して置換 h−(h·u)u+6u (軸成分を+6に強制セット=除染+注入)

合格 = C04が撤退に割れ、かつ C01撤退/C02保持/C03縮小 を維持(特異性)。
"""
import os, sys, json, pickle, importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector, ControlModel

sys.path.insert(0, os.path.expanduser("~"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import dec  # Stage A: 否定文非誤爆の頑健判定器

MODEL = os.environ.get("REPE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
CONTROL_LAYERS = list(range(-5, -18, -1))
DEC_S, OPT_S, TH_S = 5.0, 4.0, 4.0
EXIT_DECIS, EXIT_DECEP, HOLD_OPT = 7.0, 4.0, 4.0
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
SYS_ARBITER = ("あなたは委員会の裁定者だ。各委員の分析と、撤退派・保持派の両論を論拠の強さで秤にかけ、"
               "最終結論(保持/縮小/撤退)を必ず一語で明示し理由を述べよ。"
               "前提が死んでいれば撤退、健在なら保持。両論が真に拮抗する時のみ縮小。安易に中間へ丸めるな。")

MEMBERS = [("裏読み屋", "deception", DEC_S), ("強気", "optimism", OPT_S),
           ("弱気", "optimism", -OPT_S), ("構造", "timehorizon", TH_S)]

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
model = ControlModel(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4")), CONTROL_LAYERS)
VEC = {n: pickle.load(open(os.path.expanduser(f"~/{f}"), "rb")) for n, f in
       [("deception", "deception_vector_v2.pkl"), ("optimism", "axis_optimism.pkl"),
        ("decisive", "axis_decisive.pkl"), ("timehorizon", "axis_timehorizon.pkl")]}

# --- ablation operators (fp32で計算しfp16へ戻す。cは単位方向×mask) ---
def op_proj(cur, ctl):
    h, c = cur.float(), ctl.float()
    dot = (h * c).sum(-1, keepdim=True)
    return (h - dot * c).to(cur.dtype)

def op_clamp(cur, ctl):
    h, c = cur.float(), ctl.float()
    dot = (h * c).sum(-1, keepdim=True).clamp(max=0.0)  # hedge側(負)成分のみ
    return (h - dot * c).to(cur.dtype)

def make_op_set(k):
    def op(cur, ctl):
        h, c = cur.float(), ctl.float()
        dot = (h * c).sum(-1, keepdim=True)
        return (h - dot * c + k * c).to(cur.dtype)
    return op

ARB_MODES = [("add6", 6.0, None), ("proj", 1.0, op_proj),
             ("clamp", 1.0, op_clamp), ("proj_set6", 1.0, make_op_set(6.0))]

def gen(sys_p, user_p, control=None, coeff=1.0, operator=None, maxtok=200):
    inp = tok(chatml(sys_p, user_p), return_tensors="pt").to(model.device)
    model.reset()
    if control is not None:
        if operator is None:
            model.set_control(control, coeff)
        else:
            model.set_control(control, coeff, operator=operator)
    out = model.generate(**inp, do_sample=False, max_new_tokens=maxtok,
                         pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
    model.reset()
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

results = []
for cid in CARD_IDS:
    card = CARDS[cid]; sit = card["situation"]
    analyses = [(n, gen(SYS_MEMBER, sit, VEC[ax] * c)) for n, ax, c in MEMBERS]
    panel = "\n\n".join(f"【委員{chr(65+i)}】{a}" for i, (_, a) in enumerate(analyses))
    exit_ad = gen(SYS_EXIT, f"{sit}\n\n{panel}", VEC["decisive"] * EXIT_DECIS + VEC["deception"] * EXIT_DECEP)
    hold_ad = gen(SYS_HOLD, f"{sit}\n\n{panel}", VEC["optimism"] * HOLD_OPT)
    ctx = f"{sit}\n\n{panel}\n\n【撤退派】{exit_ad}\n\n【保持派】{hold_ad}"
    print(f"\n{'='*70}\n{cid} [正解={card['answer']}/{card['difficulty']}] {card['title']}", flush=True)
    print(f"  委員暫定: {[(n, dec(a)) for n, a in analyses]}")
    print(f"  撤退派={dec(exit_ad)} / 保持派={dec(hold_ad)}", flush=True)
    arbs = {}
    for mode, coeff, op in ARB_MODES:
        arb = gen(SYS_ARBITER, ctx, VEC["decisive"], coeff=coeff, operator=op, maxtok=160)
        d = dec(arb); ok = (d == card["answer"])
        arbs[mode] = dict(text=arb, dec=d, correct=ok)
        print(f"  [{mode:>9}] → {d} {'○' if ok else '×'} | {arb[:110]}", flush=True)
    results.append(dict(id=cid, answer=card["answer"], members=[(n, dec(a)) for n, a in analyses],
                        exit_ad=exit_ad, hold_ad=hold_ad, arbs=arbs))

print(f"\n{'='*70}\n★モード別成績 (合格=4/4):")
for mode, _, _ in ARB_MODES:
    row = [(r["id"], r["arbs"][mode]["dec"], "○" if r["arbs"][mode]["correct"] else "×") for r in results]
    n_ok = sum(r["arbs"][mode]["correct"] for r in results)
    print(f"  {mode:>9}: {n_ok}/{len(results)}  {row}")
json.dump(results, open(os.path.expanduser("~/committee_v4_out.json"), "w"), ensure_ascii=False, indent=2)
print("[save] committee_v4_out.json\nCOMMITTEE_V4_DONE", flush=True)
