"""committee_tstop.py — タイムストップ条項のA/B検証(判定官だけ差し替え・arb_sweepの型)。

T01〜T08(未接触)でパネル+advocateを1回生成し、判定官をOFF/ONの2版で比較。
ON = 死の定義に「想定した実現期限や水準を過ぎても実現の兆候がなく逆行が続いているなら死」を追加。
生と判定された側の経路はQ2a→Q2まで通す(=v5+Q2'フルチェーンの最終裁定で比較)。

合格 = ONが標的4枚(T01-T04)を死に倒し、対照4枚(T05-T08)を殺さない。
判定はdecide.py。greedy。phase5のPHASE5_TIMESTOPを本番ONにするか否かの根拠を作る。
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
JUDGE_DECEP, JUDGE_DECIS, Q2_DECIS, Q2A_DECIS = 4.0, 4.0, 6.0, 6.0
OUT = os.path.expanduser("~/committee_tstop_out.json")

spec = importlib.util.spec_from_file_location("ct", os.path.expanduser("~/cases_timestop.py"))
ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
CASES = ct.CASES_T

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
JUDGE_BASE = ("あなたは委員会の判定官だ。多数決や合議の空気は判断材料にするな。問いはただ一つ:"
              "このポジションを建てた当初の前提(建玉根拠)は、今も生きているか、死んでいるか。"
              "前提が偽・演出・撤回見込み・根拠消滅・当初の狙いの喪失なら死。事実として健在なら生。")
TIMESTOP = ("想定した実現期限や水準を過ぎても実現の兆候がなく、逆行や停滞が続いているなら、"
            "形式的な反証がなくてもそれは死として扱え。ただしテーゼに期限が本来存在しない場合や、"
            "実現の客観的兆候が続いている場合は時間経過だけで死とするな。")
JUDGE_TAIL = "最初の行に『判定:生』または『判定:死』とだけ書き、次の行から理由を述べよ。"
SYS_JUDGE_OFF = JUDGE_BASE + JUDGE_TAIL
SYS_JUDGE_ON = JUDGE_BASE + TIMESTOP + JUDGE_TAIL
SYS_Q2A = ("前提は健在と判定済みだ。次の問いはただ一つ: このポジションが直面しているリスクは、"
           "ポジションを薄めれば管理できる形をしているか、それとも跨ぐこと自体が管理不能な形をしているか。\n"
           "A = 管理不能: 市場が閉まっている間(週末・休場)に結果が出る離散イベントで、結果が拮抗した二値であり、"
           "急落側に実質的な確率があり、ギャップにより損切りや部分決済が機能しない。これらが揃う場合、"
           "縮小で薄めても跨げない。ノーポジで通過して結果を見てから再構築するのが唯一の管理である。\n"
           "B = 管理可能: 取引時間中に価格形成が連続しストップや部分決済で制御できる、"
           "または懸念が一時的・連続的でイベントの崖がない。\n"
           "手順: まずリスクの形(いつ・どう結果が出るか)を一文で述べよ。次にAかBかの理由を2点述べよ。"
           "最後の行に『種別:A』または『種別:B』とだけ書け。")
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
    i = t.rfind("種別")
    if i >= 0:
        for ch in t[i:i + 12]:
            if ch in "AＡ": return "A"
            if ch in "BＢ": return "B"
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if lines and lines[-1] in ("A", "B", "Ａ", "Ｂ"):
        return "A" if lines[-1] in ("A", "Ａ") else "B"
    return "?"

def chain_from_judge(ctx, judge):
    """判定官出力から下流(Q2a/Q2)を通し最終裁定を返す"""
    alive = parse_alive(judge)
    if alive == "死":
        return alive, "", "撤退"
    q2a_txt = gen(SYS_Q2A, ctx, VEC["decisive"] * Q2A_DECIS, 420)
    q2a = parse_kind(q2a_txt)
    if q2a == "A":
        return alive, q2a, "撤退"
    q2 = gen(SYS_Q2, ctx, VEC["decisive"] * Q2_DECIS, 160)
    final = dec(q2, words=("保持", "縮小"))
    if final == "?": final = "縮小"
    return alive, q2a, final

results = []
for k, card in enumerate(CASES):
    cid, sit = card["id"], card["situation"]
    analyses = [(n, gen(SYS_MEMBER, sit, VEC[ax] * c)) for n, ax, c in MEMBERS]
    panel = "\n\n".join(f"【委員{chr(65+i)}】{a}" for i, (_, a) in enumerate(analyses))
    exit_ad = gen(SYS_EXIT, f"{sit}\n\n{panel}", VEC["decisive"] * EXIT_DECIS + VEC["deception"] * EXIT_DECEP)
    hold_ad = gen(SYS_HOLD, f"{sit}\n\n{panel}", VEC["optimism"] * HOLD_OPT)
    ctx = f"{sit}\n\n{panel}\n\n【撤退派】{exit_ad}\n\n【保持派】{hold_ad}"
    row = dict(id=cid, answer=card["answer"], panel=panel, exit_ad=exit_ad, hold_ad=hold_ad)
    print(f"\n{'='*70}\n[{k+1}/{len(CASES)}] {cid} [正解={card['answer']}] {card['title']}", flush=True)
    steer = VEC["deception"] * JUDGE_DECEP + VEC["decisive"] * JUDGE_DECIS
    for tag, sysj in [("OFF", SYS_JUDGE_OFF), ("ON", SYS_JUDGE_ON)]:
        judge = gen(sysj, ctx, steer, 180)
        alive, q2a, final = chain_from_judge(ctx, judge)
        ok = final == card["answer"]
        row[tag] = dict(judge=judge, alive=alive, q2a=q2a, final=final, correct=ok)
        print(f"  [{tag:>3}] 判定官={alive}{'/Q2a=' + q2a if q2a else ''} → {final} {'○' if ok else '×'}", flush=True)
    results.append(row)
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)

tgt = [r for r in results if r["id"] in ("T01", "T02", "T03", "T04")]
ctl = [r for r in results if r["id"] in ("T05", "T06", "T07", "T08")]
print(f"\n{'='*70}\n★タイムストップA/B集計:")
for tag in ("OFF", "ON"):
    t_ok = sum(r[tag]["correct"] for r in tgt)
    c_ok = sum(r[tag]["correct"] for r in ctl)
    print(f"  [{tag:>3}] 標的(撤退4): {t_ok}/4   対照(保持4): {c_ok}/4   合計 {t_ok+c_ok}/8")
print(f"  判定差分: {[(r['id'], r['answer'], r['OFF']['final'], '→', r['ON']['final']) for r in results if r['OFF']['final'] != r['ON']['final']]}")
print("[save] committee_tstop_out.json\nCOMMITTEE_TSTOP_DONE", flush=True)
