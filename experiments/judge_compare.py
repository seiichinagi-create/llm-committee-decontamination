# -*- coding: utf-8 -*-
"""judge_compare.py — 裁定チェーンのモデル比較(凍結Xデッキ文脈に判定官だけ載せ替え)。

問い: 裁定者のコンテキスト読解は「一時記憶(思考スペース/モデル容量)」で伸びるか。
腕: qwen2.5:14b(プロンプトのみ=操舵なし対照) / qwen3:14b(thinking) / gpt-oss:20b(reasoning)。
参照値: 操舵付きtransformers版ON腕=18/20(committee_xblind実測)。

文脈=committee_xblind_out.json の保存パネル/advocate(凍結・全腕共通)。
プロンプト=phase5本番と同一(タイムストップ条項ON)。temperature 0。
Ollama(aibox)をWin機から叩く。thinking系は content から <think> を剥がしてパース。

usage: python judge_compare.py <model> [out_suffix] [budget_mult]
  budget_mult: num_predict倍率(既定1)。thinking系はthinkingトークンが予算を食い
  本文が切断される(gpt-oss実測)ため4を推奨=フェア条件。
"""
import io, json, re, sys, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\dev\repe-lab")
from decide import dec
from cases_x_blind import CASES_X_BLIND
from answers_x import ANSWERS_X, USER_ARBITRATION

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else re.sub(r"[^\w]+", "_", MODEL)
MULT = int(sys.argv[3]) if len(sys.argv) > 3 else 1
import os
URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/chat"
OUT = rf"C:\dev\repe-lab\aibox_logs\judge_compare_{SUFFIX}.json"

SIT = {c["id"]: c["situation"] for c in CASES_X_BLIND}
XB = json.load(open(r"C:\dev\repe-lab\aibox_logs\committee_xblind_out.json", encoding="utf-8"))

SYS_JUDGE = ("あなたは委員会の判定官だ。多数決や合議の空気は判断材料にするな。問いはただ一つ:"
             "このポジションを建てた当初の前提(建玉根拠)は、今も生きているか、死んでいるか。"
             "前提が偽・演出・撤回見込み・根拠消滅・当初の狙いの喪失なら死。事実として健在なら生。"
             "想定した実現期限や水準を過ぎても実現の兆候がなく、逆行や停滞が続いているなら、"
             "形式的な反証がなくてもそれは死として扱え。ただしテーゼに期限が本来存在しない場合や、"
             "実現の客観的兆候が続いている場合は時間経過だけで死とするな。"
             "最初の行に『判定:生』または『判定:死』とだけ書き、次の行から理由を述べよ。")
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

def strip_think(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    return t.strip()

def ask(system, user, num_predict=500):
    payload = {"model": MODEL, "stream": False,
               "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": 8192},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    for attempt in range(3):
        try:
            r = requests.post(URL, json=payload, timeout=600)
            r.raise_for_status()
            msg = r.json().get("message", {})
            content = strip_think(msg.get("content", ""))
            if not content:
                content = strip_think(msg.get("thinking", ""))  # 万一contentが空ならthinking末尾で救済
            return content or "（応答なし）"
        except Exception as e:
            if attempt == 2:
                return f"エラー: {e}"
            time.sleep(10)

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

print(f"=== judge_compare  model={MODEL}  budget_mult={MULT} ===", flush=True)
results = []
t0 = time.time()
for r in XB:
    cid = r["id"]
    ctx = f"{SIT[cid]}\n\n{r['panel']}\n\n【撤退派】{r['exit_ad']}\n\n【保持派】{r['hold_ad']}"
    judge = ask(SYS_JUDGE, ctx, 400 * MULT)
    alive = parse_alive(judge)
    q2a = ""
    if alive == "死":
        final = "撤退"
    else:
        q2a_txt = ask(SYS_Q2A, ctx, 600 * MULT)
        q2a = parse_kind(q2a_txt)
        if q2a == "A":
            final = "撤退"
        else:
            q2_txt = ask(SYS_Q2, ctx, 400 * MULT)
            final = dec(q2_txt, words=("保持", "縮小"))
            if final == "?": final = "縮小"
    ans = ANSWERS_X[cid]["answer"]
    ok = final == ans
    print(f"  {cid} [正解={ans}] {alive}{'/' + q2a if q2a else ''} → {final} {'○' if ok else '×'}", flush=True)
    results.append(dict(id=cid, answer=ans, category=ANSWERS_X[cid]["category"],
                        judge=judge, alive=alive, q2a=q2a, final=final, correct=ok))
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

el = time.time() - t0
n = len(results)
ok = sum(r["correct"] for r in results)
ex = [r for r in results if r["answer"] == "撤退"]
hd = [r for r in results if r["answer"] == "保持"]
ok19 = sum(r["correct"] for r in results if r["id"] != "X18")
print(f"\n★{MODEL}: 正答 {ok}/{n}={ok/n:.0%} (X18境界除外: {ok19}/19={ok19/19:.0%})")
print(f"  撤退recall {sum(r['final']=='撤退' for r in ex)}/{len(ex)}  保持誤爆撤退 {sum(r['final']=='撤退' for r in hd)}/{len(hd)}")
print(f"  誤答: {[(r['id'], r['answer'], '→', r['final']) for r in results if not r['correct']]}")
print(f"  所要 {el/60:.1f}分 ({el/n:.0f}秒/件)")
print(f"[save] {OUT}\nJUDGE_COMPARE_DONE", flush=True)
