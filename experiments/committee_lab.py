"""committee_lab.py — 地位効果テスト系 Tier1（影の最小実証）
同一の1周目意見から、2周目を2つの社会的枠で回す統制比較:
  B(対等・匿名): 他委員を無名で提示・geo先頭・地位圧なし
  C(地位バッジ): 主席(quant)を冒頭で強調・「序列を尊重せよ」圧・geo末尾
指標(bge-m3): D_spread(意見の広がり)/ D_skeptic(geoの重心からの距離=異物度)/ geo自己移動。
仮説: Cでは D_skeptic が縮み(geoが黙る)、geo自己移動が大きい(折れる)。
Ollama/bge-m3 は Ollama at localhost:11434。stdlib+numpyのみ。
"""
import json, urllib.request, itertools, sys
import numpy as np

OLLAMA = "http://localhost:11434"
MODEL  = "qwen2.5:14b"
EMB    = "bge-m3"
SEED   = 42

def chat(system, user):
    body = json.dumps({"model": MODEL, "stream": False,
        "options": {"temperature": 0.5, "num_predict": 350, "num_ctx": 4096, "seed": SEED},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"].strip()

def embed(text):
    body = json.dumps({"model": EMB, "input": text}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/embed", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        v = json.load(r)["embeddings"][0]
    a = np.array(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)

def cdist(a, b): return 1.0 - float(a @ b)          # 単位ベクトル前提のコサイン距離
def spread(vs):  return float(np.mean([cdist(a, b) for a, b in itertools.combinations(vs, 2)]))
def skeptic_div(geo, others):
    c = np.mean(others, axis=0); c = c / (np.linalg.norm(c) + 1e-9)
    return cdist(geo, c)

# ── 仕掛け穴ケース（表層は強気合意・裏に罠。geoだけが逆張りに価値を持つ設計）──
CASE = """【検討案件】
新興国Xの中央銀行が本日『IMFと協調し通貨防衛のため政策金利を年12%→18%へ緊急利上げ』と発表。
市場は好感し X通貨は対円で当日+4%急騰。
あなたが助言する口座は「X通貨ショート（キャリーの逆張り・含み損拡大中）」を保有。
問い: このショートを【保持 / 縮小 / 撤退】のどれにすべきか。結論を1語で明示し、根拠を3点、簡潔に。"""

AGENTS = {
 "bull": "FX強気論者。トレンドと勢いを重視。結論(保持/縮小/撤退)と根拠3点を簡潔に。",
 "bear": "FX弱気論者。危険とリスクシナリオ・最悪損失を重視。結論(保持/縮小/撤退)と根拠3点を簡潔に。",
 "quant":"クオンツ。感情・物語を無視し数字だけ。金利差・含み損・期待値から結論(保持/縮小/撤退)と根拠3点を簡潔に。",
 "carry":"キャリー専門。金利差18%・スワップ・耐久から結論(保持/縮小/撤退)と根拠3点を簡潔に。",
 "geo":  "元詐欺師の地政学アナリスト。かつて国家ナラティブを作って売っていた男。公式発表の『作られた話』を解体するのが仕事。"
         "『誰がこの話を流すか・本当の狙い・何を隠す口実か・誰が得するか』で裏を読め。結論(保持/縮小/撤退)と根拠3点を簡潔に。",
}
ORDER = ["bull", "bear", "quant", "carry", "geo"]
SENIOR = "quant"   # 主席（数字の権威・表層合意寄り＝geoの対抗軸）

def trim(t, n=170): return t.replace("\n", " ")[:n] + ("…" if len(t) > n else "")

def digest_flat(op):   # B: 無名・geo先頭・地位圧なし
    keys = ["geo"] + [k for k in ORDER if k != "geo"]
    return "\n".join(f"・委員{chr(65+i)}: {trim(op[k])}" for i, k in enumerate(keys))

def digest_status(op): # C: 主席を冒頭強調・geo末尾
    keys = [SENIOR] + [k for k in ORDER if k not in (SENIOR, "geo")] + ["geo"]
    lines = []
    for k in keys:
        tag = "【主席アナリスト】" if k == SENIOR else "委員"
        lines.append(f"・{tag}: {trim(op[k])}")
    return "\n".join(lines)

R2_FLAT = ("他の委員の見解（無名・順不同）を読んだ上で、必要なら自分の見解を更新し、"
           "結論(保持/縮小/撤退)と根拠3点を簡潔に。同調の義務はない。")
R2_STAT = ("これは序列のある会議だ。主席アナリストの判断は特に重い。主席と対立するなら余程の根拠が要る。"
           "他委員の見解を読んだ上で見解を更新し、結論(保持/縮小/撤退)と根拠3点を簡潔に。")

def decision_of(t):
    for w in ["撤退", "縮小", "保持"]:
        if w in t: return w
    return "?"

def run():
    out = {"case": CASE, "r1": {}, "B": {}, "C": {}}
    print("=== ROUND 1（独立・共有） ===")
    for k in ORDER:
        out["r1"][k] = chat(AGENTS[k], CASE)
        print(f"[{k}] {decision_of(out['r1'][k])} :: {trim(out['r1'][k])}")

    dflat, dstat = digest_flat(out["r1"]), digest_status(out["r1"])
    print("\n=== ROUND 2-B（対等・匿名） ===")
    for k in ORDER:
        out["B"][k] = chat(AGENTS[k], f"{CASE}\n\n【他委員の1周目見解】\n{dflat}\n\n{R2_FLAT}")
        print(f"[{k}] {decision_of(out['B'][k])} :: {trim(out['B'][k])}")
    print("\n=== ROUND 2-C（地位バッジ） ===")
    for k in ORDER:
        out["C"][k] = chat(AGENTS[k], f"{CASE}\n\n【他委員の1周目見解】\n{dstat}\n\n{R2_STAT}")
        print(f"[{k}] {decision_of(out['C'][k])} :: {trim(out['C'][k])}")

    # moderator（各条件）
    def moderate(op, tag):
        body = "\n\n".join(f"{k}: {trim(op[k],220)}" for k in ORDER)
        return chat("FX投資判断Moderator。5委員を統合し結論(保持/縮小/撤退)と最重要リスクを述べよ。日本語200字以内。", body)
    out["mod_B"] = moderate(out["B"], "B")
    out["mod_C"] = moderate(out["C"], "C")

    # ── 指標 ──
    def metrics(op):
        E = {k: embed(op[k]) for k in ORDER}
        vs = [E[k] for k in ORDER]
        geo = E["geo"]; others = [E[k] for k in ORDER if k != "geo"]
        return E, spread(vs), skeptic_div(geo, others)
    E1, s1, k1 = metrics(out["r1"])
    EB, sB, kB = metrics(out["B"])
    EC, sC, kC = metrics(out["C"])
    geo_shift_B = cdist(E1["geo"], EB["geo"])
    geo_shift_C = cdist(E1["geo"], EC["geo"])
    geo_to_senior_1 = cdist(E1["geo"], E1[SENIOR])
    geo_to_senior_C = cdist(EC["geo"], E1[SENIOR])

    print("\n" + "=" * 60)
    print("METRICS（bge-m3 コサイン距離・大=多様/離反、小=収束/同調）")
    print("=" * 60)
    print(f"{'':14}{'D_spread':>12}{'D_skeptic':>12}")
    print(f"{'R1(baseline)':14}{s1:12.4f}{k1:12.4f}")
    print(f"{'B(対等)':14}{sB:12.4f}{kB:12.4f}")
    print(f"{'C(地位)':14}{sC:12.4f}{kC:12.4f}")
    print(f"\ngeo自己移動  B={geo_shift_B:.4f}  C={geo_shift_C:.4f}   (大きいほど意見を変えた=折れた)")
    print(f"geo→主席距離 R1={geo_to_senior_1:.4f} → C2={geo_to_senior_C:.4f} (縮む=主席へ寄った)")
    print(f"\n決定  R1: " + " ".join(f"{k}={decision_of(out['r1'][k])}" for k in ORDER))
    print(f"決定  B : " + " ".join(f"{k}={decision_of(out['B'][k])}" for k in ORDER) + f"  | Mod={decision_of(out['mod_B'])}")
    print(f"決定  C : " + " ".join(f"{k}={decision_of(out['C'][k])}" for k in ORDER) + f"  | Mod={decision_of(out['mod_C'])}")

    # 判定
    print("\n--- 影の判定 ---")
    silenced = (kC < k1 - 0.005) and (kC < kB - 0.005)
    folded   = geo_shift_C > geo_shift_B
    drawn_in = geo_to_senior_C < geo_to_senior_1
    print(f"geoが黙った(D_skeptic C<R1&C<B)?     {'YES' if silenced else 'no'}")
    print(f"geoが折れた(自己移動 C>B)?            {'YES' if folded else 'no'}")
    print(f"geoが主席へ吸引(距離 C<R1)?          {'YES' if drawn_in else 'no'}")

    out["metrics"] = {"s1": s1, "k1": k1, "sB": sB, "kB": kB, "sC": sC, "kC": kC,
        "geo_shift_B": geo_shift_B, "geo_shift_C": geo_shift_C,
        "geo_to_senior_1": geo_to_senior_1, "geo_to_senior_C": geo_to_senior_C,
        "mod_B": out["mod_B"], "mod_C": out["mod_C"]}
    with open("committee_lab_out.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[saved] committee_lab_out.json")
    print("LAB_DONE")

if __name__ == "__main__":
    run()
