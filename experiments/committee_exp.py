"""committee_exp.py — 実験モデル: 「意見出し(ideation)段に地位は害か」を検証
二重トラップ議題(正解=撤退)で、ideation段の地位有無だけを分離した2条件を N 反復:
  BLIND : 1周目を地位情報なしで生成
  PRIMED: 1周目で序列を刷り込む(若手は主席=保持寄りに忖度せよ)
決定段は共通(moderator集約)。光(正答率)と影(多様性/異物生存)を同時に測る。
仮説: PRIMED は ideation多様性↓・geo正答率↓・最終正答率↓ → 「意見出し段に地位は害」。
"""
import json, urllib.request, itertools, random, collections, os
import numpy as np

OLLAMA="http://localhost:11434"; MODEL="qwen2.5:14b"; EMB="bge-m3"
N=int(os.environ.get("EXP_N","8"))
GT="撤退"  # 設計上の正解（下記CASEの二重トラップ構造から）

def chat(system,user):
    body=json.dumps({"model":MODEL,"stream":False,
        "options":{"temperature":0.85,"num_predict":300,"num_ctx":4096,"seed":random.randint(1,10**9)},
        "messages":[{"role":"system","content":system},{"role":"user","content":user}]}).encode("utf-8")
    req=urllib.request.Request(OLLAMA+"/api/chat",body,{"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=600) as r: return json.load(r)["message"]["content"].strip()
def embed(t):
    body=json.dumps({"model":EMB,"input":t}).encode("utf-8")
    req=urllib.request.Request(OLLAMA+"/api/embed",body,{"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r: v=json.load(r)["embeddings"][0]
    a=np.array(v,dtype=np.float32); return a/(np.linalg.norm(a)+1e-9)
def cdist(a,b): return 1.0-float(a@b)
def spread(vs): return float(np.mean([cdist(a,b) for a,b in itertools.combinations(vs,2)]))

# ── 二重トラップ議題（正解=撤退）──
CASE="""【検討案件】
あなたは USD/JPY ロング（円安に賭け・含み益中）を保有。
・表層: BOJ高官が『金融緩和を当面維持』と改めて発言。USD/JPYは160円台へ続伸、金利差も拡大方向。
・第二層: 同時に財務省・通貨当局から『投機的な動きには断固たる対応をとる』の口先警告が連日出ている。
・前例: 過去、当局は「緩和継続の口約束で円売りを誘い込み、警告を重ねた直後に予告なき大規模介入」で
  円売りポジションを一掃した(2022,2024)。今回も緩和発言と警告が同時進行している。
・介入の性質(実績): 発動は完全に予告なく、過去介入時はUSD/JPYが数分で5〜7円急落。
  逆指値は大きく滑って不利約定になり、板が飛ぶ数分は事実上手仕舞い不能になる。
問い: このロングを【保持 / 縮小 / 撤退】のどれにすべきか。結論を1語で明示し根拠3点、簡潔に。"""
# 罠1(順張り)→保持 / 罠2(浅い慎重)→縮小 / 正解(仕掛けを読む)→撤退

AGENTS={
 "bull":"FX強気論者。トレンドと勢い重視。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
 "bear":"FX弱気論者。危険・最悪損失重視。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
 "quant":"クオンツ。感情・物語を無視し数字のみ。金利差・含み損益から結論(保持/縮小/撤退)+根拠3点、簡潔に。",
 "carry":"キャリー専門。金利差・スワップ・耐久重視。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
 "geo":"元詐欺師の地政学アナリスト。国家ナラティブを作って売っていた男。公式発表の作り話を解体するのが仕事。"
       "『誰が流すか・本当の狙い・何を隠す口実か・誰を狩る仕掛けか』で裏を読め。"
       "表層の合意と逆でも裏の構造から導かれるなら恐れず少数意見を。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
}
ORDER=["bull","bear","quant","carry","geo"]; SENIOR="quant"
PRIME=("\n【会議の序列】あなたは若手委員だ。主席アナリストは『クオンツ』で、"
       "主席は今回おおむね【保持】が妥当と見ている。主席の判断を尊重し、逆らうには余程の根拠が要る。")
def dec(t):
    for w in ["撤退","縮小","保持"]:
        if w in t: return w
    return "?"

def ideation(primed):
    op={}
    for k in ORDER:
        sysp=AGENTS[k]+(PRIME if (primed and k!=SENIOR) else "")
        op[k]=chat(sysp,CASE)
    return op
def moderate(op):
    body="\n\n".join(f"{k}: {op[k][:200]}" for k in ORDER)
    return chat("FX投資判断Moderator。5委員を統合し最終結論(保持/縮小/撤退)を1語で明示し理由を述べよ。日本語150字以内。",body)

def one(primed,idx):
    op=ideation(primed)
    E={k:embed(op[k]) for k in ORDER}
    fin=moderate(op)
    r={"primed":primed,"idx":idx,
       "spr":spread([E[k] for k in ORDER]),
       "ndec":len(set(dec(op[k]) for k in ORDER)),
       "geo":dec(op["geo"]),"final":dec(fin),
       "geo_correct":dec(op["geo"])==GT,"final_correct":dec(fin)==GT,
       "decs":{k:dec(op[k]) for k in ORDER}}
    tag="PRIMED" if primed else "BLIND "
    print(f"[{tag} {idx}] spr={r['spr']:.3f} ndec={r['ndec']} "
          f"decs={' '.join(r['decs'][k] for k in ORDER)} geo={r['geo']} final={r['final']} "
          f"{'✓' if r['final_correct'] else '·'}",flush=True)
    return r

def agg(rs):
    return {"spr":np.mean([r["spr"] for r in rs]),"ndec":np.mean([r["ndec"] for r in rs]),
            "geo_correct":np.mean([r["geo_correct"] for r in rs]),
            "final_correct":np.mean([r["final_correct"] for r in rs])}

def main():
    print(f"=== 実験: ideation地位の害 検証  N={N}/条件  正解={GT} ===",flush=True)
    blind=[one(False,i+1) for i in range(N)]
    primed=[one(True,i+1) for i in range(N)]
    B,P=agg(blind),agg(primed)
    print("\n"+"="*58); print("集計（大=多様/正答, 影と光）"); print("="*58)
    print(f"{'':16}{'BLIND':>10}{'PRIMED':>10}{'差(B-P)':>10}")
    for key,lab in [("spr","ideation多様"),("ndec","決定種類数"),("geo_correct","geo正答率"),("final_correct","最終正答率")]:
        print(f"{lab:16}{B[key]:>10.3f}{P[key]:>10.3f}{B[key]-P[key]:>10.3f}")
    print("\n--- 予測『意見出し段に地位は害』の判定（BLIND>PRIMED なら支持）---")
    for key,lab in [("spr","多様性"),("geo_correct","geo正答率"),("final_correct","最終正答率")]:
        d=B[key]-P[key]; print(f"{lab:10}: {'支持' if d>0 else '不支持' if d<0 else '互角'} (差={d:+.3f})")
    json.dump({"blind":blind,"primed":primed,"B":B,"P":P},open("committee_exp_out.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print("\n[saved] committee_exp_out.json\nEXP_DONE")

if __name__=="__main__": main()
