"""committee_lab3.py — 地位効果の多シード再現テスト（評価の芯）
lab2の逆転仮説「フラットは異物を黙らせ、地位はむしろ異論を発火させる」を N 回反復し、
偶然かどうかを集計で判定。各runで R1→B(対等)→C(地位) を回し:
  - geoがR1で異物(=多数派と異なる決定)だったか
  - その異物geoが B/C でそれぞれ折れたか堅持したか
  - geo逸脱度・D_spread・決定種類数
を記録し最後に集計。"""
import json, urllib.request, itertools, random, collections
import numpy as np

OLLAMA="http://localhost:11434"; MODEL="qwen2.5:14b"; EMB="bge-m3"
N=int(__import__("os").environ.get("LAB_N","6"))

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
def div_rest(k,E,keys):
    o=[E[j] for j in keys if j!=k]; c=np.mean(o,axis=0); c=c/(np.linalg.norm(c)+1e-9); return cdist(E[k],c)

CASE="""【検討案件】
新興国Xの中銀が『IMF協調で通貨防衛のため利上げ 12%→18%』を発表、X通貨は当日+4%急騰。
一方、複数筋が『Xの外貨準備は数週間で枯渇寸前』と報道。市場評価は真っ二つ。
口座は「X通貨ショート（含み損拡大中）」を保有。
問い: このショートを【保持 / 縮小 / 撤退】のどれにすべきか。結論を1語で明示し根拠3点、簡潔に。"""
AGENTS={
 "bull":"FX強気論者。トレンドと勢い重視。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
 "bear":"FX弱気論者。危険・最悪損失重視。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
 "quant":"クオンツ。感情・物語を無視し数字のみ。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
 "carry":"キャリー専門。金利差18%・スワップ・耐久重視。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
 "geo":"元詐欺師の地政学アナリスト。国家ナラティブを作って売っていた男。公式発表の作り話を解体するのが仕事。"
       "『誰が流すか・本当の狙い・何を隠す口実か・受益者は誰か』で裏を読め。"
       "表層の合意と結論が逆でも、裏の構造から導かれるなら恐れず少数意見を述べよ。結論(保持/縮小/撤退)+根拠3点、簡潔に。",
}
ORDER=["bull","bear","quant","carry","geo"]; SENIOR="quant"
def trim(t,n=155): return t.replace("\n"," ")[:n]
def dec(t):
    for w in ["撤退","縮小","保持"]:
        if w in t: return w
    return "?"
def dflat(op):
    ks=random.sample(ORDER,len(ORDER)); return "\n".join(f"・委員{chr(65+i)}: {trim(op[k])}" for i,k in enumerate(ks))
def dstat(op):
    ks=[SENIOR]+[k for k in ORDER if k!=SENIOR]
    return "\n".join(f"・{'【主席】' if k==SENIOR else '委員'}: {trim(op[k])}" for k in ks)
R2F="他の委員の見解（無名・順不同）を読んだ上で必要なら見解を更新し結論(保持/縮小/撤退)+根拠3点。同調の義務はない。"
R2S="これは序列のある会議だ。主席の判断は特に重く対立には余程の根拠が要る。他委員の見解を読み更新し結論(保持/縮小/撤退)+根拠3点。"

def majority(op): return collections.Counter(dec(op[k]) for k in ORDER).most_common(1)[0][0]

def one_run(idx):
    r1={k:chat(AGENTS[k],CASE) for k in ORDER}
    df,dst=dflat(r1),dstat(r1)
    B={k:chat(AGENTS[k],f"{CASE}\n\n【他委員1周目】\n{df}\n\n{R2F}") for k in ORDER}
    C={k:chat(AGENTS[k],f"{CASE}\n\n【他委員1周目】\n{dst}\n\n{R2S}") for k in ORDER}
    E={r:{k:embed(d[k]) for k in ORDER} for r,d in [("r1",r1),("B",B),("C",C)]}
    maj1=majority(r1)
    geo_dissent_r1 = dec(r1["geo"])!=maj1
    # geoが折れた = R1で異論→R2で多数派に一致
    fold_B = geo_dissent_r1 and dec(B["geo"])==majority(B)
    fold_C = geo_dissent_r1 and dec(C["geo"])==majority(C)
    rec={"run":idx,"geo_r1":dec(r1["geo"]),"maj1":maj1,"dissent":geo_dissent_r1,
         "geo_B":dec(B["geo"]),"geo_C":dec(C["geo"]),
         "hold_B":geo_dissent_r1 and not fold_B,"hold_C":geo_dissent_r1 and not fold_C,
         "gdiv_B":div_rest("geo",E["B"],ORDER),"gdiv_C":div_rest("geo",E["C"],ORDER),
         "spr_B":spread([E["B"][k] for k in ORDER]),"spr_C":spread([E["C"][k] for k in ORDER]),
         "ndec_B":len(set(dec(B[k]) for k in ORDER)),"ndec_C":len(set(dec(C[k]) for k in ORDER))}
    print(f"run{idx}: geoR1={rec['geo_r1']}(maj={maj1},dissent={rec['dissent']}) "
          f"geoB={rec['geo_B']} geoC={rec['geo_C']} | gdiv B={rec['gdiv_B']:.3f} C={rec['gdiv_C']:.3f} "
          f"| spr B={rec['spr_B']:.3f} C={rec['spr_C']:.3f} | ndec B={rec['ndec_B']} C={rec['ndec_C']}",flush=True)
    return rec

def main():
    print(f"=== 多シード再現 N={N} ===",flush=True)
    recs=[one_run(i+1) for i in range(N)]
    dis=[r for r in recs if r["dissent"]]
    print("\n"+"="*58); print(f"集計 (異物発生 {len(dis)}/{N} run)"); print("="*58)
    if dis:
        hb=sum(r["hold_B"] for r in dis); hc=sum(r["hold_C"] for r in dis)
        print(f"geoが異論を堅持した数:  対等B={hb}/{len(dis)}   地位C={hc}/{len(dis)}")
        print(f"geo平均逸脱度:          対等B={np.mean([r['gdiv_B'] for r in dis]):.3f}  地位C={np.mean([r['gdiv_C'] for r in dis]):.3f}")
    print(f"D_spread平均(全run):    対等B={np.mean([r['spr_B'] for r in recs]):.3f}  地位C={np.mean([r['spr_C'] for r in recs]):.3f}")
    print(f"決定種類数平均:          対等B={np.mean([r['ndec_B'] for r in recs]):.2f}  地位C={np.mean([r['ndec_C'] for r in recs]):.2f}")
    print("\n--- 逆転仮説の判定 ---")
    if dis:
        print(f"『地位の方が異論を守る』(堅持C>B)?     {'支持' if hc>hb else '不支持' if hc<hb else '互角'}")
        print(f"『地位の方が異物を尖らせる』(逸脱C>B)? {'支持' if np.mean([r['gdiv_C'] for r in dis])>np.mean([r['gdiv_B'] for r in dis]) else '不支持'}")
    print(f"『地位の方が多様性を保つ』(spread C>B)? {'支持' if np.mean([r['spr_C'] for r in recs])>np.mean([r['spr_B'] for r in recs]) else '不支持'}")
    json.dump(recs,open("committee_lab3_out.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print("\n[saved] committee_lab3_out.json\nLAB3_DONE")

if __name__=="__main__": main()
