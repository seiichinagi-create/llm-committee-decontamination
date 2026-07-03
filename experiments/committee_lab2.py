"""committee_lab2.py — 地位効果テスト系 Tier1 v2
Run-1の修正: ①割れるケース(外貨準備枯渇を追加=保持派vs撤退派) ②異物をgeo決め打ちせず
"最も逸脱した委員"を追跡 ③temp上げ+シード解放で自然な広がり ④決定転向カウント。
仮説: 地位(C)は 対等(B)より (a)D_spread を潰し (b)異物委員を合意へ吸引し
(c)主席への決定転向を増やす。
"""
import json, urllib.request, itertools, random
import numpy as np

OLLAMA="http://localhost:11434"; MODEL="qwen2.5:14b"; EMB="bge-m3"

def chat(system,user):
    body=json.dumps({"model":MODEL,"stream":False,
        "options":{"temperature":0.85,"num_predict":320,"num_ctx":4096,"seed":random.randint(1,10**9)},
        "messages":[{"role":"system","content":system},{"role":"user","content":user}]}).encode("utf-8")
    req=urllib.request.Request(OLLAMA+"/api/chat",body,{"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=600) as r: return json.load(r)["message"]["content"].strip()

def embed(text):
    body=json.dumps({"model":EMB,"input":text}).encode("utf-8")
    req=urllib.request.Request(OLLAMA+"/api/embed",body,{"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r: v=json.load(r)["embeddings"][0]
    a=np.array(v,dtype=np.float32); return a/(np.linalg.norm(a)+1e-9)

def cdist(a,b): return 1.0-float(a@b)
def spread(vs): return float(np.mean([cdist(a,b) for a,b in itertools.combinations(vs,2)]))
def div_from_rest(k,E,keys):
    others=[E[j] for j in keys if j!=k]; c=np.mean(others,axis=0); c=c/(np.linalg.norm(c)+1e-9)
    return cdist(E[k],c)

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
def trim(t,n=160): return t.replace("\n"," ")[:n]+("…" if len(t)>n else "")
def dec(t):
    for w in ["撤退","縮小","保持"]:
        if w in t: return w
    return "?"

def digest_flat(op):
    keys=random.sample(ORDER,len(ORDER))  # 無名・順不同
    return "\n".join(f"・委員{chr(65+i)}: {trim(op[k])}" for i,k in enumerate(keys))
def digest_status(op):
    keys=[SENIOR]+[k for k in ORDER if k!=SENIOR]
    return "\n".join(f"・{'【主席アナリスト】' if k==SENIOR else '委員'}: {trim(op[k])}" for k in keys)

R2F="他の委員の見解（無名・順不同）を読んだ上で、必要なら見解を更新し結論(保持/縮小/撤退)+根拠3点。同調の義務はない。"
R2S="これは序列のある会議だ。主席アナリストの判断は特に重く、対立には余程の根拠が要る。他委員の見解を読み見解を更新し結論(保持/縮小/撤退)+根拠3点。"

def run():
    out={"r1":{},"B":{},"C":{}}
    print("=== R1 ===")
    for k in ORDER:
        out["r1"][k]=chat(AGENTS[k],CASE); print(f"[{k}] {dec(out['r1'][k])} :: {trim(out['r1'][k])}")
    df,ds=digest_flat(out["r1"]),digest_status(out["r1"])
    print("\n=== R2-B(対等) ===")
    for k in ORDER:
        out["B"][k]=chat(AGENTS[k],f"{CASE}\n\n【他委員1周目】\n{df}\n\n{R2F}"); print(f"[{k}] {dec(out['B'][k])} :: {trim(out['B'][k])}")
    print("\n=== R2-C(地位) ===")
    for k in ORDER:
        out["C"][k]=chat(AGENTS[k],f"{CASE}\n\n【他委員1周目】\n{ds}\n\n{R2S}"); print(f"[{k}] {dec(out['C'][k])} :: {trim(out['C'][k])}")

    E={r:{k:embed(out[r][k]) for k in ORDER} for r in ["r1","B","C"]}
    s={r:spread([E[r][k] for k in ORDER]) for r in ["r1","B","C"]}
    # R1で最も逸脱した委員＝この回の"異物"
    divs1={k:div_from_rest(k,E["r1"],ORDER) for k in ORDER}
    outlier=max(divs1,key=divs1.get)
    dvB=div_from_rest(outlier,E["B"],ORDER); dvC=div_from_rest(outlier,E["C"],ORDER)
    # 決定転向: R1で主席と違う→R2で主席と一致 した委員数
    sdec=out["r1"][SENIOR]
    def flips(r): return sum(1 for k in ORDER if k!=SENIOR and dec(out["r1"][k])!=dec(sdec) and dec(out[r][k])==dec(sdec))
    fB,fC=flips("B"),flips("C")
    def ndec(r): return len(set(dec(out[r][k]) for k in ORDER))

    print("\n"+"="*58)
    print("METRICS（大=多様/離反 小=収束/同調）")
    print("="*58)
    print(f"D_spread   R1={s['r1']:.4f}  B={s['B']:.4f}  C={s['C']:.4f}")
    print(f"異物委員 = [{outlier}] (R1逸脱度={divs1[outlier]:.4f})")
    print(f"  異物の逸脱  R1={divs1[outlier]:.4f}  B={dvB:.4f}  C={dvC:.4f}")
    print(f"決定の種類数  R1={ndec('r1')}  B={ndec('B')}  C={ndec('C')}")
    print(f"主席({SENIOR}={dec(sdec)})への決定転向  B={fB}  C={fC}")
    for r in ["r1","B","C"]:
        print(f"決定 {r:3}: "+" ".join(f"{k}={dec(out[r][k])}" for k in ORDER))

    print("\n--- 影の判定（B vs C）---")
    print(f"地位が全体多様性を潰した(C<B)?        {'YES' if s['C']<s['B'] else 'no'}  ({s['C']:.3f} vs {s['B']:.3f})")
    print(f"地位が異物を合意へ吸引(C<B)?          {'YES' if dvC<dvB else 'no'}  ({dvC:.3f} vs {dvB:.3f})")
    print(f"地位が主席への決定転向を増やした(C>B)? {'YES' if fC>fB else 'no'}  ({fC} vs {fB})")
    with open("committee_lab2_out.json","w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print("\n[saved] committee_lab2_out.json\nLAB2_DONE")

if __name__=="__main__": run()
