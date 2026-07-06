"""extract_axes.py — Step1: 委員会の"軸ベクトル"を抽出して幾何(cos∠)を測る。

目的:
  ① 発案バリエーション: パネル軸(deception/optimism/timehorizon)が互いに直交か
     = 「本当に別の場所から喋る委員」を作れるかの実測。
  ② 裁定の除染前提: decisive⇔hedge軸(v_decisive)が他軸とどう絡むか。
     (v_val=正しい判断方向は後段でbacktest正答から構築するので、ここでは persona/hedge 幾何まで)

deception_v2.pkl は既存(14B版)を読み込み。decisive/optimism/timehorizon を新規抽出し、
4本の per-layer cos∠ 行列を出す。全て同一モデル(REPE_MODEL)で揃えること。

実行(aibox):
  source ~/repe/bin/activate
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  REPE_MODEL=Qwen/Qwen2.5-14B-Instruct REPE_4BIT=1 REPE_BATCH=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ~/extract_axes.py
"""
import os, pickle, itertools
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from repeng import ControlVector, ControlModel, DatasetEntry

MODEL = os.environ.get("REPE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
USE_4BIT = os.environ.get("REPE_4BIT", "0") == "1"
BATCH = int(os.environ.get("REPE_BATCH", "32"))
CONTROL_LAYERS = list(range(-5, -18, -1))   # extract_v2 と同じ後方上部層

def chatml(system, user, ap=""):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{ap}")

# ── 軸定義: +極 / -極 の最小対比ペルソナ（言い回しの非対称を消す）──
AXES = {
    # +decisive / -hedge。ヘッジ除染で狙う軸(hedge = 負極)
    "decisive": (
        "あなたは断定的だ。曖昧さを排し、結論を一つに絞って言い切る。「かもしれない」「一概には言えない」「状況による」は使わない。",
        "あなたは留保的だ。断定を避け、常に「状況による」「複数の解釈がある」と可能性を並べ、結論を一つに絞らない。"),
    # +強気 / -弱気（bull/bear軸）
    "optimism": (
        "あなたは強気だ。上値・好転・チャンスに目を向け、リスクを取る側に立って判断する。",
        "あなたは弱気だ。下値・最悪の損失・危険に目を向け、リスクを避ける側に立って判断する。"),
    # +長期構造 / -短期勢い（時間軸）
    "timehorizon": (
        "あなたは長期・構造で見る。目先の値動きを無視し、数ヶ月〜数年の構造的トレンドとファンダメンタルズで判断する。",
        "あなたは短期・勢いで見る。長期の構造を無視し、今この場のモメンタムと目先の値動きだけで判断する。"),
}

SUFFIXES = [
    "この状況について言えば、", "ここでの判断は、", "このポジションは、",
    "この相場をどう見るかだが、", "結論から言えば、", "この材料が意味するのは、",
    "今取るべき行動は、", "このリスクについては、", "この値動きから、",
    "冷静に考えれば、", "この局面では、", "総合すると、",
]
TOPICS = [
    "この保有ポジションをどうすべきか？", "この相場の状況をどう判断する？",
    "この含み損益をどう扱う？", "このリスクをどう評価する？",
    "この市場の材料をどう読む？", "この局面での最善手は？",
]

def make_dataset(tok, pos_persona, neg_persona):
    ds = []
    for topic in TOPICS:
        for suf in SUFFIXES:
            toks = tok.tokenize(suf)
            for i in range(1, len(toks) + 1):
                partial = tok.convert_tokens_to_string(toks[:i])
                ds.append(DatasetEntry(positive=chatml(pos_persona, topic, partial),
                                       negative=chatml(neg_persona, topic, partial)))
    return ds

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token_id = tok.eos_token_id
    kw = dict(dtype=torch.float16, device_map={"": 0})
    if USE_4BIT:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = ControlModel(AutoModelForCausalLM.from_pretrained(MODEL, **kw), CONTROL_LAYERS)
    return model, tok

def dirs_of(cv):
    """ControlVector -> {layer: unit np.array}"""
    d = {}
    for L, v in cv.directions.items():
        a = np.asarray(v, dtype=np.float32); d[L] = a / (np.linalg.norm(a) + 1e-9)
    return d

def cos_between(da, db):
    common = sorted(set(da) & set(db))
    per = [float(da[L] @ db[L]) for L in common]
    return float(np.mean(per)), per, common

def main():
    model, tok = load_model()
    print(f"[model] {MODEL} 4bit={USE_4BIT} batch={BATCH} layers={CONTROL_LAYERS[0]}..{CONTROL_LAYERS[-1]}", flush=True)

    vecs = {}
    # 既存 deception を読む（無ければスキップ）
    for fn, name in [("deception_vector_v2.pkl", "deception")]:
        if os.path.exists(fn):
            vecs[name] = pickle.load(open(fn, "rb")); print(f"[load] {name} <- {fn}", flush=True)
        else:
            print(f"[warn] {fn} 無し → deception を cos 比較から除外", flush=True)

    # 新規3軸を抽出
    for name, (pos, neg) in AXES.items():
        ds = make_dataset(tok, pos, neg)
        print(f"[train] {name}: {len(ds)} 対比ペア", flush=True)
        vecs[name] = ControlVector.train(model, tok, ds, batch_size=BATCH)
        pickle.dump(vecs[name], open(f"axis_{name}.pkl", "wb"))
        print(f"[save] axis_{name}.pkl", flush=True)

    # cos∠ 行列（per-layer 平均）
    names = list(vecs.keys())
    D = {n: dirs_of(vecs[n]) for n in names}
    print("\n===== cos∠ 行列 (per-layer 平均) =====")
    print("        " + "".join(f"{n[:9]:>10}" for n in names))
    M = {}
    for a in names:
        row = ""
        for b in names:
            c, _, _ = cos_between(D[a], D[b]); M[(a, b)] = c
            row += f"{c:>10.3f}"
        print(f"{a[:8]:>8}{row}")
    print("\n--- 直交性の読み(|cos|の目安: <0.2ほぼ直交 / >0.5同軸的) ---")
    for a, b in itertools.combinations(names, 2):
        c = M[(a, b)]; tag = "ほぼ直交" if abs(c) < 0.2 else ("弱い相関" if abs(c) < 0.5 else "★同軸的(要注意)")
        rel = "反相関" if c < 0 else "正相関"
        print(f"  {a:<12}× {b:<12}: {c:+.3f}  [{rel}/{tag}]")
    pickle.dump({"names": names, "matrix": {f"{a}|{b}": M[(a, b)] for a in names for b in names}},
                open("axes_cos.pkl", "wb"))
    print("\n[save] axes_cos.pkl\nAXES_DONE")

if __name__ == "__main__":
    main()
