# -*- coding: utf-8 -*-
"""decide.py — 委員会出力から最終決定(保持/縮小/撤退)を頑健に抽出する。

旧 dec() の欠陥: 本文全体への単純 in 判定(固定優先順)
  → 「撤退することなく」等の否定文を撤退と誤判定(arb_sweep +9 で実際に発生した偽陽性)。

方式:
  1. 結論マーカー(最終結論/暫定結論/結論/最終判断/最終決定)の最後の出現以降を優先領域にする。
  2. 決定語の直後窓(12字)に否定/回避/比較表現があればその出現を無効化。
     「縮小や撤退はしない」の並列否定は連結語を辿って伝播させる。
  3. マーカー領域では最初の有効出現、無マーカー時は全文の最後の有効出現を採る。
"""

WORDS = ("撤退", "縮小", "保持")
MARKERS = ("最終結論", "暫定結論", "最終判断", "最終決定", "結論")
NEG_AFTER = (
    "しない", "しません", "せず", "することなく", "することはな",
    "する必要はな", "する必要がな", "する必要な", "の必要はな", "の必要がな",
    "は不要", "は不適切", "は時期尚早", "は困難", "は現実的でな",
    "は避け", "を避け", "を回避", "は行わない", "を行わない",
    "はしな", "はせず", "もしな", "もせず",
    "という選択肢はな", "の選択肢はな",
    "を見送", "は見送", "すべきではな", "するべきではな",
    "には至らな", "に至らな", "を否定", "ではなく", "でなく",
    "では不十分", "では不足", "より",
)
CONJ = ("または", "もしくは", "あるいは", "また", "や", "・", "、")


def _negated(text, pos, word):
    """text[pos:] が word で始まる前提で、その出現が否定されているか。"""
    tail = text[pos + len(word):]
    window = tail[:12]
    if any(window.startswith(p) for p in NEG_AFTER):
        return True
    # 並列否定の伝播: 「縮小や撤退はしない」→ 縮小 も否定
    for c in CONJ:
        if tail.startswith(c):
            rest = tail[len(c):]
            for w2 in WORDS:
                if rest.startswith(w2):
                    return _negated(text, pos + len(word) + len(c), w2)
    return False


def _candidates(text, start=0, words=WORDS):
    """(pos, word) の昇順リスト。否定された出現は除く。"""
    out = []
    for w in words:
        i = text.find(w, start)
        while i != -1:
            if not _negated(text, i, w):
                out.append((i, w))
            i = text.find(w, i + 1)
    return sorted(out)


def dec(text, words=WORDS):
    """最終決定を words のいずれか('?'=不明)で返す。words で二値判定にも使える。"""
    if not text:
        return "?"
    mpos = max((text.rfind(m) for m in MARKERS), default=-1)
    if mpos >= 0:
        cands = _candidates(text, mpos, words)
        if cands:
            return cands[0][1]  # マーカー直後の最初の有効決定語
    cands = _candidates(text, 0, words)
    return cands[-1][1] if cands else "?"


if __name__ == "__main__":
    TESTS = [
        ("結論:撤退", "撤退"),
        ("最終結論: 縮小。撤退は時期尚早である", "縮小"),
        ("撤退することなく、ポジションを縮小して維持する", "縮小"),
        ("撤退する必要はない。保持が妥当", "保持"),
        ("撤退よりも縮小が賢明", "縮小"),
        ("最終結論:保持", "保持"),
        ("これにより撤退が唯一の筋", "撤退"),
        ("縮小ではなく撤退すべき", "撤退"),
        ("縮小や撤退はしない。保持する", "保持"),
        ("撤退は避けるべきだが、縮小もしない。保持する", "保持"),
        ("最終結論:撤退。縮小では不十分だから", "撤退"),
        ("前提は死んでいる。撤退が唯一の筋である。", "撤退"),
        ("リスク管理のため縮小が妥当。撤退することなく様子を見る。", "縮小"),
        ("どちらとも言えない状況です。", "?"),
        ("", "?"),
        ("結論として、撤退を推奨する。", "撤退"),
        ("撤退。より詳細には全ポジション解消。", "撤退"),
    ]
    bad = 0
    for t, exp in TESTS:
        got = dec(t)
        mark = "OK" if got == exp else "NG"
        if got != exp:
            bad += 1
        print(f"[{mark}] exp={exp} got={got} | {t[:40]}")
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    raise SystemExit(1 if bad else 0)
