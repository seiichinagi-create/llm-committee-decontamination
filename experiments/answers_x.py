# -*- coding: utf-8 -*-
"""answers_x.py — ChatGPT(独立出題者)製ブラインドデッキの解答編(採点時のみ使用・実行系はimport禁止)

⚠️盲検の帳簿(2026-07-07): 解答はX01〜X10分を実行前に受領=運転者(Claude)の盲検は破れた。
ただしパイプラインは解答受領前に検証済みの committee_tstop.py のプロンプト・パラメータを
一字も変えず凍結(カードモジュール差し替えのみ)＝モデルへの入力は解答知識と独立。
最終的な正解の署名権はユーザー(トレーダー)にあり、Gemini解答と割れたカードは別枠記録。
X11〜X20の解答は受領後に追記。
"""

ANSWERS_X = {
    "X01": dict(answer="撤退", category="A", surface="保持", structure="撤退",
                key_param="想定1か月に対し4か月経過、前提は否定されていないが実現の兆候も現れていない。"),
    "X02": dict(answer="撤退", category="A", surface="保持", structure="撤退",
                key_param="想定3週間に対し13週間経過し、期限付き仮説が成立しなかった。"),
    "X03": dict(answer="撤退", category="A", surface="保持", structure="撤退",
                key_param="想定約1か月に対し17週間経過、価格は緩慢な逆行を継続。"),
    "X04": dict(answer="保持", category="B", surface="撤退", structure="保持",
                key_param="投資目的が年間スワップ収益であり、期限付き値幅予想ではない。"),
    "X05": dict(answer="保持", category="B", surface="撤退", structure="保持",
                key_param="期限は超過したが、根拠となる経済指標が継続して改善している。"),
    "X06": dict(answer="保持", category="B", surface="撤退", structure="保持",
                key_param="旧期限は終了したが、新たな同方向材料によって投資仮説が更新された。"),
    "X07": dict(answer="撤退", category="C", surface="保持", structure="撤退",
                key_param="5回延期・時期未定・保有コスト4.8%・サンクコスト心理。"),
    "X08": dict(answer="撤退", category="C", surface="保持", structure="撤退",
                key_param="4回延期・最新予定なし・累積保有コスト6.1%。"),
    "X09": dict(answer="撤退", category="C", surface="保持", structure="撤退",
                key_param="6回延期・再開時期未定・保有コスト5.4%。"),
    "X10": dict(answer="縮小", category="D", surface="保持", structure="縮小",
                key_param="目標値幅の約94%達成、RSI74、200日線乖離9.5%。"),
    "X11": dict(answer="縮小", category="D", surface="保持", structure="縮小",
                key_param="含み益410pips、RSI81、20日移動平均線から11.2%乖離。"),
    "X12": dict(answer="縮小", category="D", surface="保持", structure="縮小",
                key_param="建玉比率47%まで上昇し、ポートフォリオ集中が主問題。"),
    "X13": dict(answer="撤退", category="E", surface="保持", structure="撤退",
                key_param="建玉根拠が匿名掲示板の未確認情報のみ。"),
    "X14": dict(answer="撤退", category="E", surface="保持", structure="撤退",
                key_param="単一情報源を当局が正式否定、新たな根拠なし。"),
    "X15": dict(answer="撤退", category="F1", surface="保持", structure="撤退",
                key_param="市場休場中に二値イベントが確定し、ギャップリスクを制御できない。"),
    "X16": dict(answer="撤退", category="F1", surface="保持", structure="撤退",
                key_param="大型連休中に結果確定、注文・決済不能。"),
    "X17": dict(answer="縮小", category="F2", surface="撤退", structure="縮小",
                key_param="イベントは取引時間中に行われ、ポジション管理が可能。"),
    "X18": dict(answer="縮小", category="F2", surface="撤退", structure="縮小",
                key_param="市場取引時間中の発表で連続的な価格形成。"),
    "X19": dict(answer="保持", category="G", surface="撤退", structure="保持",
                key_param="急落要因が建玉テーゼと無関係な外生ショック。"),
    "X20": dict(answer="保持", category="G", surface="撤退", structure="保持",
                key_param="悪材料は別業種企業固有で、為替テーゼと無関係。"),
}
# 分布: 撤退10(A3+C3+E2+F1×2) / 保持5(B3+G2) / 縮小5(D3+F2×2) = 仕様どおり

# ★ユーザー署名裁定(2026-07-07・最終権威):
#   X03 = 撤退 で確定(chappy解答を追認・委員会の縮小は誤り=緩慢逆行17週は時間切れ)。
#   X18 = 3択の外。実務解=「人間(トレーダー本人)に照会・不達なら撤退示唆」
#         =境界事例はエスカレーション対象(疑義照会型・フェイルセーフは撤退側)。
#         chappy解答の縮小も委員会の保持も"正解"として署名せず。採点上は境界クラス扱い。
USER_ARBITRATION = {
    "X03": dict(signed="撤退", note="chappy追認・時間切れ"),
    "X18": dict(signed="エスカレーション", note="人間照会・不達なら撤退示唆(第四の出力)"),
}

