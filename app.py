"""④ 個人診断アプリ：②③の学習済みモデルをそのまま使ってボトルネックを判定する。

設計サマリー「④ 個人診断への実装」の実装。②（重回帰）と③（ロジスティック回帰）は
QWL設問由来の4変数（train / wrktime_r / suphelp_r / promteok_r）で学習したまま、
係数をいっさい変えずに搭載する（再学習しない）。

2つのモードを用意する。いずれも回答者の氏名を入力する前提とする。

1. 組織向け一覧（CSV一括）
   多数の回答を読み込み、偏差基準で「働く環境が整っていない度合い」を並べて介入先を選ぶ。

2. QWL4問モード（1人）
   本人に4問だけ答えてもらい、実測値で判定する。推定誤差が入らない。

設計サマリー④が指定していた「8項目の基本プロフィールから推定する」方式は搭載しない。
実測診断との一致率が 28.3%（全員に一律「支援」と答える 31.1% より低い）で、
個人の処方に使えないことが検証で分かったため。

起動:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sqlite3

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 設定 ----

DB_PATH = "gss_qwl.db"
CSV_PATH = "gss_qwl_recent_subset.csv"
SAMPLE_CSV = "sample_responses.csv"

X_COLS = ["train", "wrktime_r", "suphelp_r", "promteok_r"]
Y_COL = "satjob1_r"

CATEGORY = {
    "train": "能力",
    "wrktime_r": "余力",
    "suphelp_r": "支援",
    "promteok_r": "機会",
}
# CSV で受け取る「素の設問名」と、変換後の変数名の対応
RAW_TO_R = {
    "emptrain": "train",
    "wrktime": "wrktime_r",
    "suphelp": "suphelp_r",
    "promteok": "promteok_r",
}
# 画面での4件法の表示。main.ipynb の _r 変数（5 − 原コード、4が最上位）に合わせる。
# 表示される値がそのままモデルに入るので、画面内で数字が食い違わない。
SCALE4 = {
    4: "4：とてもそう思う",
    3: "3：ややそう思う",
    2: "2：あまりそう思わない",
    1: "1：まったくそう思わない",
}
TRAIN_LABEL = {1: "1：Yes（受けた）", 0: "0：No（受けていない）"}
CEILING = {"train": 1, "wrktime_r": 4, "suphelp_r": 4, "promteok_r": 4}

NAME_COL = "name"           # 氏名（必須）
CLOSE_MARGIN = 0.05         # 1位と2位の寄与度差がこれ未満なら「実質同順位」として併記する

# 画面に出る予測満足度の説明。1人ぶんの値（PRED_HELP）と、
# それを読み込んだ人数ぶん平均した値（AVG_PRED_HELP）は別物なので、説明も分ける。
SCALE_HELP = (
    "\n\n- 4：とても満足"
    "\n- 3：やや満足"
    "\n- 2：あまり満足していない"
    "\n- 1：まったく満足していない"
)
PRED_HELP = "同じ4つの回答をした人たちの、満足度の平均値。" + SCALE_HELP
AVG_PRED_HELP = "読み込んだ全員の予測満足度を、さらに平均した値。" + SCALE_HELP
HIGH_SAT_HELP = "同じ4つの回答をした人たちのうち、「4：とても満足」と答えた人の割合。"

st.set_page_config(page_title="職場ボトルネック診断", page_icon="🧭", layout="wide")


# ------------------------------------------------------ データとモデル ----


@st.cache_data(show_spinner="データを読み込んでいます…")
def load_work() -> pd.DataFrame:
    """分析対象者（2018・2022年／就業中／被雇用者＝2,928人）を作る。

    main.ipynb の 0-2〜2 とまったく同じ手順。DB がなければ CSV から作る。
    """
    if os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM gss_qwl", conn)
    elif os.path.exists(CSV_PATH):
        raw = pd.read_csv(CSV_PATH)
        df = raw[raw["year"] != 2014].copy()
    else:
        st.error(
            f"データが見つかりません。`{DB_PATH}` または `{CSV_PATH}` を "
            "このスクリプトと同じフォルダに置いてください。"
        )
        st.stop()

    work = df[df["wrkstat"].isin([1, 2]) & (df["wrkslf"] == 2)].copy()

    work["train"] = work["emptrain"].map({1: 1, 2: 0})
    work["wrktime_r"] = 5 - work["wrktime"]
    work["suphelp_r"] = 5 - work["suphelp"]
    work["promteok_r"] = 5 - work["promteok"]
    work["satjob1_r"] = 5 - work["satjob1"]

    return work


@st.cache_resource(show_spinner="モデルを学習しています…")
def fit_models(work: pd.DataFrame) -> dict:
    """重回帰（設計サマリー②）とロジスティック回帰（③）を、main.ipynb と同じ設定で学習させる。"""
    reg_data = work.dropna(subset=[Y_COL] + X_COLS)
    X = reg_data[X_COLS]
    y = reg_data[Y_COL]

    ols = LinearRegression().fit(X, y)
    ols_std = LinearRegression().fit(
        StandardScaler().fit_transform(X),
        StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel(),
    )

    high_sat = (reg_data[Y_COL] == 4).astype(int)
    logit = LogisticRegression(C=1e9, max_iter=1000).fit(X, high_sat)
    logit_std = LogisticRegression(C=1e9, max_iter=1000).fit(
        StandardScaler().fit_transform(X), high_sat
    )

    return {
        "n": len(reg_data),
        "means": X.mean(),
        "ols": ols,
        "ols_coef": pd.Series(ols.coef_, index=X_COLS),
        "r2": ols.score(X, y),
        "beta": pd.Series(ols_std.coef_, index=X_COLS),
        "logit": logit,
        "logit_coef": pd.Series(logit.coef_[0], index=X_COLS),
        "accuracy": logit.score(X, high_sat),
        "base_rate": max(high_sat.mean(), 1 - high_sat.mean()),
        "odds_ratio": pd.Series(np.exp(logit.coef_[0]), index=X_COLS),
        "odds_ratio_std": pd.Series(np.exp(logit_std.coef_[0]), index=X_COLS),
    }


# ------------------------------------------------------------ 判定処理 ----


def contributions(values: pd.DataFrame, models: dict) -> pd.DataFrame:
    """「（値 − 全体平均）× 重回帰の係数」を、複数人ぶんまとめて計算する。"""
    return (values[X_COLS] - models["means"]) * models["ols_coef"]


def diagnose(values: pd.DataFrame, models: dict) -> pd.DataFrame:
    """寄与度から優先ボトルネックを判定し、2モデルの一致と順位の僅差を添えて返す。"""
    contrib = contributions(values, models)
    logit_contrib = (values[X_COLS] - models["means"]) * models["logit_coef"]

    ordered = np.sort(contrib.values, axis=1)
    out = pd.DataFrame(index=values.index)
    out["優先ボトルネック"] = contrib.idxmin(axis=1).map(CATEGORY)
    out["最小寄与度"] = contrib.min(axis=1)
    out["1位2位の差"] = ordered[:, 1] - ordered[:, 0]
    out["合計寄与度"] = contrib.sum(axis=1)
    out["予測満足度"] = models["ols"].predict(values[X_COLS])
    out["高満足確率"] = models["logit"].predict_proba(values[X_COLS])[:, 1]
    out["分類モデルの判定"] = logit_contrib.idxmin(axis=1).map(CATEGORY)
    out["2モデルの一致"] = np.where(out["優先ボトルネック"] == out["分類モデルの判定"], "○", "×")
    out["僅差"] = np.where(out["1位2位の差"] < CLOSE_MARGIN, "⚠️ 実質同順位", "")
    for col in X_COLS:
        out[f"寄与度_{CATEGORY[col]}"] = contrib[col]
    return out


def read_responses(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """CSVの素の回答（emptrain 1/2、他は1〜4）を、モデルが使う4変数に変換する。

    氏名（name）は必須。誰に介入するかを決めるための一覧なので、匿名行は受け付けない。
    """
    problems = []
    required = [NAME_COL] + list(RAW_TO_R)
    missing = [c for c in required if c not in raw.columns]
    if missing:
        problems.append("必須列がありません：" + "、".join(missing))
        return pd.DataFrame(), problems

    df = raw.copy()
    df[NAME_COL] = df[NAME_COL].astype(str).str.strip()
    unnamed = df[NAME_COL].isin(["", "nan", "None"])
    if unnamed.any():
        problems.append(f"氏名が空の行が {int(unnamed.sum())} 件あります（除外）")
        df = df[~unnamed]

    dup = df[NAME_COL].duplicated(keep=False)
    if dup.any():
        problems.append(
            f"氏名が重複しています：{'、'.join(sorted(df.loc[dup, NAME_COL].unique()))}"
            "（同姓同名の場合は employee_id 列で区別してください）"
        )

    df["train"] = df["emptrain"].map({1: 1, 2: 0, 0: 0})
    for src in ["wrktime", "suphelp", "promteok"]:
        vals = pd.to_numeric(df[src], errors="coerce")
        bad = vals.notna() & ~vals.isin([1, 2, 3, 4])
        if bad.any():
            problems.append(f"`{src}` に1〜4以外の値が {int(bad.sum())} 件あります（除外）")
        df[RAW_TO_R[src]] = 5 - vals.where(vals.isin([1, 2, 3, 4]))

    incomplete = df[X_COLS].isna().any(axis=1)
    if incomplete.any():
        problems.append(f"4項目が揃わない行が {int(incomplete.sum())} 件あります（除外）")

    return df[~incomplete].reset_index(drop=True), problems


# ---------------------------------------------------------------- 画面 ----

work = load_work()
models = fit_models(work)

st.title("🧭 職場ボトルネック診断")
st.caption(
    "職務満足を数値で予測する**重回帰モデル**と、高満足かどうかを判定する"
    "**分類モデル（ロジスティック回帰）**。この2つを再学習せずにそのまま搭載し、"
    "4つのボトルネック（能力・余力・支援・機会）のうちどれを優先すべきかを判定する。"
)

mode = st.sidebar.radio(
    "モード",
    ["🏢 組織向け一覧（CSV一括）", "🙋 QWL4問モード（1人）", "📐 モデルの中身"],
)
st.sidebar.divider()


# ================================================== 組織向け一覧（CSV） ====

if mode.startswith("🏢"):
    st.subheader("組織向け一覧：介入先を選ぶ")
    st.caption(
        "偏差基準（(値 − 全回答者平均) × 重回帰の係数）で、全体平均と比べて働く環境が"
        "整っていない度合いを測り、支援の優先度が高い順に並べる。記名・個人単位を前提とする。"
    )
    st.warning(
        "**前提**：回答者が率直に答えることを前提としている。記名式では `suphelp`（上司の助力）が"
        "過大に評価されやすいが、本アプリではその影響を検証・補正していない。"
    )

    with st.sidebar:
        st.markdown("**CSVの形式**")
        st.code(
            "name,employee_id,team,emptrain,wrktime,suphelp,promteok\n"
            "山田 太郎,E001,営業1課,1,2,3,3\n"
            "鈴木 花子,E002,開発部,2,1,1,4",
            language="text",
        )
        st.caption(
            "`name`（氏名）は必須。`employee_id` `team` は任意。"
            "`emptrain` は 1=Yes / 2=No（人事データから結合可）。"
            "他の3問は**調査票の原コード**で 1=とてもそう思う 〜 4=まったくそう思わない"
            "（main.ipynb と同じく、読み込み時に `5 − 原コード` へ変換する）。"
            "4問モードの画面表示は変換後の値（4が最上位）なので、向きが逆になる点に注意。"
        )
        template = pd.DataFrame(
            {
                "name": ["山田 太郎", "鈴木 花子"],
                "employee_id": ["E001", "E002"],
                "team": ["営業1課", "開発部"],
                "emptrain": [1, 2],
                "wrktime": [2, 1],
                "suphelp": [3, 1],
                "promteok": [3, 4],
            }
        )
        st.download_button(
            "テンプレートCSVをダウンロード",
            template.to_csv(index=False).encode("utf-8-sig"),
            file_name="qwl_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("回答CSVをアップロード", type="csv")
    use_sample = st.checkbox(
        f"サンプルデータ（`{SAMPLE_CSV}`）で試す", value=uploaded is None
    )

    raw = None
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    elif use_sample and os.path.exists(SAMPLE_CSV):
        raw = pd.read_csv(SAMPLE_CSV)
        st.info(
            f"サンプルデータ `{SAMPLE_CSV}` を読み込んでいる"
            "（回答はGSSの実データ60人ぶん、氏名・所属はダミー）。"
        )

    if raw is None:
        st.stop()

    responses, problems = read_responses(raw)
    for p in problems:
        st.warning(p)
    if responses.empty:
        st.stop()

    result = diagnose(responses, models)
    label_cols = [c for c in [NAME_COL, "employee_id", "team"] if c in responses.columns]
    table = pd.concat([responses[label_cols], result], axis=1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("対象者", f"{len(table)}人")
    c2.metric("平均予測満足度", f"{table['予測満足度'].mean():.2f}", help=AVG_PRED_HELP)
    c3.metric("2モデルが一致", f"{(table['2モデルの一致'] == '○').mean():.0%}")
    c4.metric(
        "上位2つが僅差",
        f"{(table['僅差'] != '').mean():.0%}",
        help=f"1番目と2番目のボトルネックの差が {CLOSE_MARGIN} 未満だった人の割合。"
        "この人たちはどちらが1位とは言い切れないので、上位2つを併記して読む。",
    )

    st.markdown("#### ボトルネックの分布（何人に何が必要か）")
    dist = (
        table["優先ボトルネック"].value_counts().rename_axis("ボトルネック").reset_index(name="人数")
    )
    st.altair_chart(
        alt.Chart(dist)
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X("人数:Q"),
            y=alt.Y("ボトルネック:N", sort="-x"),
            tooltip=["ボトルネック", "人数"],
        )
        .properties(height=160),
        width="stretch",
    )

    st.markdown("#### 介入先の候補（合計寄与度が低い順）")
    st.caption(
        "合計寄与度＝4カテゴリの寄与度の合計。マイナスが大きいほど、"
        "全体平均と比べて職場環境が不利な状態にある。"
    )

    only = st.multiselect(
        "ボトルネックで絞り込む",
        options=sorted(table["優先ボトルネック"].unique()),
        default=[],
        help="「支援の介入先だけ見たい」といった用途に使う。",
    )
    view = table if not only else table[table["優先ボトルネック"].isin(only)]
    view = view.sort_values("合計寄与度")

    show_cols = (
        label_cols
        + [
            "優先ボトルネック",
            "僅差",
            "分類モデルの判定",
            "2モデルの一致",
            "合計寄与度",
            "最小寄与度",
            "予測満足度",
            "高満足確率",
        ]
        + [f"寄与度_{CATEGORY[c]}" for c in X_COLS]
    )

    st.dataframe(
        view[show_cols].style.format(
            {
                "合計寄与度": "{:+.3f}",
                "最小寄与度": "{:+.3f}",
                "予測満足度": "{:.2f}",
                "高満足確率": "{:.1%}",
                **{f"寄与度_{CATEGORY[c]}": "{:+.3f}" for c in X_COLS},
            }
        ).background_gradient(subset=["合計寄与度"], cmap="RdYlGn"),
        width="stretch",
        hide_index=True,
        height=420,
    )

    st.download_button(
        "診断結果をCSVでダウンロード",
        view[show_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="qwl_diagnosis.csv",
        mime="text/csv",
    )

    st.caption(
        "**取り扱い注意**：この一覧は氏名つきの職場評価であり、"
        "とくに `suphelp`（上司の助力）は本人と上司の関係に直結する。"
        "閲覧範囲を限定し、当人の上司にそのまま共有しない運用を推奨する。"
    )

    if "team" in table.columns:
        st.markdown("#### チーム別の集計（参考）")
        st.caption(
            "`suphelp` は本来その上司の性質なので、支援についてはチームが自然な単位になる。"
        )
        team = table.groupby("team").agg(
            人数=("優先ボトルネック", "size"),
            合計寄与度の平均=("合計寄与度", "mean"),
            予測満足度=("予測満足度", "mean"),
            最多ボトルネック=("優先ボトルネック", lambda s: s.mode().iat[0]),
        )
        for col in X_COLS:
            team[f"寄与度_{CATEGORY[col]}"] = table.groupby("team")[
                f"寄与度_{CATEGORY[col]}"
            ].mean()
        st.dataframe(
            team.sort_values("合計寄与度の平均").style.format(
                {
                    "合計寄与度の平均": "{:+.3f}",
                    "予測満足度": "{:.2f}",
                    **{f"寄与度_{CATEGORY[c]}": "{:+.3f}" for c in X_COLS},
                }
            ),
            width="stretch",
        )

    st.info(
        f"**僅差の扱い**：1位と2位の寄与度差が {CLOSE_MARGIN} 未満の人には「⚠️ 実質同順位」を付けている。"
        "係数は回答者全体の平均から求めたもので、人によって効き方が違う可能性は反映していない。"
        "差が小さい場合は1位を断定せず、上位2つを併記して読むこと。"
    )


# ================================================== QWL4問モード（1人） ====

elif mode.startswith("🙋"):
    st.subheader("QWL4問モード：本人の回答で判定する")
    st.caption("4変数の実測値をそのまま両モデルに代入するので、推定誤差が入らない。")

    # スライダー両端に出る選択肢ラベル（グレーの目盛り表示）を消す
    st.markdown(
        '<style>[data-testid="stSliderTickBar"]{display:none;}</style>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        name = st.text_input("氏名", placeholder="山田 太郎")
        team = st.text_input("所属（任意）", placeholder="営業1課")
        st.markdown("**社員の回答**")
        train = st.radio(
            "過去12か月に、勤務先から公式の研修を受けましたか",
            options=[1, 0],
            format_func=lambda v: TRAIN_LABEL[v],
            horizontal=True,
        )
        wrktime_r = st.select_slider(
            "仕事を終わらせるのに十分な時間がある",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda v: SCALE4[v],
        )
        suphelp_r = st.select_slider(
            "上司は、私が仕事を進めるうえで助けになってくれる",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda v: SCALE4[v],
        )
        promteok_r = st.select_slider(
            "昇進の見込みがある",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda v: SCALE4[v],
        )

    name = (name or "").strip()
    if not name:
        st.info("サイドバーで氏名を入力すると診断を表示する。")
        st.stop()

    # 画面の選択肢は main.ipynb の _r 変数と同じ向きなので、変換せずそのまま使う
    values = pd.DataFrame(
        [
            {
                "train": train,
                "wrktime_r": wrktime_r,
                "suphelp_r": suphelp_r,
                "promteok_r": promteok_r,
            }
        ]
    )
    res = diagnose(values, models).iloc[0]
    contrib = contributions(values, models).iloc[0]

    who = f"{name} さん" + (f"（{team.strip()}）" if team and team.strip() else "")

    if res["最小寄与度"] < 0:
        st.success(
            f"**{who}** の優先ボトルネックは **{res['優先ボトルネック']}**"
            f"（寄与度 {res['最小寄与度']:+.3f}）。"
        )
    else:
        st.info(
            f"**{who}** は4カテゴリとも全体平均を上回っており、明確なボトルネックは検出されない"
            f"（最小は {res['優先ボトルネック']}、{res['最小寄与度']:+.3f}）。"
        )

    if res["僅差"]:
        st.warning(
            f"1位と2位の差が {res['1位2位の差']:.3f} と小さい。"
            f"分類モデルでは「{res['分類モデルの判定']}」が最下位になっており、順位は確定的ではない。"
            "上位2つを併記して読むこと。"
        )
    elif res["2モデルの一致"] == "×":
        st.warning(
            f"分類モデルでは「{res['分類モデルの判定']}」が最もマイナスで、重回帰の結果と食い違っている。"
        )
    else:
        st.caption(
            f"分類モデルでも「{res['分類モデルの判定']}」が最下位で、重回帰と一致した"
            "（ダブルチェック：一致）。"
        )

    c1, c2 = st.columns(2)
    c1.metric("予測満足度（1〜4）", f"{res['予測満足度']:.2f}", help=PRED_HELP)
    c2.metric("高満足になる確率", f"{res['高満足確率']:.1%}", help=HIGH_SAT_HELP)

    detail = pd.DataFrame(
        {
            "カテゴリ": [CATEGORY[c] for c in X_COLS],
            "本人の回答": [values.iloc[0][c] for c in X_COLS],
            "全体平均": models["means"].values,
            "偏差": [values.iloc[0][c] - models["means"][c] for c in X_COLS],
            "寄与度": contrib.values,
            "改善余地": [
                (CEILING[c] - values.iloc[0][c]) * models["ols_coef"][c] for c in X_COLS
            ],
        }
    )

    st.altair_chart(
        alt.Chart(detail)
        .mark_bar()
        .encode(
            x=alt.X("寄与度:Q", title="満足度への寄与度"),
            y=alt.Y("カテゴリ:N", sort=[CATEGORY[c] for c in X_COLS], title=""),
            color=alt.condition(
                alt.datum["寄与度"] < 0, alt.value("#d95f5f"), alt.value("#4c78a8")
            ),
            tooltip=["カテゴリ", alt.Tooltip("寄与度:Q", format=".3f")],
        )
        .properties(height=180),
        width="stretch",
    )

    st.dataframe(
        detail.style.format(
            {
                "全体平均": "{:.2f}",
                "偏差": "{:+.2f}",
                "寄与度": "{:+.3f}",
                "改善余地": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "**寄与度**は「全体平均と比べてどれだけ不利か」＝組織が介入先を選ぶための量。"
        "**改善余地**は「最上位まで上げたときに戻る満足度」＝本人が何から変えるかを選ぶための量。"
        "問いが違うので、両者が別のカテゴリを指すことがある。"
    )

    st.download_button(
        "この診断結果をCSVでダウンロード",
        detail.assign(氏名=name, 所属=team, 優先ボトルネック=res["優先ボトルネック"])
        .to_csv(index=False)
        .encode("utf-8-sig"),
        file_name=f"qwl_diagnosis_{name}.csv",
        mime="text/csv",
    )


# ======================================================= モデルの中身 ====

else:
    st.subheader("搭載している学習済みモデル")
    st.caption(
        f"学習に使ったのは分析対象者 {len(work):,} 人のうち、"
        f"目的変数と4変数がすべて揃う {models['n']:,} 人。"
        "どのモードでも、この係数は変更しない（再学習しない）。"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("決定係数 R²", f"{models['r2']:.3f}")
    c2.metric("正答率", f"{models['accuracy']:.3f}")
    c3.metric("基準率", f"{models['base_rate']:.3f}")

    model_tbl = pd.DataFrame(
        {
            "ボトルネック": [CATEGORY[c] for c in X_COLS],
            "変数": X_COLS,
            "回帰係数": models["ols_coef"].values,
            "標準化係数 β": models["beta"].values,
            "オッズ比": models["odds_ratio"].values,
            "標準化オッズ比": models["odds_ratio_std"].values,
            "全回答者平均": models["means"].values,
        }
    ).sort_values("標準化係数 β", ascending=False)

    st.dataframe(
        model_tbl.style.format(
            {
                "回帰係数": "{:.4f}",
                "標準化係数 β": "{:.3f}",
                "オッズ比": "{:.3f}",
                "標準化オッズ比": "{:.3f}",
                "全回答者平均": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        f"""
- 重回帰の切片は **{models['ols'].intercept_:.4f}**。予測満足度 = 切片 + Σ(係数 × 値)。
  予測満足度は、同じ4つの回答をした人たちの、満足度の平均値である。
- 分類モデルは `LogisticRegression(C=1e9, max_iter=1000)`。解釈が目的なので正則化を実質的に外している。
- `train` だけが0/1のダミーなので、オッズ比を横並びに比較できない。順位比較は標準化した列で行う。
"""
    )

    st.markdown(
        """
#### このアプリが主張できる範囲

| | |
| --- | --- |
| 言える | この人の4つの入力のうち、集団平均の重みで見てどれから手をつけるべきかの順位 |
| 言えない | この人の満足度がいくつになるか（R² = 0.251、7割以上は説明の外） |
| 言えない | 因果関係（反復横断データであり、相関のみ） |
| 留保 | 係数は回答者全体の平均。40歳で2分割して推定し直すと、11.1%の人で優先ボトルネックの1位が入れ替わる |
| 留保 | ウェイト（`wtssps` / `wtssnrps`）は使っていないので、全米水準の推定には使えない |

個人単位の係数は原理的に推定できない（同一個人を追跡しないため、1人の中での変化を観測できない）。
属性グループ単位の係数は推定可能で、支援が1位・能力が最下位という順位はどの層でも変わらないが、
2位以下の重みは年齢層・職種水準で2倍前後動く。
"""
    )
