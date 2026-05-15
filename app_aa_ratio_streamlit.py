from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from aa_ratio_inference import (
    AA20_ORDER,
    concentrations_to_diff_ratio,
    predict_adjusted_aa_composition,
)


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"


def _default_input_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AminoAcid": AA20_ORDER,
            "SelfConcentration": [1.0] * len(AA20_ORDER),
            "IdealConcentration": [1.0] * len(AA20_ORDER),
        }
    )


def _table_to_dicts(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    self_dict = dict(zip(df["AminoAcid"], df["SelfConcentration"]))
    ideal_dict = dict(zip(df["AminoAcid"], df["IdealConcentration"]))
    return self_dict, ideal_dict


def main() -> None:
    st.set_page_config(page_title="AA Ratio Predictor", page_icon="🧬", layout="wide")
    st.title("修正アミノ酸組成比率 予測アプリ")
    st.caption("自分と理想のアミノ酸濃度を入力すると、学習済みSVRモデルから修正アミノ酸組成比率を推定します。")

    with st.sidebar:
        st.header("設定")
        model_dir = Path(
            st.text_input("モデルディレクトリ", value=str(DEFAULT_MODEL_DIR))
        )
        calc_mode = st.selectbox(
            "差分比率の計算方法",
            options=["percent", "ratio"],
            format_func=lambda x: "(self-ideal)/ideal*100" if x == "percent" else "self/ideal",
        )

    st.subheader("入力: 自分濃度 と 理想濃度")
    st.write("20種類のアミノ酸を編集してください。")

    if "aa_input_df" not in st.session_state:
        st.session_state["aa_input_df"] = _default_input_table()

    edited_df = st.data_editor(
        st.session_state["aa_input_df"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "AminoAcid": st.column_config.TextColumn(disabled=True),
            "SelfConcentration": st.column_config.NumberColumn(format="%.6f"),
            "IdealConcentration": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    st.session_state["aa_input_df"] = edited_df.copy()

    col1, col2 = st.columns([1, 1])
    with col1:
        run = st.button("予測する", type="primary", use_container_width=True)
    with col2:
        if st.button("入力をリセット", use_container_width=True):
            st.session_state["aa_input_df"] = _default_input_table()
            st.rerun()

    if not run:
        return

    try:
        self_dict, ideal_dict = _table_to_dicts(edited_df)
        diff_ratio = concentrations_to_diff_ratio(
            self_concentration=self_dict,
            ideal_concentration=ideal_dict,
            mode=calc_mode,
        )

        pred = predict_adjusted_aa_composition(
            self_vs_ideal_diff_ratio=diff_ratio.to_dict(),
            model_dir=model_dir,
        )

        result_df = pd.DataFrame(
            {
                "AminoAcid": AA20_ORDER,
                "SelfConcentration": [self_dict[a] for a in AA20_ORDER],
                "IdealConcentration": [ideal_dict[a] for a in AA20_ORDER],
                "DiffRatioInput": [diff_ratio[a] for a in AA20_ORDER],
                "PredictedAdjustedRatio": [pred[a] for a in AA20_ORDER],
            }
        )

        st.success("予測が完了しました。")

        st.subheader("出力: 修正アミノ酸組成比率")
        st.dataframe(result_df, use_container_width=True)

        st.subheader("可視化")
        chart_df = result_df.set_index("AminoAcid")
        st.line_chart(chart_df[["SelfConcentration", "IdealConcentration", "PredictedAdjustedRatio"]])

        st.download_button(
            label="結果をCSVでダウンロード",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_adjusted_aa_ratio.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
