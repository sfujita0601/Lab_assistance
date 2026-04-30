from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def load_main_and_meta(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    file_name = uploaded_file.name
    stem_name = Path(file_name).stem

    errors: list[str] = []
    for engine in ["calamine", "openpyxl"]:
        try:
            excel = pd.ExcelFile(uploaded_file, engine=engine)
            sheet_names = excel.sheet_names
            if stem_name in sheet_names:
                main_sheet = stem_name
            else:
                main_sheet = next((s for s in sheet_names if s.lower() != "meta"), sheet_names[0])

            main_df = pd.read_excel(uploaded_file, sheet_name=main_sheet, header=None, engine=engine)
            meta_df = pd.read_excel(uploaded_file, sheet_name="meta", engine=engine)
            return main_df, meta_df, main_sheet
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
            uploaded_file.seek(0)

    raise ValueError(f"Excel read failed: {errors}")


def parse_standard_list(text_value: str) -> list[float]:
    values = [v.strip() for v in text_value.split(",") if v.strip()]
    return [float(v) for v in values]


def build_sample_names(meta_df: pd.DataFrame, pair_count: int) -> tuple[list[str], pd.DataFrame]:
    sample_names: list[str] = []

    id_col = next((c for c in meta_df.columns if str(c).lower() == "id"), None)
    label_col = next(
        (
            c
            for c in meta_df.columns
            if str(c).lower() == "label" or str(c).strip() == "ラベル"
        ),
        None,
    )

    if id_col is not None and label_col is not None:
        for _, row in meta_df.iterrows():
            sample_names.append(f"{row[id_col]}_{row[label_col]}")
    else:
        wide_id_cols = [str(c).strip() for c in meta_df.columns if str(c).strip().startswith("ID_")]

        if wide_id_cols:
            def sort_key(name: str) -> tuple[int, str]:
                suffix = name.split("_", 1)[1] if "_" in name else name
                return (int(suffix), name) if suffix.isdigit() else (10**9, name)

            wide_id_cols = sorted(wide_id_cols, key=sort_key)
            for _, row in meta_df[wide_id_cols].iterrows():
                parts = []
                for col_name in wide_id_cols:
                    value = row[col_name]
                    if pd.isna(value):
                        continue
                    value_str = str(value).strip()
                    if value_str:
                        parts.append(value_str)
                sample_names.append("_".join(parts) if parts else "Sample")

    if len(sample_names) == 0:
        sample_names = [f"Sample_{i+1}" for i in range(pair_count)]

    min_len = min(len(sample_names), pair_count)
    sample_names = sample_names[:min_len]

    meta_aligned = pd.DataFrame(index=range(min_len))
    if not meta_df.empty:
        meta_aligned = meta_df.iloc[:min_len].reset_index(drop=True).copy()
        meta_aligned.columns = [str(c).strip() for c in meta_aligned.columns]

    return sample_names, meta_aligned


def analyze_uploaded_xlsx(uploaded_file, standard_concentrations: list[float]) -> tuple[pd.DataFrame, dict]:
    main_df, meta_df, main_sheet = load_main_and_meta(uploaded_file)

    data_df = main_df.iloc[11:, :].copy()
    if data_df.shape[0] < 9 or data_df.shape[1] < 3:
        raise ValueError("12行目以降のプレート領域が不足しています。")

    plate_df = data_df.iloc[1:9, 1:].copy()
    plate_df = plate_df.apply(pd.to_numeric, errors="coerce")
    plate_df = plate_df.dropna(axis=1, how="all")
    if plate_df.shape[1] < 2:
        raise ValueError("プレートに標準列とサンプル列が見つかりません。")

    standards_abs = plate_df.iloc[:, 0].dropna().to_numpy()
    if len(standards_abs) < len(standard_concentrations):
        raise ValueError("標準吸光度の点数が不足しています。")

    standards_abs = standards_abs[: len(standard_concentrations)]
    standards_conc = np.array(standard_concentrations, dtype=float)

    slope, intercept = np.polyfit(standards_conc, standards_abs, 1)
    pred = slope * standards_conc + intercept
    ss_res = np.sum((standards_abs - pred) ** 2)
    ss_tot = np.sum((standards_abs - np.mean(standards_abs)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    sample_wells = plate_df.iloc[:, 1:]
    sample_abs = sample_wells.to_numpy().flatten(order="F")
    sample_abs = sample_abs[~np.isnan(sample_abs)]

    pair_count = len(sample_abs) // 2
    if pair_count == 0:
        raise ValueError("サンプル吸光度が見つかりません。")

    paired = sample_abs[: pair_count * 2].reshape(-1, 2)
    sample_abs_1 = paired[:, 0]
    sample_abs_2 = paired[:, 1]
    sample_abs_mean = paired.mean(axis=1)

    if abs(slope) < 1e-12:
        raise ValueError("標準直線の傾きが0に近く、濃度を逆算できません。")
    sample_conc = (sample_abs_mean - intercept) / slope

    sample_names, meta_aligned = build_sample_names(meta_df, pair_count)
    keep_n = len(sample_names)

    result_df = pd.DataFrame(
        {
            "SampleName": sample_names,
            "Absorbance_1": sample_abs_1[:keep_n],
            "Absorbance_2": sample_abs_2[:keep_n],
            "AbsorbanceMean": sample_abs_mean[:keep_n],
            "Concentration_g_per_ml": sample_conc[:keep_n],
            "StandardCurveSlope": slope,
            "StandardCurveIntercept": intercept,
            "StandardCurveR2": r_squared,
        }
    )

    for col in meta_aligned.columns:
        if col not in result_df.columns:
            result_df[col] = meta_aligned[col].to_numpy()

    info = {
        "main_sheet": main_sheet,
        "slope": slope,
        "intercept": intercept,
        "r2": r_squared,
        "standards_conc": standards_conc,
        "standards_abs": standards_abs,
    }
    return result_df, info


def make_standard_curve_figure(info: dict) -> plt.Figure:
    x = info["standards_conc"]
    y = info["standards_abs"]
    slope = info["slope"]
    intercept = info["intercept"]
    r2 = info["r2"]

    x_line = np.linspace(float(np.min(x)) * 0.8, float(np.max(x)) * 1.1, 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="tab:blue", label="Standards")
    ax.plot(x_line, y_line, color="tab:red", label=f"y={slope:.5f}x+{intercept:.5f}")
    ax.set_xlabel("Concentration (g/ml)")
    ax.set_ylabel("Absorbance")
    ax.text(
        0.05,
        0.95,
        f"R^2 = {r2:.5f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.legend()
    fig.tight_layout()
    return fig


def make_group_bar_scatter_figure(result_df: pd.DataFrame, group_column: str) -> plt.Figure:
    value_column = "Concentration_g_per_ml"
    plot_df = result_df[[group_column, value_column]].dropna().copy()
    plot_df[group_column] = plot_df[group_column].astype(str)

    group_order = sorted(plot_df[group_column].unique(), key=str)
    summary_df = (
        plot_df.groupby(group_column)[value_column]
        .agg(["mean", "std", "count"])
        .reindex(group_order)
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(group_order), 1)))
    fig, ax = plt.subplots(figsize=(max(5, len(group_order) * 1.6), 4.8))
    x = np.arange(len(group_order))

    ax.bar(
        x,
        summary_df["mean"].to_numpy(),
        yerr=summary_df["std"].fillna(0).to_numpy(),
        color=colors[: len(group_order)],
        alpha=0.7,
        capsize=5,
        edgecolor="black",
    )

    for idx, group_name in enumerate(group_order):
        vals = plot_df.loc[plot_df[group_column] == group_name, value_column].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(vals), x[idx]) + jitter,
            vals,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
            s=45,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_order)
    ax.set_xlabel(group_column)
    ax.set_ylabel(value_column)
    ax.set_title(f"{value_column} grouped by {group_column}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    PASSWORD = "dousai"

    pw = st.text_input("パスワードを入力", type="password")
    if pw != PASSWORD:
        st.warning("パスワードが違います")
        st.stop()
    st.set_page_config(page_title="TG Analyzer", layout="wide")
    st.title("TG Analyzer (xlsx upload)")

    st.markdown("アップロードした xlsx から、検量線と群別棒グラフを表示します。")

    uploaded = st.file_uploader("xlsxファイルを選択", type=["xlsx", "xlsm", "xls"])
    std_text = st.text_input(
        "標準濃度リスト（カンマ区切り）",
        "75,150,300,600,75,150,300,600",
    )

    if uploaded is None:
        st.info("xlsx ファイルをアップロードしてください。")
        return

    try:
        standard_concentrations = parse_standard_list(std_text)
        if len(standard_concentrations) == 0:
            st.error("標準濃度リストが空です。")
            return

        result_df, info = analyze_uploaded_xlsx(uploaded, standard_concentrations)
    except Exception as exc:
        st.error(f"解析に失敗しました: {exc}")
        return

    st.success(f"解析完了: main sheet = {info['main_sheet']}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("検量線")
        fig_std = make_standard_curve_figure(info)
        st.pyplot(fig_std)

    with c2:
        st.subheader("濃度結果")
        st.dataframe(result_df, use_container_width=True)

    group_candidates = [
        c
        for c in result_df.columns
        if c.startswith("ID_") and c != "ID_1"
    ]
    if len(group_candidates) == 0:
        group_candidates = [c for c in result_df.columns if c.startswith("ID_")]

    if len(group_candidates) > 0:
        group_col = st.selectbox("群分け列", group_candidates, index=0)
        st.subheader(f"棒グラフ + 散布図 ({group_col})")
        fig_group = make_group_bar_scatter_figure(result_df, group_col)
        st.pyplot(fig_group)
    else:
        st.warning("群分けに使える ID 列が見つかりませんでした。")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="結果CSVをダウンロード",
        data=csv_bytes,
        file_name=f"{Path(uploaded.name).stem}_concentration.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
