# streamlit_app.py
import streamlit as st
import numpy as np
from scipy.stats import f_oneway


PASSWORD = "dousai"

pw = st.text_input("パスワードを入力", type="password")
if pw != PASSWORD:
    st.warning("パスワードが違います")
    st.stop()


st.title("ラット体重ラウンドロビン割付＆ANOVA検定アプリ")
st.write("体重をカンマ区切りで入力すると、群分けと群間有意差を表示します。")

# 体重入力
weights_input = st.text_input("ラットの体重をカンマ区切りで入力（例: 250,248,252,260,255,249,251,258,247）")

# 群数設定
n_groups = st.number_input("群の数", min_value=2, max_value=10, value=3, step=1)

if st.button("計算"):
    try:
        # 入力を数値リストに変換
        weights = [float(w.strip()) for w in weights_input.split(",")]
        n = len(weights)
        if n % n_groups != 0:
            st.error(f"ラット数 ({n}) は群数 ({n_groups}) で割り切れません。")
        else:
            n_per_group = n // n_groups

            # 体重を大きい順にソートし、群の順番をランダムにして割付
            sorted_idx = sorted(range(n), key=lambda i: weights[i], reverse=True)
            group_order = np.random.permutation(n_groups)
            groups = [[] for _ in range(n_groups)]
            for i, rat in enumerate(sorted_idx):
                group = group_order[i % n_groups]
                groups[group].append(rat)

            group_info = []
            group_weights_list = []
            for g in groups:
                w = [weights[i] for i in g]
                group_weights_list.append(w)
                group_info.append({
                    "indices": sorted(g),
                    "weights": w,
                    "mean": np.mean(w)
                })

            # ANOVA検定
            f_stat, p_val = f_oneway(*group_weights_list)

            # 結果表示
            st.subheader("ラウンドロビン割付結果")
            for i, g in enumerate(group_info, 1):
                st.write(f"Group {i}: indices {g['indices']}, weights {g['weights']}, mean {round(g['mean'],2)}")

            st.write(f"全体平均: {round(np.mean(weights),2)}")
            st.write(f"ANOVA: F = {round(f_stat,4)}, p = {p_val}")
            if p_val < 0.05:
                st.success("→ 群間に有意差があります (p<0.05)")
            else:
                st.info("→ 群間に有意差はありません (p>=0.05)")

    except Exception as e:
        st.error(f"入力エラー: {e}")
