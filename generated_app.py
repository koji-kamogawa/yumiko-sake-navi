import streamlit as st
import requests
import os
import json
import math
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# ---------- Sakenowa API データ取得（キャッシュ） ----------
@st.cache_data
def fetch_json(url: str) -> dict:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

@st.cache_data
def get_areas() -> list:
    return fetch_json("https://muro.sakenowa.com/sakenowa-data/api/areas")["areas"]

@st.cache_data
def get_brands() -> list:
    return fetch_json("https://muro.sakenowa.com/sakenowa-data/api/brands")["brands"]

@st.cache_data
def get_breweries() -> list:
    return fetch_json("https://muro.sakenowa.com/sakenowa-data/api/breweries")["breweries"]

@st.cache_data
def get_flavor_tags() -> list:
    return fetch_json("https://muro.sakenowa.com/sakenowa-data/api/flavor-tags")["tags"]

@st.cache_data
def get_flavor_charts() -> list:
    data = fetch_json("https://muro.sakenowa.com/sakenowa-data/api/flavor-charts")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["flavorChart", "flavorCharts", "flavor-charts"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        for val in data.values():
            if isinstance(val, list):
                return val
    return []

@st.cache_data
def get_brand_flavor_tags() -> dict:
    items = fetch_json("https://muro.sakenowa.com/sakenowa-data/api/brand-flavor-tags")["flavorTags"]
    return {item["brandId"]: item["tagIds"] for item in items}

@st.cache_data
def get_rankings() -> list:
    return fetch_json("https://muro.sakenowa.com/sakenowa-data/api/rankings")["overall"]

# ---------- データ準備 ----------
def prepare_data():
    areas = get_areas()
    brands = get_brands()
    breweries = get_breweries()
    flavor_tags = get_flavor_tags()
    flavor_charts_list = get_flavor_charts()
    flavor_charts_dict = {c["brandId"]: c for c in flavor_charts_list}
    brand_flavor_tags = get_brand_flavor_tags()
    rankings = get_rankings()
    ranking_map = {r["brandId"]: r for r in rankings}

    # マッピング作成
    area_id_to_name = {a["id"]: a["name"] for a in areas}
    area_name_to_id = {a["name"]: a["id"] for a in areas}
    brewery_map = {b["id"]: b for b in breweries}

    # ブランド情報を拡充
    brand_list = []
    for brand in brands:
        bid = brand["id"]
        brewery_id = brand.get("breweryId")
        brewery_info = brewery_map.get(brewery_id, {})
        area_id = brewery_info.get("areaId")
        area_name = area_id_to_name.get(area_id, "不明")
        chart = flavor_charts_dict.get(bid, {})
        ft_values = [
            chart.get("f1", 0.5),
            chart.get("f2", 0.5),
            chart.get("f3", 0.5),
            chart.get("f4", 0.5),
            chart.get("f5", 0.5),
            chart.get("f6", 0.5)
        ]
        rank_info = ranking_map.get(bid, {})
        rank_score = rank_info.get("score", 0)
        rank_order = rank_info.get("rank", 9999)
        brand_list.append({
            "id": bid,
            "name": brand["name"],
            "breweryId": brewery_id,
            "breweryName": brewery_info.get("name", "不明"),
            "areaId": area_id,
            "areaName": area_name,
            "type": brand.get("type", ""),
            "rice": brand.get("rice", ""),
            "ft_values": ft_values,
            "rank_score": rank_score,
            "rank_order": rank_order
        })

    return {
        "areas": areas,
        "brands": brand_list,
        "breweries": breweries,
        "flavor_tags": flavor_tags,
        "flavor_charts_dict": flavor_charts_dict,
        "brand_flavor_tags": brand_flavor_tags,
        "ranking_map": ranking_map,
        "area_names": sorted(list(area_name_to_id.keys())),
        "brewery_names": sorted([b["name"] for b in breweries]),
        "area_id_to_name": area_id_to_name,
        "area_name_to_id": area_name_to_id,
        "brewery_map": brewery_map
    }

# ---------- レーダーチャート作成 ----------
def create_radar_chart(ft_values: list, name: str = "", color: str = "#1f77b4", fill: str = "toself", opacity: float = 0.3) -> go.Scatterpolar:
    ft_labels = ["華やか", "芳醇", "重厚", "穏やか", "ドライ", "軽快"]
    return go.Scatterpolar(
        r=ft_values + [ft_values[0]],
        theta=ft_labels + [ft_labels[0]],
        fill=fill,
        name=name,
        line=dict(color=color),
        fillcolor=color.replace("1", "0.3") if opacity < 1 else color,
        opacity=opacity
    )

# ---------- コサイン類似度計算（NumPyのみで実装） ----------
def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2つの行列間のコサイン類似度を計算する"""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_normalized = a / np.where(a_norm == 0, 1, a_norm)
    b_normalized = b / np.where(b_norm == 0, 1, b_norm)
    return np.dot(a_normalized, b_normalized.T)

# ---------- 類似度計算 ----------
def compute_similarity(target_ft: list, all_brands: list) -> list:
    target = np.array(target_ft).reshape(1, -1)
    all_ft = np.array([b["ft_values"] for b in all_brands])
    similarities = cosine_similarity_numpy(target, all_ft)[0]
    results = []
    for i, brand in enumerate(all_brands):
        results.append({
            **brand,
            "similarity": float(similarities[i])
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results

# ---------- フィルタリング ----------
def filter_brands(brands: list, selected_areas: list, selected_brewery: str, selected_type: str, selected_rice: str) -> list:
    filtered = brands
    if selected_areas:
        filtered = [b for b in filtered if b["areaName"] in selected_areas]
    if selected_brewery and selected_brewery != "すべて":
        filtered = [b for b in filtered if b["breweryName"] == selected_brewery]
    if selected_type:
        filtered = [b for b in filtered if selected_type.lower() in (b.get("type", "") or "").lower()]
    if selected_rice:
        filtered = [b for b in filtered if selected_rice.lower() in (b.get("rice", "") or "").lower()]
    return filtered

# ---------- Streamlit アプリ ----------
def main():
    st.set_page_config(page_title="味で選ぶ、日本酒 NAVI", layout="wide")
    # タイトル左側に画像を表示
    col1, col2 = st.columns([1, 8], vertical_alignment="center")
    with col1:
        st.image("sake_logo.jpg", width=80)
    with col2:
        st.title("味で選ぶ、日本酒 NAVI")
    st.markdown("味の特徴を分析しながら、あなたの好みに合う日本酒を探索できます。")
    st.markdown(
    '<a href="https://sakenowa.com" target="_blank">【さけのわデータ】を利用しています</a>',
    unsafe_allow_html=True
)

    with st.spinner("さけのわデータを読み込み中..."):
        try:
            data = prepare_data()
        except Exception as e:
            st.error(f"データの取得に失敗しました: {e}")
            return

    # ---------- サイドバー ----------
    st.sidebar.header("🔍 フィルタ")
    selected_areas = st.sidebar.multiselect("地域", data["area_names"])
    brewery_options = ["すべて"] + data["brewery_names"]
    selected_brewery = st.sidebar.selectbox("蔵元", brewery_options)
    selected_type = st.sidebar.text_input("種類（例: 純米大吟醸）")
    selected_rice = st.sidebar.text_input("原料米（例: 山田錦）")

    st.sidebar.header("🎨 フレーバーコントロール")
    f1 = st.sidebar.slider("華やか", 0.0, 1.0, 0.5, 0.01)
    f2 = st.sidebar.slider("芳醇", 0.0, 1.0, 0.5, 0.01)
    f3 = st.sidebar.slider("重厚", 0.0, 1.0, 0.5, 0.01)
    f4 = st.sidebar.slider("穏やか", 0.0, 1.0, 0.5, 0.01)
    f5 = st.sidebar.slider("ドライ", 0.0, 1.0, 0.5, 0.01)
    f6 = st.sidebar.slider("軽快", 0.0, 1.0, 0.5, 0.01)

    st.sidebar.header("🔍 検索モード")
    search_mode = st.sidebar.radio(
        "検索モードを選択",
        ["銘柄ベース検索", "フレーバー直接指定検索"]
    )

    # フィルタ適用
    filtered_brands = filter_brands(data["brands"], selected_areas, selected_brewery, selected_type, selected_rice)

    # セッションステート初期化
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
        st.session_state.results = []
        st.session_state.target_ft = None
        st.session_state.target_name = ""

    # ---------- メインコンテンツ ----------
    if search_mode == "銘柄ベース検索":
        st.subheader("📋 銘柄ベース検索")
        brand_names = ["--- 銘柄を選択 ---"] + [b["name"] for b in filtered_brands]
        selected_brand_name = st.selectbox("銘柄を選択", brand_names)

        if st.button("この銘柄と似た日本酒を検索"):
            if selected_brand_name == "--- 銘柄を選択 ---":
                st.warning("銘柄を選択してください。")
            else:
                selected_brand = next((b for b in filtered_brands if b["name"] == selected_brand_name), None)
                if selected_brand:
                    target_ft = selected_brand["ft_values"]
                    results = compute_similarity(target_ft, filtered_brands)
                    st.session_state.results = results[:10]
                    st.session_state.target_ft = target_ft
                    st.session_state.target_name = selected_brand_name
                    st.session_state.show_results = True

    else:  # フレーバー直接指定検索
        st.subheader("🎨 フレーバー直接指定検索")
        st.markdown("サイドバーのスライダーでフレーバーを調整し、検索ボタンを押してください。")

        # 現在のスライダー値でレーダーチャートプレビュー
        current_ft = [f1, f2, f3, f4, f5, f6]
        fig_preview = go.Figure(data=create_radar_chart(current_ft, "あなたの好み", "#ff6b6b", "toself", 0.4))
        fig_preview.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_preview, width="stretch")

        if st.button("このフレーバーで検索"):
            target_ft = [f1, f2, f3, f4, f5, f6]
            results = compute_similarity(target_ft, filtered_brands)
            st.session_state.results = results[:10]
            st.session_state.target_ft = target_ft
            st.session_state.target_name = "あなたの好み"
            st.session_state.show_results = True

    # ---------- 結果表示 ----------
    if st.session_state.show_results and st.session_state.results:
        st.divider()
        st.subheader(f"🔍 検索結果: 「{st.session_state.target_name}」に似た銘柄")

        # ソート切り替え
        sort_option = st.radio("並び順", ["類似度順", "ランキング順"], horizontal=True)
        display_results = st.session_state.results.copy()
        if sort_option == "ランキング順":
            display_results.sort(key=lambda x: x["rank_order"])
        else:
            display_results.sort(key=lambda x: x["similarity"], reverse=True)

        # テーブル表示
        table_data = []
        for i, r in enumerate(display_results):
            table_data.append({
                "順位": i + 1,
                "銘柄名": r["name"],
                "地域": r["areaName"],
                "蔵元": r["breweryName"],
                "類似度": f"{r['similarity']:.4f}",
                "ランク": r["rank_order"]
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, width="stretch", hide_index=True)

        # グラフ表示: 選択銘柄と類似銘柄のレーダーチャート重ね表示
        st.subheader("📊 フレーバーチャート比較")
        fig = go.Figure()

        # ターゲットのチャート
        if st.session_state.target_ft:
            fig.add_trace(create_radar_chart(
                st.session_state.target_ft,
                st.session_state.target_name,
                "#ff6b6b",
                "toself",
                0.5
            ))

        # 類似銘柄のチャート（上位5件）
        colors = ["#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#dfe6e9"]
        for i, r in enumerate(display_results[:5]):
            fig.add_trace(create_radar_chart(
                r["ft_values"],
                f"{i+1}. {r['name']}",
                colors[i % len(colors)],
                "none",
                1.0
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig, width="stretch")

        # 各銘柄の詳細
        st.subheader("📋 銘柄詳細")
        for i, r in enumerate(display_results):
            with st.expander(f"第{i+1}位: {r['name']} (類似度: {r['similarity']:.4f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**地域**: {r['areaName']}")
                    st.markdown(f"**蔵元**: {r['breweryName']}")
                with col2:
                    st.markdown(f"**類似度**: {r['similarity']:.4f}")
                    st.markdown(f"**ランキング**: {r['rank_order']}位")

                # 個別レーダーチャート
                fig_individual = go.Figure()
                if st.session_state.target_ft:
                    fig_individual.add_trace(create_radar_chart(
                        st.session_state.target_ft,
                        st.session_state.target_name,
                        "#ff6b6b",
                        "toself",
                        0.3
                    ))
                fig_individual.add_trace(create_radar_chart(
                    r["ft_values"],
                    r["name"],
                    "#4ecdc4",
                    "toself",
                    0.5
                ))
                fig_individual.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=350
                )
                st.plotly_chart(fig_individual, width="stretch")

if __name__ == "__main__":
    main()