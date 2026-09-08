import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from config import settings
from preprocessing.pipeline import preprocess_single_text
from feature_extraction.tfidf import transform_tfidf

# --- Page Configurations ---
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Wisata Garut",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Premium CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
        
        /* Font and Background styling */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Premium Card style */
        .premium-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            border: 1px solid #eef2f6;
            margin-bottom: 20px;
        }
        
        .kpi-card {
            background: linear-gradient(135deg, #ffffff 0%, #f9fbff 100%);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
            border-left: 6px solid #17a2b8;
            border-top: 1px solid #eef2f6;
            border-right: 1px solid #eef2f6;
            border-bottom: 1px solid #eef2f6;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        }
        
        .kpi-value {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .kpi-label {
            font-size: 14px;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Badges for Policy Classifications */
        .badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
        }
        
        .badge-promotional {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #c8e6c9;
        }
        
        .badge-intervention {
            background-color: #ffebee;
            color: #c62828;
            border: 1px solid #ffcdd2;
        }
        
        .badge-monitoring {
            background-color: #fff3e0;
            color: #ef6c00;
            border: 1px solid #ffe0b2;
        }
        
        .badge-insufficient {
            background-color: #f5f5f5;
            color: #616161;
            border: 1px solid #e0e0e0;
        }
        
        /* Custom Header Styling */
        .main-header {
            font-size: 38px;
            font-weight: 700;
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        
        .subheader {
            font-size: 16px;
            color: #64748b;
            margin-bottom: 25px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading (Cached) ---
@st.cache_data
def load_dashboard_data():
    predicted_path = settings.FINAL_DATA_DIR / "predicted_reviews.csv"
    dest_path = settings.FINAL_DATA_DIR / "destination_sentiment_summary.csv"
    cat_path = settings.FINAL_DATA_DIR / "category_sentiment_summary.csv"
    meta_path = settings.FINAL_DATA_DIR / "experiment_metadata.json"
    metrics_path = settings.FINAL_DATA_DIR / "model_metrics.csv"
    exp_comp_path = settings.FINAL_DATA_DIR / "imbalance_experiment_comparison.csv"
    
    df_reviews = pd.read_csv(predicted_path) if predicted_path.exists() else None
    df_dests = pd.read_csv(dest_path) if dest_path.exists() else None
    df_cats = pd.read_csv(cat_path) if cat_path.exists() else None
    
    metadata = {}
    if meta_path.exists():
        import json
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    df_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    df_exp_comp = pd.read_csv(exp_comp_path) if exp_comp_path.exists() else None
    
    return df_reviews, df_dests, df_cats, metadata, df_metrics, df_exp_comp

# --- Inference Resource Loading (Cached) ---
@st.cache_resource
def load_inference_assets():
    model_path = settings.MODELS_DIR / "svm_model.joblib"
    vectorizer_path = settings.MODELS_DIR / "tfidf_vectorizer.joblib"
    
    model = joblib.load(model_path) if model_path.exists() else None
    vectorizer = joblib.load(vectorizer_path) if vectorizer_path.exists() else None
    
    return model, vectorizer

# --- Execution ---
inject_custom_css()
df_reviews, df_dests, df_cats, metadata, df_metrics, df_exp_comp = load_dashboard_data()
model, vectorizer = load_inference_assets()

# Sidebar Setup
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="color: #1e3c72; font-weight: 700; margin-bottom: 0px;">🗺️ Wisata Garut</h2>
    <span style="color: #64748b; font-size: 13px; font-weight: 600;">Sentiment & Recommendation Engine</span>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "PILIH HALAMAN:",
    [
        "📊 Ringkasan & Dashboard",
        "🗺️ Eksplorasi Destinasi",
        "🏢 Analisis Kategori Wisata",
        "🎯 Target & Rekomendasi Kebijakan",
        "🔮 Uji Sentimen Ulasan (Inference)",
        "🛠️ Parameter & Evaluasi Model"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 12px; color: #94a3b8; font-weight: 500;">
    <b>Sistem Pendukung Keputusan</b><br>
    Peta Persepsi Ulasan Google Maps berbasis Support Vector Machine (SVM) Kabupaten Garut.<br><br>
    © 2026 Academic Research Pipeline
</div>
""", unsafe_allow_html=True)

# Main Title block
st.markdown('<div class="main-header">Analisis Sentimen Destinasi Wisata Garut</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Platform Visualisasi Data Mining Ulasan Google Maps & Rekomendasi Kebijakan Pariwisata</div>', unsafe_allow_html=True)

if df_reviews is None or df_dests is None or df_cats is None:
    st.error("Data final hasil pipeline tidak ditemukan. Pastikan Anda sudah menjalankan pipeline (`python main.py`) untuk menghasilkan berkas CSV di folder `data/final`.")
    st.stop()

# Helper untuk normalisasi string sentimen agar kompatibel (Bahasa Indonesia & Inggris)
def normalize_sentiment(val):
    v = str(val).lower()
    if 'pos' in v:
        return 'Positif'
    elif 'neg' in v:
        return 'Negatif'
    return 'Netral'

df_reviews['sentiment_clean'] = df_reviews['predicted_sentiment'].apply(normalize_sentiment)

# --- MENU: Ringkasan & Dashboard ---
if menu == "📊 Ringkasan & Dashboard":
    st.markdown("### 📊 Ringkasan Data & Model")
    
    # KPI Row
    kpi_cols = st.columns(5)
    
    total_reviews = len(df_reviews)
    total_dests = len(df_dests)
    accuracy = metadata.get("overall_accuracy", 0.8879)
    
    pos_reviews = (df_reviews["sentiment_clean"] == "Positif").sum()
    neg_reviews = (df_reviews["sentiment_clean"] == "Negatif").sum()
    
    pos_pct = (pos_reviews / total_reviews) * 100
    neg_pct = (neg_reviews / total_reviews) * 100
    
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #3b82f6;">
            <div class="kpi-value" style="color: #3b82f6;">{total_reviews:,}</div>
            <div class="kpi-label">Total Ulasan</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[1]:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #6366f1;">
            <div class="kpi-value" style="color: #6366f1;">{total_dests}</div>
            <div class="kpi-label">Total Destinasi</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[2]:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #10b981;">
            <div class="kpi-value" style="color: #10b981;">{pos_pct:.1f}%</div>
            <div class="kpi-label">Ulasan Positif</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[3]:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #f43f5e;">
            <div class="kpi-value" style="color: #f43f5e;">{neg_pct:.1f}%</div>
            <div class="kpi-label">Ulasan Negatif</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[4]:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #f59e0b;">
            <div class="kpi-value" style="color: #f59e0b;">{accuracy*100:.2f}%</div>
            <div class="kpi-label">Akurasi Model</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations Row
    vis_cols = st.columns([1, 1])
    
    with vis_cols[0]:
        st.markdown("##### Distribusi Sentimen Ulasan (Keseluruhan)")
        sentiment_counts = df_reviews["sentiment_clean"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentimen", "Jumlah"]
        
        fig_pie = px.pie(
            sentiment_counts, 
            values="Jumlah", 
            names="Sentimen",
            color="Sentimen",
            color_discrete_map={"Positif": "#10b981", "Netral": "#94a3b8", "Negatif": "#f43f5e"},
            hole=0.4,
            template="plotly_white"
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with vis_cols[1]:
        st.markdown("##### Distribusi Rating Bintang vs Prediksi Sentimen")
        rating_sent = df_reviews.groupby(["rating", "sentiment_clean"]).size().reset_index(name="count")
        rating_sent.columns = ["Rating Bintang", "Sentimen", "Jumlah"]
        
        fig_bar = px.bar(
            rating_sent,
            x="Rating Bintang",
            y="Jumlah",
            color="Sentimen",
            color_discrete_map={"Positif": "#10b981", "Netral": "#94a3b8", "Negatif": "#f43f5e"},
            barmode="stack",
            template="plotly_white"
        )
        fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, legend_title="Sentimen")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    st.markdown("##### Destinasi dengan Ulasan Terbanyak")
    top_dests = df_dests.sort_values(by="total_reviews", ascending=False).head(10)
    
    fig_top = px.bar(
        top_dests,
        y="destination_name",
        x="total_reviews",
        orientation="h",
        color="average_rating",
        color_continuous_scale="Viridis",
        labels={"destination_name": "Nama Destinasi", "total_reviews": "Jumlah Ulasan", "average_rating": "Rating Rata-rata"},
        template="plotly_white"
    )
    fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig_top, use_container_width=True)

# --- MENU: Eksplorasi Destinasi ---
elif menu == "🗺️ Eksplorasi Destinasi":
    st.markdown("### 🗺️ Eksplorasi Sentimen Destinasi Wisata")
    
    dest_list = sorted(df_dests["destination_name"].unique())
    selected_dest = st.selectbox("Pilih Destinasi Wisata:", dest_list)
    
    dest_row = df_dests[df_dests["destination_name"] == selected_dest].iloc[0]
    
    detail_cols = st.columns([1, 1, 1])
    
    with detail_cols[0]:
        st.markdown("##### Informasi Destinasi")
        st.markdown(f"**Nama:** {dest_row['destination_name']}")
        st.markdown(f"**Kategori:** {dest_row['category']}")
        st.markdown(f"**Rating Google Maps:** ⭐ {dest_row['average_rating']:.1f}")
        st.markdown(f"**Jumlah Ulasan Teranalisis:** {dest_row['total_reviews']} ulasan")
        
        p_class = dest_row['policy_class']
        badge_class = "badge-insufficient"
        if p_class == "Promotional Priority":
            badge_class = "badge-promotional"
        elif p_class == "Intervention Priority":
            badge_class = "badge-intervention"
        elif p_class == "Monitoring / Improvement Priority":
            badge_class = "badge-monitoring"
            
        st.markdown(f"""
        **Rekomendasi Kebijakan:** <br>
        <span class="badge {badge_class}">{p_class}</span>
        """, unsafe_allow_html=True)
        
    with detail_cols[1]:
        st.markdown("##### Distribusi Sentimen Destinasi")
        dest_sent_data = pd.DataFrame({
            "Sentimen": ["Positif", "Netral", "Negatif"],
            "Jumlah": [dest_row["positive_count"], dest_row["neutral_count"], dest_row["negative_count"]]
        })
        fig_dest_pie = px.pie(
            dest_sent_data,
            values="Jumlah",
            names="Sentimen",
            color="Sentimen",
            color_discrete_map={"Positif": "#10b981", "Netral": "#94a3b8", "Negatif": "#f43f5e"},
            hole=0.3,
            template="plotly_white"
        )
        fig_dest_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
        st.plotly_chart(fig_dest_pie, use_container_width=True)
        
    with detail_cols[2]:
        st.markdown("##### Panduan Aksi Kebijakan")
        if p_class == "Promotional Priority":
            st.success("🎯 **Destinasi Unggulan!** Promosikan tempat ini secara masif di media sosial, brosur wisata, dan kampanye dinas pariwisata untuk menarik wisatawan baru.")
        elif p_class == "Intervention Priority":
            st.error("⚠️ **Prioritas Intervensi Lapangan!** Tinjau langsung destinasi ini. Cari tahu keluhan spesifik wisatawan (kebersihan, harga tiket, layanan, dll.) dan lakukan perbaikan segera.")
        elif p_class == "Monitoring / Improvement Priority":
            st.warning("📊 **Pantau & Tingkatkan!** Pengunjung cenderung bersikap netral atau moderat. Lakukan renovasi sarana, berikan edukasi layanan kepada pengelola, dan tingkatkan daya tarik.")
        else:
            st.info("ℹ️ **Data Ulasan Kurang!** Ulasan tidak mencukupi untuk klasifikasi kebijakan (< 10 ulasan). Rekomendasikan pengelola mengajak pengunjung menulis ulasan digital.")
            
    st.markdown("---")
    
    st.markdown("##### 💬 Penjelajah Ulasan Pengunjung")
    dest_reviews = df_reviews[df_reviews["destination_name"] == selected_dest].copy()
    
    filter_sent = st.radio(
        "Saring berdasarkan sentimen ulasan:",
        ["Semua", "Positif", "Netral", "Negatif"],
        horizontal=True
    )
    
    if filter_sent != "Semua":
        dest_reviews = dest_reviews[dest_reviews["sentiment_clean"] == filter_sent]
        
    st.markdown(f"Menampilkan **{len(dest_reviews)}** ulasan:")
    
    limit_revs = st.slider("Jumlah ulasan yang ditampilkan:", 5, min(100, max(5, len(dest_reviews))), 15)
    
    if len(dest_reviews) == 0:
        st.info("Tidak ada ulasan yang sesuai dengan filter sentimen.")
    else:
        for idx, row in dest_reviews.head(limit_revs).iterrows():
            stars = "⭐" * int(float(row["rating"]))
            sent_badge = ""
            s_val = row["sentiment_clean"]
            if s_val == "Positif":
                sent_badge = '<span class="badge badge-promotional">Positif</span>'
            elif s_val == "Netral":
                sent_badge = '<span class="badge badge-monitoring">Netral</span>'
            elif s_val == "Negatif":
                sent_badge = '<span class="badge badge-intervention">Negatif</span>'
                
            author_info = f"**{row['author']}** ({row['review_date']}) &nbsp;&nbsp;&nbsp; {stars} &nbsp;&nbsp;&nbsp; {sent_badge}"
            
            with st.chat_message("user"):
                st.markdown(author_info, unsafe_allow_html=True)
                st.markdown(f"*\"{row['review_text']}\"*")

# --- MENU: Analisis Kategori Wisata ---
elif menu == "🏢 Analisis Kategori Wisata":
    st.markdown("### 🏢 Analisis Berdasarkan Kategori Wisata")
    
    st.markdown("##### Tabel Ringkasan Sentimen per Kategori Wisata")
    
    df_cats_show = df_cats.copy()
    df_cats_show.columns = [
        "Kategori Wisata", "Jumlah Destinasi", "Total Ulasan", 
        "Persentase Positif (%)", "Persentase Netral (%)", "Persentase Negatif (%)", 
        "Rating Rata-rata", "Dest. Positif Rerata (%)", "Dest. Netral Rerata (%)", "Dest. Negatif Rerata (%)"
    ]
    
    st.dataframe(df_cats_show, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    vis_cat_cols = st.columns(2)
    df_cats_filtered = df_cats[df_cats["total_reviews"] >= 10].sort_values(by="total_reviews", ascending=False).head(15)
    
    with vis_cat_cols[0]:
        st.markdown("##### Kategori dengan Persentase Sentimen Positif Tertinggi (Min. 10 Ulasan)")
        fig_cat_pos = px.bar(
            df_cats_filtered.sort_values(by="positive_percentage", ascending=True),
            y="category",
            x="positive_percentage",
            orientation="h",
            color="positive_percentage",
            color_continuous_scale="Greens",
            labels={"category": "Kategori Wisata", "positive_percentage": "Positif (%)"},
            template="plotly_white"
        )
        fig_cat_pos.update_layout(height=400, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cat_pos, use_container_width=True)
        
    with vis_cat_cols[1]:
        st.markdown("##### Kategori dengan Jumlah Ulasan Terbanyak")
        fig_cat_rev = px.bar(
            df_cats_filtered,
            y="category",
            x="total_reviews",
            orientation="h",
            color="total_reviews",
            color_continuous_scale="Blues",
            labels={"category": "Kategori Wisata", "total_reviews": "Jumlah Ulasan"},
            template="plotly_white"
        )
        fig_cat_rev.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cat_rev, use_container_width=True)

# --- MENU: Target & Rekomendasi Kebijakan ---
elif menu == "🎯 Target & Rekomendasi Kebijakan":
    st.markdown("### 🎯 Rekomendasi Kebijakan & Target Prioritas")
    
    st.markdown("""
    Sistem mengelompokkan destinasi wisata ke dalam **4 Kategori Kebijakan** yang bersifat saling lepas (*mutually exclusive*):
    1. **Promotional Priority**: Destinasi dengan kepuasan publik tinggi (Ulasan Positif ≥ 70%, Rating Rata-rata ≥ 4.0, Total Ulasan ≥ 10). Direkomendasikan untuk promosi besar-besaran.
    2. **Intervention Priority**: Destinasi dengan tingkat ketidakpuasan publik tinggi (Ulasan Negatif ≥ 15%, Total Ulasan ≥ 10). Direkomendasikan untuk tinjauan dan intervensi operasional lapangan.
    3. **Monitoring / Improvement Priority**: Destinasi dengan persepsi moderat (tidak masuk kriteria Promosi maupun Intervensi, Total Ulasan ≥ 10).
    4. **Insufficient Evidence**: Destinasi dengan data ulasan digital yang belum mencukupi (< 10 ulasan).
    """)
    
    policy_tabs = st.tabs([
        "🔴 Prioritas Intervensi (Intervention)",
        "🟢 Prioritas Promosi (Promotional)",
        "🟡 Pemantauan (Monitoring)",
        "⚪ Bukti Kurang (Insufficient)"
    ])
    
    display_cols = ["destination_name", "category", "total_reviews", "positive_percentage", "neutral_percentage", "negative_percentage", "average_rating"]
    rename_dict = {
        "destination_name": "Nama Destinasi",
        "category": "Kategori",
        "total_reviews": "Total Ulasan",
        "positive_percentage": "Positif (%)",
        "neutral_percentage": "Netral (%)",
        "negative_percentage": "Negatif (%)",
        "average_rating": "Avg Rating"
    }
    
    with policy_tabs[0]:
        st.error("⚠️ **TARGET PRIORITAS INTERVENSI (TINJAUAN SEGERA)**")
        df_intervene = df_dests[df_dests["policy_class"] == "Intervention Priority"].sort_values(by="negative_percentage", ascending=False)
        st.markdown(f"Terdapat **{len(df_intervene)}** destinasi dalam kategori ini:")
        st.dataframe(df_intervene[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
        
        if len(df_intervene) > 0:
            st.markdown("##### 📍 Catatan Analisis Kebijakan Kritis:")
            for i, (_, row) in enumerate(df_intervene.head(3).iterrows()):
                st.markdown(f"""
                - **{row['destination_name']}** ({row['category']}): Memiliki akumulasi sentimen negatif sebesar **{row['negative_percentage']}%** dari total **{row['total_reviews']}** ulasan. 
                  *Rekomendasi:* Dinas Pariwisata berkoordinasi dengan pengelola untuk meninjau mutu pelayanan, harga tiket masuk, dan kebersihan fasilitas fisik di lokasi.
                """)
                
    with policy_tabs[1]:
        st.success("🎯 **TARGET PRIORITAS PROMOSI (DESTINASI UNGGULAN)**")
        df_promo = df_dests[df_dests["policy_class"] == "Promotional Priority"].sort_values(by="positive_percentage", ascending=False)
        st.markdown(f"Terdapat **{len(df_promo)}** destinasi dalam kategori ini:")
        st.dataframe(df_promo[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
        
        if len(df_promo) > 0:
            st.markdown("##### 📍 Catatan Analisis Kebijakan Promosi:")
            for i, (_, row) in enumerate(df_promo.head(3).iterrows()):
                st.markdown(f"""
                - **{row['destination_name']}** ({row['category']}): Memiliki akumulasi sentimen positif luar biasa sebesar **{row['positive_percentage']}%** dari total **{row['total_reviews']}** ulasan.
                  *Rekomendasi:* Masukkan dalam program promosi tahunan pariwisata daerah (*branding* pariwisata unggulan Garut).
                """)
                
    with policy_tabs[2]:
        st.warning("📊 **TARGET PEMANTAUAN & PENINGKATAN LAYANAN**")
        df_monitor = df_dests[df_dests["policy_class"] == "Monitoring / Improvement Priority"].sort_values(by="neutral_percentage", ascending=False)
        st.markdown(f"Terdapat **{len(df_monitor)}** destinasi dalam kategori ini:")
        st.dataframe(df_monitor[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
        
    with policy_tabs[3]:
        st.info("ℹ️ **DATA ULASAN KURANG (EVALUASI SCRAPING ULANG)**")
        df_insufficient = df_dests[df_dests["policy_class"] == "Insufficient Evidence"].sort_values(by="total_reviews", ascending=False)
        st.markdown(f"Terdapat **{len(df_insufficient)}** destinasi dalam kategori ini:")
        st.dataframe(df_insufficient[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)

# --- MENU: Uji Sentimen Ulasan (Inference) ---
elif menu == "🔮 Uji Sentimen Ulasan (Inference)":
    st.markdown("### 🔮 Uji Sentimen Ulasan Mandiri (Real-time Inference)")
    st.markdown("""
    Gunakan modul inferensi ini untuk menguji prediksi model SVM secara langsung. 
    Masukkan teks ulasan pariwisata berbahasa Indonesia, lalu sistem akan menjalankan preprocessing teks (pembersihan, stopword removal, dan stemming Bahasa Indonesia) serta memprediksi sentimen ulasan secara real-time.
    """)
    
    if model is None or vectorizer is None:
        st.error("Model SVM atau Vectorizer TF-IDF tidak ditemukan di folder `models/`. Harap pastikan model biner telah dilatih.")
    else:
        examples = [
            "Tempatnya sangat indah, udaranya sejuk banget dan pemandangannya memukau. Pelayanannya ramah!",
            "Pelayanannya lambat sekali, makanan dingin, dan toiletnya kotor. Kecewa banget kemari.",
            "Tempat wisatanya biasa saja, parkirannya luas tapi harga tiket agak mahal.",
            "Akses jalan menuju lokasi rusak parah dan berbahaya."
        ]
        
        st.markdown("**Contoh Ulasan untuk Dicoba:**")
        cols_ex = st.columns(len(examples))
        selected_example = None
        for i, ex in enumerate(examples):
            with cols_ex[i]:
                if st.button(f"Contoh {i+1}", help=ex):
                    selected_example = ex
                    
        input_text = st.text_area(
            "Masukkan teks ulasan pengunjung di sini:",
            value=selected_example if selected_example else "",
            height=120,
            placeholder="Tulis ulasan Anda..."
        )
        
        if st.button("Analisis Sentimen", type="primary"):
            if not input_text.strip():
                st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
            else:
                with st.spinner("Menjalankan Preprocessing & SVM Klasifikasi..."):
                    cleaned_text = preprocess_single_text(input_text)
                    X_tfidf = transform_tfidf([cleaned_text], vectorizer)
                    
                    pred_label = model.predict(X_tfidf)[0]
                    
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_tfidf)[0]
                    else:
                        decision_scores = model.decision_function(X_tfidf)[0]
                        exp_scores = np.exp(decision_scores - np.max(decision_scores))
                        probs = exp_scores / np.sum(exp_scores)
                    
                    # ✅ FIX PERBAIKAN MAPPING: 0=Positif, 1=Netral, 2=Negatif (Sesuai Pipeline Notebook)
                    sentiment_map = {0: "Positif", 1: "Netral", 2: "Negatif"}
                    pred_sentiment = sentiment_map[pred_label]
                    
                    st.markdown("---")
                    st.markdown("##### Hasil Analisis Sentimen")
                    
                    banner_color = "#e8f5e9"
                    text_color = "#2e7d32"
                    emoji = "😊"
                    label_id = "POSITIF (Positive)"
                    
                    if pred_sentiment == "Netral":
                        banner_color = "#fff3e0"
                        text_color = "#ef6c00"
                        emoji = "😐"
                        label_id = "NETRAL (Neutral)"
                    elif pred_sentiment == "Negatif":
                        banner_color = "#ffebee"
                        text_color = "#c62828"
                        emoji = "😡"
                        label_id = "NEGATIF (Negative)"
                        
                    st.markdown(f"""
                    <div style="background-color: {banner_color}; color: {text_color}; padding: 20px; border-radius: 12px; border: 1px solid {text_color}; text-align: center;">
                        <span style="font-size: 40px;">{emoji}</span>
                        <h3 style="margin: 10px 0px 5px 0px; font-weight: 700;">Sentimen Terprediksi: {label_id}</h3>
                        <p style="margin: 0; font-size: 14px; font-weight: 600;">Model Linear SVM mengklasifikasikan ulasan ini sebagai sentimen {pred_sentiment}.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("##### Estimasi Tingkat Keyakinan Terkalibrasi (Calibrated Probabilities):")
                    
                    prob_cols = st.columns(3)
                    with prob_cols[0]:
                        st.write(f"😊 **Positif (0):** {probs[0]*100:.2f}%")
                        st.progress(float(probs[0]))
                    with prob_cols[1]:
                        st.write(f"😐 **Netral (1):** {probs[1]*100:.2f}%")
                        st.progress(float(probs[1]))
                    with prob_cols[2]:
                        st.write(f"😡 **Negatif (2):** {probs[2]*100:.2f}%")
                        st.progress(float(probs[2]))
                        
                    st.markdown("---")
                    st.markdown("##### Detail Preprocessing Teks (NLP Pipeline)")
                    st.markdown(f"**Teks Asli:** *\"{input_text}\"*")
                    st.markdown(f"**Teks Hasil Preprocessing:** `{cleaned_text}`")

# --- MENU: Parameter & Evaluasi Model ---
elif menu == "🛠️ Parameter & Evaluasi Model":
    st.markdown("### 🛠️ Parameter Eksperimen & Evaluasi Model SVM")
    
    # Sub-tabs for Evaluasi
    eval_tabs = st.tabs([
        "🧪 Komparasi 5 Skenario Imbalance",
        "📊 Performa Model Terpilih",
        "🎯 Metadata Eksperimen"
    ])
    
    with eval_tabs[0]:
        st.markdown("##### 🧪 Studi Komparasi 5 Skenario Penanganan Class Imbalance")
        st.markdown("""
        Untuk membuktikan keterandalan model dan terhindar dari bias kelas mayoritas, dilakukan pengujian komparatif terhadap **5 Skenario Penanganan Class Imbalance** secara terisolasi pada Training Set (80%):
        1. **Baseline**: Tanpa teknik balancing.
        2. **Class Weight Balanced**: Pembobotan algoritma (`class_weight='balanced'`).
        3. **Random Oversampling (ROS)**: Memperbanyak sampel kelas minoritas dari data asli.
        4. **Random UnderSampling (RUS)**: Mengurangi sampel dari kelas mayoritas.
        5. **SMOTE**: Sintesis sampel baru untuk kelas minoritas.
        """)
        
        if df_exp_comp is not None:
            st.dataframe(df_exp_comp, use_container_width=True, hide_index=True)
        else:
            st.info("File komparasi `imbalance_experiment_comparison.csv` belum ditemukan. Jalankan pipeline `main.py` terlebih dahulu.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grid Confusion Matrix 5 Skenario
        st.markdown("##### Visualisasi Confusion Matrix Komparatif (5 Skenario)")
        all_cm_path = settings.FINAL_DATA_DIR / "all_scenarios_confusion_matrices.png"
        if all_cm_path.exists():
            st.image(str(all_cm_path), caption="Perbandingan Confusion Matrix dari Kelima Skenario Handling Imbalance pada Test Set (3.585 Data Uji)", use_container_width=True)
        else:
            st.info("Gambar grid Confusion Matrix 5 skenario belum tersedia.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Bar Chart Macro F1 vs Balanced Acc
        st.markdown("##### Grafik Perbandingan Macro F1-Score & Balanced Accuracy")
        exp_img_path = settings.FINAL_DATA_DIR / "imbalance_experiments_comparison.png"
        if exp_img_path.exists():
            st.image(str(exp_img_path), caption="Grafik Komparasi Performa 5 Skenario Imbalance (Evaluasi pada Test Set Murni)", use_container_width=True)
        else:
            st.info("Grafik komparasi belum tersedia.")
            
    with eval_tabs[1]:
        st.markdown("##### Laporan Klasifikasi Rinci Model Terpilih (Classification Report)")
        if df_metrics is not None:
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("Laporan klasifikasi rinci tidak ditemukan.")
            
        st.markdown("---")
        st.markdown("##### Confusion Matrix Model Terpilih (Class Weight Balanced)")
        cm_path = settings.FINAL_DATA_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix Model SVM Terpilih pada Test Set Murni (3.585 data split)", use_container_width=True)
        else:
            st.info("Gambar Confusion Matrix tidak ditemukan.")
            
    with eval_tabs[2]:
        meta_cols = st.columns(2)
        with meta_cols[0]:
            st.markdown("##### Parameter Model & Dataset")
            st.markdown(f"**Tanggal Eksperimen:** {metadata.get('experiment_date', '2026-08-15')}")
            st.markdown(f"**Ukuran Dataset Keseluruhan:** {metadata.get('dataset_size', 17923):,} ulasan")
            st.markdown(f"**Data Pelatihan (Train Set):** {metadata.get('train_size', 14338):,} ulasan (80%)")
            st.markdown(f"**Data Pengujian (Test Set):** {metadata.get('test_size', 3585):,} ulasan (20%)")
            st.markdown(f"**Bobot Kelas (Class Weights):** {metadata.get('class_weights', 'balanced')}")
            st.markdown(f"**Kernel SVM:** {metadata.get('svm_parameters', {}).get('kernel', 'linear')}")
            st.markdown(f"**Parameter Regularisasi SVM (C):** {metadata.get('svm_parameters', {}).get('C', 0.2)}")
            st.markdown(f"**Maksimum Fitur TF-IDF:** {metadata.get('tfidf_parameters', {}).get('max_features', 10000)}")
            
        with meta_cols[1]:
            st.markdown("##### Kinerja Utama Model SVM (pada Test Set Murni)")
            st.markdown(f"**Akurasi Model Keseluruhan:** {metadata.get('overall_accuracy', 0.8879)*100:.2f}%")
            st.markdown(f"**Balanced Accuracy:** {metadata.get('balanced_accuracy', 0.6211)*100:.2f}%")
            st.markdown(f"**Macro F1-Score:** {metadata.get('macro_f1', 0.6356):.4f}")
            st.markdown(f"**Weighted F1-Score:** {metadata.get('weighted_f1', 0.8883):.4f}")
            st.markdown(f"**Akurasi Majority Baseline:** {metadata.get('baseline_accuracy', 0.8678)*100:.2f}%")
            st.markdown(f"**Macro F1 Majority Baseline:** {metadata.get('baseline_macro_f1', 0.3097):.4f}")