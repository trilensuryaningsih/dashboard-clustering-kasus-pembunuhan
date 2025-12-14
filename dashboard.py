# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║              DASHBOARD CLUSTERING KASUS PEMBUNUHAN                        ║
# ║                      Analisis Korban dengan K-Means                       ║
# ║                                                                           ║
# ║  Dibuat oleh:                                                             ║
# ║  - Trillen Surya Ningsih (2311522004)                                     ║
# ║  - Zakky Aulia Aldrin (2311522018)                                        ║
# ║  - Dimas Radithya Nurizkitha (2311523026)                                 ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                      SECTION 1: IMPORT LIBRARIES                          ║
# ║                                                                           ║
# ║  Library yang digunakan:                                                  ║
# ║  - Streamlit: Framework dashboard                                         ║
# ║  - Pandas, NumPy: Manipulasi data                                        ║
# ║  - Scikit-learn: K-Means, PCA, t-SNE, StandardScaler                     ║
# ║  - UMAP: Dimensionality reduction (opsional)                              ║
# ║  - Plotly: Visualisasi interaktif                                         ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

# ML / DR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# optional UMAP
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# Viz
import plotly.express as px

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    SECTION 2: CONFIG & CONSTANTS                          ║
# ║                                                                           ║
# ║  Konfigurasi aplikasi:                                                    ║
# ║  - DEFAULT_CSV: Path default untuk file dataset                          ║
# ║  - ACCENT: Warna aksen tema (tidak digunakan aktif)                       ║
# ║  - Page config: Judul dan layout dashboard                                ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
DEFAULT_CSV = r"C:\praktikum\homicide-data.csv"
ACCENT = "#6C5CE7"

st.set_page_config(page_title="Clustering Kasus Pembunuhan", layout="wide")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                   SECTION 3: SPLASH SCREEN / OPENING PAGE                 ║
# ║                                                                           ║
# ║  Halaman pembuka dengan animasi loading:                                  ║
# ║  - Ikon tengkorak dengan efek pulse                                       ║
# ║  - Judul dan subtitle dengan efek fade-in                                 ║
# ║  - Spinner loading                                                        ║
# ║  - Durasi tampil: 3 detik                                                 ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    # Hide default Streamlit elements during splash
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a0a0a 25%, #0f0f0f 50%, #1a0808 75%, #0a0a0a 100%);
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .splash-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: transparent;
        }
        
        .splash-skull {
            animation: pulse 1.5s ease-in-out infinite;
            filter: drop-shadow(0 0 30px rgba(239, 68, 68, 0.8));
        }
        
        .splash-title {
            color: #ef4444;
            font-size: 2.5rem;
            font-weight: 700;
            margin-top: 30px;
            text-align: center;
            animation: fadeIn 1s ease-out;
            text-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
        }
        
        .splash-subtitle {
            color: #9ca3af;
            font-size: 1.2rem;
            margin-top: 10px;
            text-align: center;
            animation: fadeIn 1s ease-out 0.3s both;
        }
        
        .splash-loader {
            margin-top: 40px;
            width: 50px;
            height: 50px;
            border: 3px solid rgba(239, 68, 68, 0.2);
            border-top: 3px solid #ef4444;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        .splash-loading-text {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 20px;
            animation: fadeIn 1s ease-out 0.6s both;
        }
        
        .splash-welcome {
            color: #ffffff;
            font-size: 1.3rem;
            font-weight: 300;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 20px;
            animation: fadeIn 0.8s ease-out;
        }
    </style>
    
    <div class="splash-container">
        <p class="splash-welcome">Dashboard Analisis Kriminal</p>
        <svg class="splash-skull" xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="9" cy="12" r="1"></circle>
            <circle cx="15" cy="12" r="1"></circle>
            <path d="M8 20v2h8v-2"></path>
            <path d="M12.5 17l-.5-1-.5 1h1z"></path>
            <path d="M16 20a2 2 0 0 0 1.56-3.25 8 8 0 1 0-11.12 0A2 2 0 0 0 8 20"></path>
        </svg>
        <h1 class="splash-title">Clustering Kasus Pembunuhan</h1>
        <p class="splash-subtitle">Mengidentifikasi Pola Korban untuk Pencegahan Kejahatan</p>
        <div class="splash-loader"></div>
        <p class="splash-loading-text">Menyiapkan analisis data...</p>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(3)
    st.session_state.splash_shown = True
    st.rerun()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    SECTION 4: CSS STYLING (MAIN THEME)                    ║
# ║                                                                           ║
# ║  Styling untuk seluruh aplikasi:                                         ║
# ║  - Sidebar: Background gelap, border merah, menu items                   ║
# ║  - Main content: Background gelap, cards, metric cards                   ║
# ║  - Typography: Warna teks, heading                                       ║
# ║  - Components: Buttons, info boxes, recommendations                      ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
st.markdown(
    """
    <style>
    /* ============================================
       SIDEBAR STYLING - Clean Modern Dark Theme
       ============================================ */
    
    /* Sidebar container - clean dark with border only */
    [data-testid="stSidebar"] {
        background: linear-gradient(165deg, #0d0d0d 0%, #121212 25%, #0f0f0f 50%, #111111 75%, #0a0a0a 100%);
        border: 2px solid rgba(220, 38, 38, 0.4);
        border-left: none;
        border-radius: 0 20px 20px 0;
        margin: 10px 0 10px 0;
        overflow: hidden;
    }
    
    /* Remove top accent line */
    [data-testid="stSidebar"]::before {
        display: none;
    }
    
    /* Sidebar inner content - prevent scrolling */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        overflow: hidden !important;
        height: 100vh !important;
    }
    
    /* Hide scrollbar on sidebar */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        overflow: hidden !important;
    }
    
    /* Custom sidebar header with SVG */
    .sidebar-header {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 6px;
        padding: 0.75rem 0.75rem;
        margin: 0 0.5rem 0.75rem 0.5rem;
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.08) 0%, transparent 100%);
        border: none;
        border-radius: 10px;
        color: #ffffff;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        position: relative;
        text-align: center;
    }
    
    .sidebar-header::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 1rem;
        right: 1rem;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(220, 38, 38, 0.4), transparent);
    }
    
    .sidebar-header svg {
        stroke: #ef4444;
        flex-shrink: 0;
        filter: drop-shadow(0 0 6px rgba(239, 68, 68, 0.4));
    }
    
    /* Hide default sidebar title */
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        display: none;
    }
    
    /* Radio button container */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 2px;
        padding: 0 0.5rem;
    }
    
    /* Hide default radio label */
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    
    /* Radio options styling - NO BORDERS except active */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: transparent;
        color: rgba(255, 255, 255, 0.65) !important;
        padding: 0.5rem 1rem;
        margin: 0;
        border-radius: 8px;
        border: 1px solid transparent;
        transition: all 0.25s ease;
        cursor: pointer;
        font-weight: 500;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        position: relative;
        overflow: hidden;
        height: 44px;
        min-height: 44px;
        max-height: 44px;
        box-sizing: border-box;
        width: 100% !important;
    }
    
    /* Hover effect - subtle glow, no border */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(30, 30, 30, 0.5) 100%);
        color: #ffffff !important;
        transform: translateX(6px);
        border: 1px solid transparent;
    }
    
    /* Selected/Active state - WITH BORDER */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(185, 28, 28, 0.15) 100%);
        color: #ffffff !important;
        border: 1px solid rgba(220, 38, 38, 0.6);
        box-shadow: 0 0 20px rgba(220, 38, 38, 0.2), inset 0 0 20px rgba(220, 38, 38, 0.05);
        font-weight: 600;
    }
    
    /* Active state left accent bar */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"]::before,
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked)::before {
        content: "";
        position: absolute;
        left: 0;
        top: 15%;
        bottom: 15%;
        width: 3px;
        background: linear-gradient(180deg, #ef4444, #dc2626);
        border-radius: 0 3px 3px 0;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.6);
    }
    
    /* Hide default radio circle */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Sidebar dividers - subtle */
    [data-testid="stSidebar"] hr {
        border-color: rgba(220, 38, 38, 0.15);
        margin: 1.5rem 0.75rem;
    }
    
    /* Sidebar expander - clean style */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(20, 20, 20, 0.6);
        color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 10px;
        border: none;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(220, 38, 38, 0.1);
    }
    
    /* ============================================
       MAIN CONTENT STYLING - Dark Theme
       ============================================ */
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
    }
    
    /* Main content text color */
    .stApp, .stApp p, .stApp span, .stApp li {
        color: #e5e5e5;
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff;
    }
    
    .header-title {
        color: #ef4444;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Dashboard Cards */
    .card {
        background: linear-gradient(135deg, rgba(20, 20, 20, 0.95) 0%, rgba(30, 30, 30, 0.9) 100%);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(220, 38, 38, 0.2);
        margin-bottom: 20px;
    }
    
    /* Dashboard Section Header with SVG */
    .dashboard-section {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(220, 38, 38, 0.3);
    }
    
    .dashboard-section .section-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(185, 28, 28, 0.15) 100%);
        border-radius: 12px;
        border: 1px solid rgba(220, 38, 38, 0.4);
    }
    
    .dashboard-section .section-icon svg {
        stroke: #ef4444;
        filter: drop-shadow(0 0 6px rgba(239, 68, 68, 0.4));
    }
    
    .dashboard-section .section-title {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Content text styling */
    .dashboard-content {
        color: #d1d5db;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .dashboard-content strong {
        color: #ef4444;
    }
    
    /* Bullet list styling */
    .bullet-list {
        list-style: none;
        padding: 0;
        margin: 16px 0;
    }
    
    .bullet-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        background: rgba(220, 38, 38, 0.08);
        border-radius: 10px;
        border-left: 3px solid #dc2626;
        transition: all 0.2s ease;
    }
    
    .bullet-item:hover {
        background: rgba(220, 38, 38, 0.12);
        transform: translateX(4px);
    }
    
    .bullet-item svg {
        stroke: #ef4444;
        flex-shrink: 0;
        margin-top: 2px;
    }
    
    .bullet-item span {
        color: #e5e5e5;
    }
    
    .bullet-item strong {
        color: #ffffff;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(25, 25, 25, 0.95) 0%, rgba(35, 35, 35, 0.9) 100%);
        padding: 20px 24px;
        border-radius: 14px;
        border: 1px solid rgba(220, 38, 38, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    
    .metric-card .metric-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        margin: 0 auto 12px auto;
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%);
        border-radius: 12px;
        border: 1px solid rgba(220, 38, 38, 0.3);
    }
    
    .metric-card .metric-icon svg {
        stroke: #ef4444;
        filter: drop-shadow(0 0 8px rgba(239, 68, 68, 0.5));
    }
    
    .metric-card .metric-label {
        color: #9ca3af;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 6px;
    }
    
    .metric-card .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Info box */
    .info-box {
        display: flex;
        align-items: center;
        gap: 14px;
        background: rgba(59, 130, 246, 0.1);
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin: 16px 0;
    }
    
    .info-box svg {
        stroke: #60a5fa;
        flex-shrink: 0;
    }
    
    .info-box span {
        color: #bfdbfe;
    }
    
    /* Streamlit overrides for dark theme */
    [data-testid="stDataFrame"] {
        background: rgba(20, 20, 20, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(25, 25, 25, 0.95) 0%, rgba(35, 35, 35, 0.9) 100%);
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(220, 38, 38, 0.3);
    }
    
    [data-testid="stMetric"] label {
        color: #9ca3af !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    .bg-info {
        background: rgba(59, 130, 246, 0.1);
        padding: 14px 18px;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 14px 0;
        color: #bfdbfe;
    }
    
    .recommendation {
        background: rgba(245, 158, 11, 0.1);
        padding: 14px 18px;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        margin: 10px 0;
        color: #fde68a;
    }
    
    .insight {
        background: rgba(16, 185, 129, 0.1);
        padding: 14px 18px;
        border-left: 4px solid #10b981;
        border-radius: 10px;
        margin: 10px 0;
        color: #a7f3d0;
    }
    
    .muted { 
        color: #9ca3af; 
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(220, 38, 38, 0.4), transparent);
        margin: 30px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 10px;"><div style="display: flex; align-items: center; justify-content: center; width: 56px; height: 56px; background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(185, 28, 28, 0.15) 100%); border-radius: 14px; border: 1px solid rgba(220, 38, 38, 0.4);"><svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 8px rgba(239, 68, 68, 0.5));"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg></div><div><h1 style="color: #ef4444; font-weight: 700; font-size: 2.2rem; margin: 0;">Clustering Kasus Pembunuhan — Analisis Korban</h1><p style="color: #9ca3af; margin: 4px 0 0 0; font-size: 1rem;">Dashboard interaktif untuk mengelompokkan kasus menurut karakteristik korban. Ikuti alur di sidebar.</p></div></div>""", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    SECTION 5: SESSION STATE INIT                          ║
# ║                                                                           ║
# ║  Inisialisasi state aplikasi untuk menyimpan:                            ║
# ║  - df_raw: Dataset asli dari CSV                                         ║
# ║  - df_cleaned: Dataset setelah cleaning                                  ║
# ║  - df_proc: Dataset setelah processing dengan cluster labels             ║
# ║  - X_scaled: Data yang sudah di-scale untuk clustering                   ║
# ║  - feature_cols: Kolom fitur yang digunakan                              ║
# ║  - cluster_labels: Hasil label clustering                                ║
# ║  - final_k, suggested_k: Nilai K untuk clustering                        ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if "df_raw" not in st.session_state:
    st.session_state.df_raw = pd.DataFrame()
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = pd.DataFrame()
if "df_proc" not in st.session_state:
    st.session_state.df_proc = pd.DataFrame()
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "cluster_labels" not in st.session_state:
    st.session_state.cluster_labels = None
if "final_k" not in st.session_state:
    st.session_state.final_k = None
if "silhouette_score_val" not in st.session_state:
    st.session_state.silhouette_score_val = None
if "suggested_k" not in st.session_state:
    st.session_state.suggested_k = None
if "suggested_method" not in st.session_state:
    st.session_state.suggested_method = None

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                     SECTION 6: UTILITY FUNCTIONS                          ║
# ║                                                                           ║
# ║  Fungsi-fungsi helper:                                                   ║
# ║  - robust_read_csv(): Membaca CSV dengan berbagai encoding               ║
# ║  - load_csv(): Wrapper cached untuk baca CSV                             ║
# ║  - detect_data_quality_issues(): Deteksi nilai kosong                    ║
# ║  - recommend_cleaning(): Rekomendasi pembersihan data                    ║
# ║  - preprocess_with_options(): Preprocessing dengan opsi cleaning         ║
# ║  - compute_k_metrics(): Hitung Elbow & Silhouette                        ║
# ║  - suggest_k(): Saran K optimal                                          ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def robust_read_csv(path_or_buffer, try_encodings=None):
    if try_encodings is None:
        try_encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"]
    if isinstance(path_or_buffer, (str, Path)):
        for enc in try_encodings:
            try:
                return pd.read_csv(path_or_buffer, encoding=enc)
            except Exception:
                pass
        with open(path_or_buffer, "rb") as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            return pd.read_csv(text)
    else:
        data = path_or_buffer.read()
        for enc in try_encodings:
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except Exception:
                pass
        txt = data.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(txt))

@st.cache_data
def load_csv(path):
    try:
        return robust_read_csv(path)
    except Exception:
        return pd.DataFrame()

def detect_data_quality_issues(df):
    issues = {}
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            issues[col] = {"type": "missing", "count": missing, "pct": 100*missing/len(df)}
    return issues

def recommend_cleaning(df):
    recs = []
    for col in df.columns:
        if col in ["victim_age"]:
            try:
                numeric = pd.to_numeric(df[col].replace({"kosong": np.nan, "": np.nan}), errors="coerce")
                if numeric.isna().sum() > 0:
                    recs.append(f"📌 **{col}**: {numeric.isna().sum()} nilai kosong. Rekomendasi: isi dengan median atau hapus baris.")
            except:
                pass
    return recs

@st.cache_data
def preprocess_with_options(df_in, features, fill_age_with_method="median", remove_duplicates=False, remove_missing=False):
    dfp = df_in.copy()
    
    # remove duplicates
    if remove_duplicates:
        dfp = dfp.drop_duplicates()
    
    # victim_age normalization
    if "victim_age" in dfp.columns:
        dfp["victim_age"] = pd.to_numeric(dfp["victim_age"].replace({"kosong": np.nan, "": np.nan}), errors="coerce")
        if fill_age_with_method == "median":
            dfp["victim_age"] = dfp["victim_age"].fillna(dfp["victim_age"].median())
        elif fill_age_with_method == "mean":
            dfp["victim_age"] = dfp["victim_age"].fillna(dfp["victim_age"].mean())
        else:
            dfp["victim_age"] = dfp["victim_age"].fillna(0)
    
    # fill categories
    for c in ["victim_race", "victim_sex", "state", "disposition"]:
        if c in dfp.columns:
            dfp[c] = dfp[c].fillna("Unknown").astype(str)
    
    # remove rows with missing in selected features
    if remove_missing:
        dfp = dfp.dropna(subset=features)
    
    X_parts = []
    cols = []
    for f in features:
        if f not in dfp.columns:
            continue
        if dfp[f].dtype.kind in "biufc":
            X_parts.append(dfp[[f]].astype(float))
            cols.append(f)
        else:
            dummies = pd.get_dummies(dfp[f].astype(str), prefix=f)
            X_parts.append(dummies)
            cols += list(dummies.columns)
    
    if not X_parts:
        return None, None, None
    X = pd.concat(X_parts, axis=1).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return dfp, X_scaled, cols

@st.cache_data
def compute_k_metrics(X, k_min=2, k_max=8, random_state=42):
    ks = list(range(k_min, k_max+1))
    inertias = []
    silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if len(set(labels)) > 1 and X.shape[0] > k:
            try:
                s = silhouette_score(X, labels)
            except Exception:
                s = None
        else:
            s = None
        silhouettes.append(s)
    return ks, inertias, silhouettes

def suggest_k(ks, inertias, silhouettes):
    valid_sil = [(k, s) for k, s in zip(ks, silhouettes) if s is not None]
    if valid_sil:
        best = max(valid_sil, key=lambda x: x[1])[0]
        return best, "silhouette"
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    if len(diffs2) > 0:
        elbow_idx = np.argmin(diffs2) + 2
        if 2 <= elbow_idx <= len(ks):
            return ks[elbow_idx-1], "elbow"
    return ks[0], "default"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                  SECTION 7: SIDEBAR NAVIGATION                            ║
# ║                                                                           ║
# ║  Navigasi sidebar dengan menu:                                           ║
# ║  - Dashboard: Halaman utama                                               ║
# ║  - Input Dataset: Upload/load CSV                                        ║
# ║  - Preprocessing Data: Cleaning & feature selection                      ║
# ║  - Analisis Data: Elbow & Silhouette method                              ║
# ║  - Visualisasi: Clustering & visualisasi 2D                             ║
# ║  - Hasil: Kesimpulan & rekomendasi                                       ║
# ║  - Team: Profil tim pengembang                                           ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# SVG Icons (clean, minimal design)
svg_icons = {
    "Dashboard": '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>''',
    "Team": '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>''',
    "Input Dataset": '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>''',
    "Preprocessing Data": '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>''',
    "Analisis Data": '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>''',
    "Visualisasi": '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>''',
    "Hasil": '''<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>'''
}

# Sidebar title with SVG
st.sidebar.markdown("""
<div class="sidebar-header">
    <div style="text-align: center; line-height: 1.3;">
        <div>Menu</div>
        <div>Analisis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Menu items with icons
menu_items = ["Dashboard", "Input Dataset", "Preprocessing Data", "Analisis Data", "Visualisasi", "Hasil", "Team"]

# Initialize session state for page
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Generate custom menu with SVG icons
def create_menu_html(items, icons, current_page):
    menu_html = '<div class="custom-menu">'
    for item in items:
        is_active = "active" if item == current_page else ""
        icon = icons.get(item, "")
        menu_html += f'''
        <div class="menu-item {is_active}" data-page="{item}">
            <span class="menu-icon">{icon}</span>
            <span class="menu-text">{item}</span>
        </div>
        '''
    menu_html += '</div>'
    return menu_html

# Create custom styled radio buttons with icons using Streamlit's native radio
# but with CSS to inject icons
st.sidebar.markdown("""
<style>
/* Custom menu styling with icons */
.custom-menu {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0 0.75rem;
}

.menu-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.9rem 1.25rem;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.25s ease;
    color: rgba(255, 255, 255, 0.65);
    font-weight: 500;
    font-size: 0.95rem;
    border: 1px solid transparent;
    background: transparent;
}

.menu-item:hover {
    background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(30, 30, 30, 0.5) 100%);
    color: #ffffff;
    transform: translateX(6px);
}

.menu-item.active {
    background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(185, 28, 28, 0.15) 100%);
    color: #ffffff;
    border: 1px solid rgba(220, 38, 38, 0.6);
    box-shadow: 0 0 20px rgba(220, 38, 38, 0.2), inset 0 0 20px rgba(220, 38, 38, 0.05);
    font-weight: 600;
    position: relative;
}

.menu-item.active::before {
    content: "";
    position: absolute;
    left: 0;
    top: 15%;
    bottom: 15%;
    width: 3px;
    background: linear-gradient(180deg, #ef4444, #dc2626);
    border-radius: 0 3px 3px 0;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.6);
}

.menu-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.menu-icon svg {
    stroke: rgba(255, 255, 255, 0.6);
    transition: all 0.25s ease;
}

.menu-item:hover .menu-icon svg {
    stroke: #ef4444;
    filter: drop-shadow(0 0 4px rgba(239, 68, 68, 0.4));
}

.menu-item.active .menu-icon svg {
    stroke: #ef4444;
    filter: drop-shadow(0 0 6px rgba(239, 68, 68, 0.5));
}

.menu-text {
    flex-grow: 1;
}

/* Hide default Streamlit radio styling completely */
[data-testid="stSidebar"] .stRadio {
    margin-top: -10px;
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    padding-left: 3rem !important;
}

/* Add icon indicators via CSS for each menu item */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label::before {
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Dashboard icon */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:nth-child(1)::after {
    content: "";
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='14' width='7' height='7'%3E%3C/rect%3E%3Crect x='3' y='14' width='7' height='7'%3E%3C/rect%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}

/* Input Dataset icon */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:nth-child(2)::after {
    content: "";
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'%3E%3C/path%3E%3Cpolyline points='17 8 12 3 7 8'%3E%3C/polyline%3E%3Cline x1='12' y1='3' x2='12' y2='15'%3E%3C/line%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}

/* Preprocessing Data - extra padding for longer text */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:nth-child(3) {
    padding-top: 0.65rem !important;
    padding-bottom: 0.65rem !important;
    height: auto !important;
    min-height: 52px !important;
    max-height: 52px !important;
}

/* Preprocessing Data icon */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:nth-child(3)::after {
    content: "";
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'%3E%3C/circle%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}

/* Analisis Data icon */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:nth-child(4)::after {
    content: "";
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'%3E%3C/line%3E%3Cline x1='12' y1='20' x2='12' y2='4'%3E%3C/line%3E%3Cline x1='6' y1='20' x2='6' y2='14'%3E%3C/line%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}

/* Visualisasi icon */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:nth-child(5)::after {
    content: "";
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Cline x1='2' y1='12' x2='22' y2='12'%3E%3C/line%3E%3Cpath d='M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}

/* Hasil icon */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:nth-child(6)::after {
    content: "";
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'%3E%3C/path%3E%3Cpolyline points='14 2 14 8 20 8'%3E%3C/polyline%3E%3Cline x1='16' y1='13' x2='8' y2='13'%3E%3C/line%3E%3Cline x1='16' y1='17' x2='8' y2='17'%3E%3C/line%3E%3Cpolyline points='10 9 9 9 8 9'%3E%3C/polyline%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}
                    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:nth-child(7)::after {
    content: "";
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2'%3E%3C/path%3E%3Ccircle cx='9' cy='7' r='4'%3E%3C/circle%3E%3Cpath d='M23 21v-2a4 4 0 0 0-3-3.87'%3E%3C/path%3E%3Cpath d='M16 3.13a4 4 0 0 1 0 7.75'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}
</style>
""", unsafe_allow_html=True)

# Create radio with icons
page = st.sidebar.radio("", menu_items, label_visibility="collapsed")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                         PAGE 1: DASHBOARD                                 ║
# ║                                                                           ║
# ║  Halaman utama yang menampilkan:                                          ║
# ║  - Latar belakang & tujuan analisis                                       ║
# ║  - Tujuan dashboard                                                       ║
# ║  - Ringkasan dataset (jika sudah dimuat)                                  ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if page == "Dashboard":
    # Section 1: Latar Belakang & Tujuan Analisis
    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg></div><h3 class="section-title">Latar Belakang & Tujuan Analisis</h3></div><div class="dashboard-content"><p>Kasus pembunuhan merupakan masalah sosial yang serius karena tidak hanya menimbulkan penderitaan bagi korban dan keluarga, tetapi juga membuat masyarakat merasa tidak aman. Selama ini, upaya pencegahan kejahatan sering kali bersifat umum dan kurang tepat sasaran karena tidak mempertimbangkan karakteristik korban. Padahal, tingkat risiko terhadap pembunuhan bisa berbeda-beda tergantung usia, jenis kelamin, dan ras seseorang.</p><p>Oleh karena itu, penting untuk melakukan analisis yang berfokus pada korban agar bisa diketahui kelompok mana yang paling rentan menjadi sasaran kejahatan. Dengan memahami pola-pola tersebut, hasil analisis dapat menjadi dasar bagi pemerintah dan aparat hukum untuk merancang kebijakan pencegahan yang lebih efektif dan tepat sasaran, seperti:</p><div class="bullet-list"><div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg><span><strong>Meningkatkan patroli dan pengawasan</strong> di wilayah yang rawan dengan tingkat pembunuhan tinggi</span></div><div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg><span><strong>Menyusun program perlindungan khusus</strong> untuk kelompok rentan, seperti perempuan yang bekerja malam hari atau lansia yang tinggal sendiri</span></div><div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg><span><strong>Mengembangkan strategi edukasi</strong> masyarakat yang disesuaikan dengan karakteristik demografis tertentu</span></div><div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg><span><strong>Mengalokasikan sumber daya</strong> lebih efisien berdasarkan pola risiko yang teridentifikasi</span></div></div></div></div>""", unsafe_allow_html=True)
    
    # Section 2: Tujuan Dashboard
    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg></div><h3 class="section-title">Tujuan Dashboard</h3></div><div class="dashboard-content"><p>Dashboard ini menggunakan <strong>K-Means Clustering</strong> untuk mengelompokkan kasus pembunuhan berdasarkan karakteristik korban (usia, jenis kelamin, ras, lokasi, dan lainnya). Dengan mengidentifikasi klaster-klaster korban yang memiliki:</p><div class="bullet-list"><div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg><span><strong>Risiko tinggi</strong> atau <strong>pola tertentu</strong></span></div><div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg><span><strong>Kerentanan khusus</strong> terhadap tindak kejahatan</span></div><div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg><span><strong>Profil demografis serupa</strong></span></div></div><p style="margin-top: 16px;">Kita dapat memahami faktor-faktor yang mempengaruhi kerentanan korban dan memberikan rekomendasi untuk pencegahan yang lebih tertarget.</p></div></div>""", unsafe_allow_html=True)
    
    # Section 3: Ringkasan Dataset
    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg></div><h3 class="section-title">Ringkasan Dataset (jika sudah dimuat)</h3></div>""", unsafe_allow_html=True)
    
    df = st.session_state.df_raw
    if df.empty:
        st.markdown("""<div class="info-box"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span>Belum ada dataset dimuat. Silakan ke <strong>'Input Dataset'</strong> untuk memuat CSV.</span></div>""", unsafe_allow_html=True)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            total = len(df)
            st.markdown(f"""<div class="metric-card"><div class="metric-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg></div><div class="metric-label">Jumlah kasus (baris)</div><div class="metric-value">{total:,}</div></div>""", unsafe_allow_html=True)
        with col2:
            if "victim_age" in df.columns:
                avg_age = pd.to_numeric(df["victim_age"].replace({"kosong": np.nan}), errors="coerce").dropna().mean()
                age_display = f"{avg_age:.1f} tahun"
            else:
                age_display = "—"
            st.markdown(f"""<div class="metric-card"><div class="metric-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg></div><div class="metric-label">Rata-rata usia korban</div><div class="metric-value">{age_display}</div></div>""", unsafe_allow_html=True)
        with col3:
            races = df.get("victim_race", pd.Series()).dropna()
            race_count = races.nunique()
            st.markdown(f"""<div class="metric-card"><div class="metric-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg></div><div class="metric-label">Ras unik</div><div class="metric-value">{race_count}</div></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg></div><h3 class="section-title">Preview data (10 baris pertama)</h3></div>""", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PAGE (LEGACY): VISUALISASI & HASIL                     ║
# ║                                                                           ║
# ║  Halaman lama yang menggabungkan visualisasi dan hasil.                   ║
# ║  (Tidak digunakan dalam menu saat ini)                                    ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "Visualisasi & Hasil":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Visualisasi hasil clustering")
    dfp = st.session_state.df_proc if not st.session_state.df_proc.empty else st.session_state.df_raw
    Xscaled = st.session_state.X_scaled
    selected = st.session_state.selected_features

    if Xscaled is None:
        st.warning("Pra-proses belum dijalankan. Pergi ke 'Preprocessing Data' dan klik preview.")
        st.stop()

    st.write("Parameter clustering:")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        use_auto = st.checkbox("Gunakan K saran otomatis (Analisis)", value=True)
    with col_b:
        k_manual = st.number_input("K manual", min_value=2, max_value=20, value=3, step=1)
    with col_c:
        dr_method = st.selectbox("Metode reduksi dimensi", options=["PCA", "t-SNE", "UMAP"] if UMAP_AVAILABLE else ["PCA", "t-SNE"])
    if dr_method == "t-SNE":
        tsne_perp = st.slider("t-SNE perplexity", 5, 50, 30)

    # compute suggested k if requested (fast compute small k range)
    suggested_k = None
    if use_auto:
        ks, inertias, silhouettes = compute_k_metrics(Xscaled, k_min=2, k_max=min(10, max(3, int(k_manual)+5)), random_state=42)
        suggested_k, _ = suggest_k(ks, inertias, silhouettes)
    final_k = suggested_k if use_auto and suggested_k is not None else int(k_manual)

    if st.button("Run Clustering & Visualize"):
        with st.spinner("Menjalankan KMeans..."):
            kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(Xscaled)
            dfp["cluster"] = labels
            st.session_state.df_proc = dfp

            # DR
            if dr_method == "PCA":
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(Xscaled)
            elif dr_method == "t-SNE":
                reducer = TSNE(n_components=2, perplexity=tsne_perp, random_state=42, init="pca")
                coords = reducer.fit_transform(Xscaled)
            elif dr_method == "UMAP" and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(Xscaled)
            else:
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(Xscaled)
            dfp["_x"] = coords[:, 0]
            dfp["_y"] = coords[:, 1]

            try:
                sil = silhouette_score(Xscaled, labels) if len(set(labels)) > 1 else None
            except Exception:
                sil = None

        st.success("Clustering selesai.")
        # display
        left, right = st.columns([2,1])
        hover_cols = [c for c in ["uid","report_date","victim_race","victim_age","victim_sex","state","disposition","lat","lon"] if c in dfp.columns]
        with left:
            fig = px.scatter(dfp, x="_x", y="_y", color=dfp["cluster"].astype(str), hover_data=hover_cols, title="Visualisasi klaster (2D)")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("**Ringkasan**")
            st.write(f"K final: {final_k}")
            if sil is not None:
                st.write(f"Silhouette: {sil:.3f}")
            st.table(dfp["cluster"].value_counts().sort_index().rename("count").to_frame())

        # cluster stats
        st.subheader("Statistik per klaster")
        stats = []
        for cl in sorted(dfp["cluster"].unique()):
            sub = dfp[dfp["cluster"] == cl]
            row = {"cluster": int(cl), "count": len(sub)}
            if "victim_age" in sub.columns:
                row["mean_age"] = round(sub["victim_age"].dropna().mean(), 2)
            for cat in ["victim_race", "victim_sex", "state", "disposition"]:
                if cat in sub.columns:
                    top = sub[cat].mode()
                    row[f"top_{cat}"] = top.iloc[0] if not top.empty else ""
            stats.append(row)
        st.dataframe(pd.DataFrame(stats).set_index("cluster"))

        # map if coordinates exist
        lat_col = "lat" if "lat" in dfp.columns else ("latitude" if "latitude" in dfp.columns else None)
        lon_col = "lon" if "lon" in dfp.columns else ("longitude" if "longitude" in dfp.columns else None)
        if lat_col and lon_col and dfp[lat_col].notna().sum() > 0:
            st.subheader("Peta: Distribusi klaster")
            df_map = dfp.dropna(subset=[lat_col, lon_col])
            fig_map = px.scatter_mapbox(df_map, lat=lat_col, lon=lon_col, color=df_map["cluster"].astype(str),
                                        hover_data=hover_cols, zoom=10, height=600)
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Kolom lat/lon tidak ada atau kosong — peta tidak ditampilkan.")

        # sample & download
        st.subheader("Contoh & Unduh Hasil")
        sel = st.selectbox("Pilih klaster untuk contoh baris", options=sorted(dfp["cluster"].unique()))
        st.dataframe(dfp[dfp["cluster"] == sel].head(25))
        csv = dfp.to_csv(index=False).encode("utf-8")
        st.download_button("Download dataset berlabel (CSV)", data=csv, file_name="homicide_with_clusters.csv", mime="text/csv")
    else:
        st.info("Klik 'Run Clustering & Visualize' untuk menjalankan KMeans dan melihat hasil.")

    st.markdown("</div>", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                       PAGE 2: INPUT DATASET                               ║
# ║                                                                           ║
# ║  Halaman untuk memuat dataset:                                            ║
# ║  - Upload file CSV                                                        ║
# ║  - Atau memuat dari path lokal default                                    ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "Input Dataset":
    # Section header
    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg></div><h3 class="section-title">Input Dataset</h3></div><div class="dashboard-content"><p>Anda bisa meng-upload file CSV atau membiarkan aplikasi membaca CSV lokal.</p></div>""", unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload CSV (opsional)", type=["csv"])
    if uploaded is not None:
        df0 = robust_read_csv(uploaded)
        st.session_state.df_raw = df0
        st.markdown("""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;">CSV berhasil diupload dan dimuat ke sesi.</span></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span><strong>Informasi</strong>: {len(df0)} baris, {len(df0.columns)} kolom</span></div>""", unsafe_allow_html=True)
        st.dataframe(df0.head(10), use_container_width=True)
    else:
        st.markdown("""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span>Tidak ada file diupload — mencoba baca CSV lokal:</span></div>""", unsafe_allow_html=True)
        st.code(DEFAULT_CSV)
        if st.button("Muat CSV lokal"):
            df0 = load_csv(DEFAULT_CSV)
            if df0.empty:
                st.markdown("""<div class="bullet-item" style="background: rgba(239, 68, 68, 0.15); border-left-color: #ef4444;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg><span style="color: #fca5a5;">Gagal membaca CSV lokal. Pastikan path benar atau upload file.</span></div>""", unsafe_allow_html=True)
            else:
                st.session_state.df_raw = df0
                st.markdown("""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;">CSV lokal berhasil dimuat.</span></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span><strong>Informasi</strong>: {len(df0)} baris, {len(df0.columns)} kolom</span></div>""", unsafe_allow_html=True)
                st.dataframe(df0.head(10), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                     PAGE 3: PREPROCESSING DATA                            ║
# ║                                                                           ║
# ║  Halaman untuk pra-proses data:                                           ║
# ║  - Deteksi masalah kualitas data                                          ║
# ║  - Rekomendasi cleaning                                                   ║
# ║  - Opsi cleaning manual (hapus duplikat, isi nilai kosong)                ║
# ║  - Pilih fitur untuk clustering                                           ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "Preprocessing Data":
    # Section header
    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg></div><h3 class="section-title">Pra-proses Data</h3></div>""", unsafe_allow_html=True)
    
    df = st.session_state.df_raw
    if df.empty:
        st.markdown("""<div class="bullet-item" style="background: rgba(245, 158, 11, 0.15); border-left-color: #f59e0b;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg><span style="color: #fde68a;">Dataset belum dimuat. Silakan ke 'Input Dataset'.</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    
    # Step 1: Deteksi Masalah Kualitas Data
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg></div><h3 class="section-title" style="font-size: 1.1rem;">1. Deteksi Masalah Kualitas Data</h3></div>""", unsafe_allow_html=True)
    issues = detect_data_quality_issues(df)
    if issues:
        st.markdown("""<div class="recommendation">""", unsafe_allow_html=True)
        st.markdown("""<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg><strong>Masalah data terdeteksi:</strong></div>""", unsafe_allow_html=True)
        for col, issue in issues.items():
            st.write(f"- **{col}**: {issue['count']} nilai kosong ({issue['pct']:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;">Tidak ada nilai kosong terdeteksi.</span></div>""", unsafe_allow_html=True)
    
    # Step 2: Rekomendasi Cleaning
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg></div><h3 class="section-title" style="font-size: 1.1rem;">2. Rekomendasi Cleaning</h3></div>""", unsafe_allow_html=True)
    recs = recommend_cleaning(df)
    if recs:
        st.markdown("""<div class="recommendation">""", unsafe_allow_html=True)
        st.markdown("""<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><strong>Rekomendasi sistem:</strong></div>""", unsafe_allow_html=True)
        for rec in recs:
            st.write(rec)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;">Tidak ada rekomendasi cleaning khusus.</span></div>""", unsafe_allow_html=True)
    
    # Step 3: Opsi Cleaning Manual
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg></div><h3 class="section-title" style="font-size: 1.1rem;">3. Opsi Cleaning Manual</h3></div>""", unsafe_allow_html=True)
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        remove_dup = st.checkbox("Hapus duplikat", value=False)
    with col_opt2:
        remove_miss = st.checkbox("Hapus baris dengan nilai kosong di fitur terpilih", value=False)
    
    fill_choice = st.selectbox("Isi nilai usia kosong dengan:", ["median", "mean", "0"], index=0)
    
    # Step 4: Pilih Fitur untuk Clustering
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg></div><h3 class="section-title" style="font-size: 1.1rem;">4. Pilih Fitur untuk Clustering</h3></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg><span>Kolom terdeteksi:</span></div>""", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"columns": df.columns.tolist()}), use_container_width=True)
    
    suggested = [c for c in ["victim_age", "victim_race", "victim_sex", "state", "disposition", "lat", "lon"] if c in df.columns]
    selected = st.multiselect("Pilih fitur untuk clustering", options=list(df.columns), default=suggested)
    st.session_state.selected_features = selected
    
    if st.button("Preview hasil pra-proses"):
        if len(selected) == 0:
            st.markdown("""<div class="bullet-item" style="background: rgba(239, 68, 68, 0.15); border-left-color: #ef4444;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg><span style="color: #fca5a5;">Pilih minimal satu fitur.</span></div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Menghitung preview..."):
                dfp, Xsc, feat_cols = preprocess_with_options(df, selected, fill_choice, remove_dup, remove_miss)
                if Xsc is None:
                    st.markdown("""<div class="bullet-item" style="background: rgba(239, 68, 68, 0.15); border-left-color: #ef4444;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg><span style="color: #fca5a5;">Tidak ada fitur yang dapat diproses. Periksa pilihan fitur.</span></div>""", unsafe_allow_html=True)
                else:
                    st.session_state.df_cleaned = dfp
                    st.session_state.X_scaled = Xsc
                    st.session_state.feature_cols = feat_cols
                    st.markdown("""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;">Pra-proses selesai.</span></div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span><strong>Hasil</strong>: {len(dfp)} baris (dari {len(df)} baris awal), {len(feat_cols)} fitur terkode</span></div>""", unsafe_allow_html=True)
                    st.markdown("""<div class="dashboard-section" style="margin-top: 16px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg></div><h3 class="section-title" style="font-size: 1.1rem;">Data setelah cleaning:</h3></div>""", unsafe_allow_html=True)
                    st.dataframe(dfp.head(10), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                       PAGE 4: ANALISIS DATA                               ║
# ║                                                                           ║
# ║  Halaman untuk analisis pemilihan K optimal:                              ║
# ║  - Metode Elbow (Inertia)                                                 ║
# ║  - Metode Silhouette Score                                                ║
# ║  - Saran K terbaik secara otomatis                                        ║
# ║  - Opsi pilih K manual                                                    ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "Analisis Data":
    # Section header
    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg></div><h3 class="section-title">Analisis: Elbow & Silhouette untuk Memilih K</h3></div>""", unsafe_allow_html=True)
    
    Xscaled = st.session_state.X_scaled
    selected = st.session_state.selected_features
    
    if Xscaled is None:
        if len(selected) == 0:
            st.markdown("""<div class="bullet-item" style="background: rgba(245, 158, 11, 0.15); border-left-color: #f59e0b;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg><span style="color: #fde68a;">Belum ada fitur terpilih. Pergi ke 'Preprocessing Data' dan pilih fitur.</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span>Jalankan preview pra-proses di halaman 'Preprocessing Data' terlebih dahulu.</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    
    # Step 1: Hitung Metrik Elbow & Silhouette
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg></div><h3 class="section-title" style="font-size: 1.1rem;">1. Hitung Metrik Elbow & Silhouette</h3></div>""", unsafe_allow_html=True)
    max_k = st.slider("Max K untuk diuji", min_value=3, max_value=12, value=8)
    
    compute = st.button("Compute Elbow & Silhouette")
    if compute:
        with st.spinner("Menghitung metrik untuk setiap K..."):
            ks, inertias, silhouettes = compute_k_metrics(Xscaled, k_min=2, k_max=max_k, random_state=42)
            suggested_k, method_used = suggest_k(ks, inertias, silhouettes)
            st.session_state.suggested_k = suggested_k
            st.session_state.suggested_method = method_used
            
            st.markdown(f"""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;"><strong>Saran K terbaik: {suggested_k}</strong> (metode: {method_used})</span></div>""", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(x=ks, y=inertias, markers=True, title="Elbow (Inertia)", 
                             labels={"x": "K", "y": "Inertia"})
                fig.add_vline(x=suggested_k, line_dash="dash", line_color="red", annotation_text=f"K={suggested_k}")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,20,20,0.8)', font_color='#e5e5e5')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.line(x=ks, y=[s if s is not None else np.nan for s in silhouettes], markers=True, 
                              title="Silhouette Score", labels={"x": "K", "y": "Silhouette"})
                fig2.add_vline(x=suggested_k, line_dash="dash", line_color="red", annotation_text=f"K={suggested_k}")
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,20,20,0.8)', font_color='#e5e5e5')
                st.plotly_chart(fig2, use_container_width=True)
    else:
        if st.session_state.suggested_k is not None:
            st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span>Saran K sebelumnya: {st.session_state.suggested_k}</span></div>""", unsafe_allow_html=True)
    
    # Step 2: Pilih K untuk Clustering
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg></div><h3 class="section-title" style="font-size: 1.1rem;">2. Pilih K untuk Clustering</h3></div>""", unsafe_allow_html=True)
    use_auto = st.checkbox("Gunakan K saran otomatis", value=True)
    if not use_auto:
        k_manual = st.number_input("K manual", min_value=2, max_value=20, value=3, step=1)
        final_k = int(k_manual)
    else:
        if st.session_state.suggested_k is not None:
            final_k = st.session_state.suggested_k
        else:
            final_k = 3
            st.markdown("""<div class="bullet-item" style="background: rgba(245, 158, 11, 0.15); border-left-color: #f59e0b;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg><span style="color: #fde68a;">Belum ada saran K. Gunakan default K=3 atau compute terlebih dahulu.</span></div>""", unsafe_allow_html=True)
    
    st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span><strong>K yang akan digunakan: {final_k}</strong></span></div>""", unsafe_allow_html=True)
    
    if st.button("Konfirmasi & Lanjut ke Visualisasi"):
        st.session_state.final_k = final_k
        st.markdown("""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;">K dikonfirmasi. Lanjut ke halaman 'Visualisasi' untuk melihat hasil clustering.</span></div>""", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                        PAGE 5: VISUALISASI                                ║
# ║                                                                           ║
# ║  Halaman untuk menjalankan clustering dan visualisasi:                    ║
# ║  - Pilih metode reduksi dimensi (PCA, t-SNE, UMAP)                        ║
# ║  - Jalankan K-Means clustering                                            ║
# ║  - Visualisasi klaster 2D                                                 ║
# ║  - Peta geografis (jika ada koordinat lat/lon)                            ║
# ║  - Download hasil clustering                                              ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "Visualisasi":
    # Section header
    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg></div><h3 class="section-title">Visualisasi Hasil Clustering</h3></div>""", unsafe_allow_html=True)
    
    Xscaled = st.session_state.X_scaled
    dfp = st.session_state.df_cleaned if not st.session_state.df_cleaned.empty else st.session_state.df_raw
    final_k = st.session_state.final_k
    
    if Xscaled is None or final_k is None:
        st.markdown("""<div class="bullet-item" style="background: rgba(245, 158, 11, 0.15); border-left-color: #f59e0b;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg><span style="color: #fde68a;">Pra-proses atau analisis belum selesai. Silakan kerjakan tahap sebelumnya.</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    
    # Step 1: Pilih Metode Reduksi Dimensi
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg></div><h3 class="section-title" style="font-size: 1.1rem;">1. Pilih Metode Reduksi Dimensi</h3></div>""", unsafe_allow_html=True)
    col_dr = st.columns([1, 2])
    with col_dr[0]:
        dr_method = st.selectbox("Metode", options=["PCA", "t-SNE", "UMAP"] if UMAP_AVAILABLE else ["PCA", "t-SNE"])
    if dr_method == "t-SNE":
        with col_dr[1]:
            tsne_perp = st.slider("t-SNE perplexity", 5, 50, 30)
    else:
        tsne_perp = None
    
    if st.button("Jalankan Clustering & Visualisasi"):
        with st.spinner("Menjalankan K-Means..."):
            kmeans = KMeans(n_clusters=int(final_k), random_state=42, n_init=10)
            labels = kmeans.fit_predict(Xscaled)
            dfp = dfp.copy()
            dfp["cluster"] = labels
            st.session_state.df_proc = dfp
            st.session_state.cluster_labels = labels
            
            # DR
            if dr_method == "PCA":
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(Xscaled)
            elif dr_method == "t-SNE":
                reducer = TSNE(n_components=2, perplexity=tsne_perp, random_state=42, init="pca")
                coords = reducer.fit_transform(Xscaled)
            elif dr_method == "UMAP" and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(Xscaled)
            else:
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(Xscaled)
            
            dfp["_x"] = coords[:, 0]
            dfp["_y"] = coords[:, 1]
            
            try:
                sil = silhouette_score(Xscaled, labels) if len(set(labels)) > 1 else None
                st.session_state.silhouette_score_val = sil
            except Exception:
                sil = None
        
        st.markdown("""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;">Clustering selesai.</span></div>""", unsafe_allow_html=True)
        
        # Display
        st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg></div><h3 class="section-title" style="font-size: 1.1rem;">2. Visualisasi Klaster (2D)</h3></div>""", unsafe_allow_html=True)
        left, right = st.columns([2, 1])
        hover_cols = [c for c in ["uid", "report_date", "victim_race", "victim_age", "victim_sex", "state", "disposition", "lat", "lon"] if c in dfp.columns]
        
        with left:
            fig = px.scatter(dfp, x="_x", y="_y", color=dfp["cluster"].astype(str), 
                           hover_data=hover_cols, title="Visualisasi Klaster (2D)", 
                           labels={"_x": "Dimensi 1", "_y": "Dimensi 2"})
            fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,20,20,0.8)', font_color='#e5e5e5')
            st.plotly_chart(fig, use_container_width=True)
        
        with right:
            st.markdown("""<div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg></div><h3 class="section-title" style="font-size: 1rem;">Ringkasan</h3></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle></svg><span><strong>K final:</strong> {final_k}</span></div>""", unsafe_allow_html=True)
            if sil is not None:
                st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg><span><strong>Silhouette:</strong> {sil:.3f}</span></div>""", unsafe_allow_html=True)
            st.table(dfp["cluster"].value_counts().sort_index().rename("count").to_frame())
        
        # Map if coordinates available
        st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg></div><h3 class="section-title" style="font-size: 1.1rem;">3. Peta Geografis</h3></div>""", unsafe_allow_html=True)
        lat_col = "lat" if "lat" in dfp.columns else ("latitude" if "latitude" in dfp.columns else None)
        lon_col = "lon" if "lon" in dfp.columns else ("longitude" if "longitude" in dfp.columns else None)
        
        if lat_col and lon_col and dfp[lat_col].notna().sum() > 0:
            df_map = dfp.dropna(subset=[lat_col, lon_col])
            fig_map = px.scatter_mapbox(df_map, lat=lat_col, lon=lon_col, color=df_map["cluster"].astype(str),
                                       hover_data=hover_cols, zoom=10, height=600, title="Distribusi Klaster per Lokasi")
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.markdown("""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span>Kolom lat/lon tidak ada atau kosong — peta tidak ditampilkan.</span></div>""", unsafe_allow_html=True)
        
        # Sample by cluster
        st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg></div><h3 class="section-title" style="font-size: 1.1rem;">4. Contoh Baris per Klaster</h3></div>""", unsafe_allow_html=True)
        sel = st.selectbox("Pilih klaster untuk contoh baris", options=sorted(dfp["cluster"].unique()))
        st.dataframe(dfp[dfp["cluster"] == sel].head(25), use_container_width=True)
        
        # Download
        csv = dfp.to_csv(index=False).encode("utf-8")
        st.download_button("Download dataset berlabel (CSV)", data=csv, file_name="homicide_with_clusters.csv", mime="text/csv")
    
    else:
        st.markdown("""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span>Klik 'Jalankan Clustering & Visualisasi' untuk memulai.</span></div>""", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                          PAGE 6: HASIL                                    ║
# ║                                                                           ║
# ║  Halaman untuk menampilkan hasil analisis:                                ║
# ║  - Statistik per klaster                                                  ║
# ║  - Insights & kesimpulan otomatis                                         ║
# ║  - Rekomendasi kebijakan                                                  ║
# ║  - Metrik kualitas clustering (Silhouette Score)                          ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "Hasil":
    # Section header
    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg></div><h3 class="section-title">Kesimpulan & Hasil Analisis</h3></div>""", unsafe_allow_html=True)
    
    dfp = st.session_state.df_proc
    labels = st.session_state.cluster_labels
    final_k = st.session_state.final_k
    sil = st.session_state.silhouette_score_val
    
    if dfp.empty or labels is None:
        st.markdown("""<div class="bullet-item" style="background: rgba(245, 158, 11, 0.15); border-left-color: #f59e0b;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg><span style="color: #fde68a;">Belum ada clustering hasil. Silakan jalankan clustering di halaman 'Visualisasi' terlebih dahulu.</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    
    # Statistik Per Klaster
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg></div><h3 class="section-title" style="font-size: 1.1rem;">Statistik Per Klaster</h3></div>""", unsafe_allow_html=True)
    stats = []
    for cl in sorted(dfp["cluster"].unique()):
        sub = dfp[dfp["cluster"] == cl]
        row = {"Klaster": int(cl), "Jumlah": len(sub), "Persentase": f"{100*len(sub)/len(dfp):.1f}%"}
        if "victim_age" in sub.columns:
            row["Rata-rata Usia"] = round(sub["victim_age"].dropna().mean(), 1)
            row["Median Usia"] = float(sub["victim_age"].median())
        for cat in ["victim_race", "victim_sex", "state", "disposition"]:
            if cat in sub.columns:
                top = sub[cat].mode()
                row[f"Top {cat}"] = top.iloc[0] if not top.empty else "—"
        stats.append(row)
    
    st.dataframe(pd.DataFrame(stats), use_container_width=True)
    
    # Insights & Kesimpulan
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg></div><h3 class="section-title" style="font-size: 1.1rem;">Insights & Kesimpulan</h3></div>""", unsafe_allow_html=True)
    
    # Generate insights
    for cl in sorted(dfp["cluster"].unique()):
        sub = dfp[dfp["cluster"] == cl]
        pct = 100*len(sub)/len(dfp)
        
        st.markdown(f"""<div class="insight">""", unsafe_allow_html=True)
        st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="3"></circle></svg><span><strong>Klaster {int(cl)}</strong> ({len(sub)} kasus, {pct:.1f}%)</span></div>""", unsafe_allow_html=True)
        
        insights = []
        if "victim_age" in sub.columns:
            avg_age = sub["victim_age"].dropna().mean()
            if avg_age < 25:
                insights.append(("users", "Didominasi korban muda (< 25 tahun) — tinggi risiko di kalangan anak-anak & remaja"))
            elif avg_age > 60:
                insights.append(("user", "Didominasi korban lansia (> 60 tahun) — kelompok rentan dengan mobilitas terbatas"))
            else:
                insights.append(("user", f"Korban dewasa (rata-rata {avg_age:.0f} tahun)"))
        
        if "victim_race" in sub.columns:
            top_race = sub["victim_race"].mode()
            if not top_race.empty:
                insights.append(("globe", f"Ras dominan: {top_race.iloc[0]}"))
        
        if "victim_sex" in sub.columns:
            top_sex = sub["victim_sex"].mode()
            if not top_sex.empty and top_sex.iloc[0] == "Female":
                insights.append(("user", "Tingkat korban perempuan tinggi — perlu program perlindungan khusus"))
            elif not top_sex.empty:
                insights.append(("user", f"Jenis kelamin dominan: {top_sex.iloc[0]}"))
        
        if "state" in sub.columns:
            top_state = sub["state"].mode()
            if not top_state.empty:
                insights.append(("map-pin", f"Lokasi dominan: {top_state.iloc[0]}"))
        
        if "disposition" in sub.columns:
            closed_rate = (sub["disposition"].str.contains("Closed", case=False).sum() / len(sub)) * 100
            insights.append(("file-text", f"Tingkat penyelesaian kasus: {closed_rate:.1f}%"))
        
        for icon_type, insight_text in insights:
            if icon_type == "users":
                svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>'
            elif icon_type == "user":
                svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
            elif icon_type == "globe":
                svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>'
            elif icon_type == "map-pin":
                svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>'
            else:
                svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>'
            st.markdown(f"""<div style="display: flex; align-items: center; gap: 8px; margin-left: 10px; color: #d1d5db; font-size: 0.95rem;">{svg}<span>{insight_text}</span></div>""", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Rekomendasi Kebijakan
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg></div><h3 class="section-title" style="font-size: 1.1rem;">Rekomendasi Kebijakan</h3></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="dashboard-content"><p>Berdasarkan hasil clustering, berikut rekomendasi untuk pencegahan yang lebih efektif:</p></div>""", unsafe_allow_html=True)
    
    recommendations = [
        ("Identifikasi Klaster Berisiko Tinggi", "Fokus pada klaster dengan jumlah kasus terbanyak atau tingkat penyelesaian rendah"),
        ("Program Perlindungan Khusus", "Untuk kelompok korban tertentu (perempuan, lansia, anak-anak)"),
        ("Penguatan Pengawasan Wilayah", "Tingkatkan patroli di lokasi-lokasi dengan konsentrasi kasus tinggi"),
        ("Edukasi Masyarakat", "Berbeda untuk setiap segmen demografis klaster"),
        ("Alokasi Sumber Daya", "Distribusikan personel & anggaran berdasarkan risiko per klaster")
    ]
    
    for i, (title, desc) in enumerate(recommendations, 1):
        st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg><span><strong>{i}. {title}</strong>: {desc}</span></div>""", unsafe_allow_html=True)
    
    # Metrik Kualitas Clustering
    st.markdown("""<div class="dashboard-section" style="margin-top: 20px;"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg></div><h3 class="section-title" style="font-size: 1.1rem;">Metrik Kualitas Clustering</h3></div>""", unsafe_allow_html=True)
    if sil is not None:
        st.markdown(f"""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg><span><strong>Silhouette Score</strong>: {sil:.3f}</span></div>""", unsafe_allow_html=True)
        if sil > 0.5:
            st.markdown("""<div class="bullet-item" style="background: rgba(16, 185, 129, 0.15); border-left-color: #10b981;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg><span style="color: #a7f3d0;">Clustering kualitas baik (silhouette > 0.5)</span></div>""", unsafe_allow_html=True)
        elif sil > 0.3:
            st.markdown("""<div class="bullet-item"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg><span>Clustering kualitas sedang (0.3 < silhouette ≤ 0.5)</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="bullet-item" style="background: rgba(245, 158, 11, 0.15); border-left-color: #f59e0b;"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg><span style="color: #fde68a;">Clustering kualitas kurang (silhouette ≤ 0.3)</span></div>""", unsafe_allow_html=True)
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                          PAGE 7: TEAM                                     ║
# ║                                                                           ║
# ║  Halaman profil tim pengembang:                                           ║
# ║  - Foto anggota tim                                                       ║
# ║  - Nama, NIM, dan role                                                    ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "Team":
    import base64 

    # Fungsi helper gambar
    def img_to_html(img_path):
        try:
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                return f'<img src="data:image/png;base64,{data}" class="member-img">'
            else:
                return f'<img src="https://via.placeholder.com/300x400/1a1a1a/ef4444?text=Foto" class="member-img">'
        except Exception:
            return f'<img src="https://via.placeholder.com/300x400/1a1a1a/ef4444?text=Error" class="member-img">'

    st.markdown("""<div class="card"><div class="dashboard-section"><div class="section-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg></div><h3 class="section-title">Tim Pengembang</h3></div><div class="dashboard-content"><p>Berikut adalah anggota tim yang menyusun dashboard analisis clustering ini.</p></div>""", unsafe_allow_html=True)
    
    # CSS Style: Foto Full Width (100%)
    st.markdown("""
    <style>
    .member-card {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.8) 0%, rgba(20, 20, 20, 0.9) 100%);
        border: 1px solid rgba(220, 38, 38, 0.3);
        border-radius: 12px;
        padding: 12px;             /* Padding dikurangi biar foto makin 'nendang' */
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }
    
    .member-card:hover {
        transform: translateY(-5px);
        border-color: #ef4444;
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.2);
    }
    
    /* PENGATURAN FOTO FULL */
    .member-img {
        height: 320px;               /* TINGGI TETAP: Biar tinggi foto seragam */
        object-fit: cover;           /* KUNCI: Crop otomatis biar penuh & tidak gepeng */
        object-position: top center; /* Fokus wajah (atas) */
        border-radius: 8px;
        border: 2px solid rgba(220, 38, 38, 0.5);
        margin-bottom: 12px;
        display: block;
    }
    
    .member-name {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 4px;
        line-height: 1.2;
    }
    .member-role {
        color: #ef4444;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 4px 0;
    }
    .member-nim {
        color: #9ca3af;
        font-size: 0.85rem;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # --- ANGGOTA 1 ---
    with col1:
        foto_1 = img_to_html("po.jpg") 
        st.markdown(f"""
        <div class="member-card">
            {foto_1}
            <div class="member-name">Trillen Surya Ningsih</div>
            <div class="member-role">2311522004</div>
            <div class="member-nim">Anggota 1</div>
        </div>
        """, unsafe_allow_html=True)

    # --- ANGGOTA 2 ---
    with col2:
        foto_2 = img_to_html("plo.png")
        st.markdown(f"""
        <div class="member-card">
            {foto_2}
            <div class="member-name">Zakky Aulia Aldrin</div>
            <div class="member-role">2311522018</div>
            <div class="member-nim">Anggota 2</div>
        </div>
        """, unsafe_allow_html=True)

    # --- ANGGOTA 3 ---
    with col3:
        foto_3 = img_to_html("sokganteng.jpg")
        st.markdown(f"""
        <div class="member-card">
            {foto_3}
            <div class="member-name">Dimas Radithya Nurizkitha</div>
            <div class="member-role">2311523026</div>
            <div class="member-nim">Anggota 3</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)