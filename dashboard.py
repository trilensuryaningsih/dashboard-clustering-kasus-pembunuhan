import os
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

# ===========================================
# CONFIG & CONSTANTS
# ===========================================
DEFAULT_CSV = r"C:\praktikum\homicide-data.csv"
ACCENT = "#6C5CE7"

st.set_page_config(page_title="Clustering Kasus Pembunuhan", layout="wide")

# CSS styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #f7f8ff 0%, #ffffff 100%);
    }}
    .header-title {{
        color: {ACCENT};
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }}
    .card {{
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid {ACCENT};
    }}
    .bg-info {{
        background: #eff6ff;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #3b82f6;
        margin: 12px 0;
    }}
    .recommendation {{
        background: #fef3c7;
        padding: 12px;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        margin: 8px 0;
    }}
    .insight {{
        background: #f0fdf4;
        padding: 12px;
        border-left: 4px solid #10b981;
        border-radius: 6px;
        margin: 8px 0;
    }}
    .muted {{ color: #6b7280; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<h1 class='header-title'>🔍 Clustering Kasus Pembunuhan — Analisis Korban</h1>", unsafe_allow_html=True)
st.markdown("Dashboard interaktif untuk mengelompokkan kasus menurut karakteristik korban. Ikuti alur di sidebar.")

# ===========================================
# SESSION STATE INIT
# ===========================================
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

# ===========================================
# UTILITY FUNCTIONS
# ===========================================
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

# ===========================================
# SIDEBAR NAVIGATION
# ===========================================
st.sidebar.title("📋 Menu Analisis")
page = st.sidebar.radio("", ["Dashboard", "Input Dataset", "Preprocessing Data", "Analisis Data", "Visualisasi", "Hasil"])

# ===========================================
# PAGE: DASHBOARD
# ===========================================
if page == "Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📖 Latar Belakang & Tujuan Analisis")
    
    background_text = """
    Kasus pembunuhan merupakan masalah sosial yang serius karena tidak hanya menimbulkan penderitaan bagi korban dan keluarga, 
    tetapi juga membuat masyarakat merasa tidak aman. Selama ini, upaya pencegahan kejahatan sering kali bersifat umum dan kurang 
    tepat sasaran karena tidak mempertimbangkan karakteristik korban. Padahal, tingkat risiko terhadap pembunuhan bisa berbeda-beda 
    tergantung usia, jenis kelamin, dan ras seseorang.
    
    Oleh karena itu, penting untuk melakukan analisis yang berfokus pada korban agar bisa diketahui kelompok mana yang paling rentan 
    menjadi sasaran kejahatan. Dengan memahami pola-pola tersebut, hasil analisis dapat menjadi dasar bagi pemerintah dan aparat hukum 
    untuk merancang kebijakan pencegahan yang lebih efektif dan tepat sasaran, seperti:
    
    - **Meningkatkan patroli dan pengawasan** di wilayah yang rawan dengan tingkat pembunuhan tinggi
    - **Menyusun program perlindungan khusus** untuk kelompok rentan, seperti perempuan yang bekerja malam hari atau lansia yang tinggal sendiri
    - **Mengembangkan strategi edukasi** masyarakat yang disesuaikan dengan karakteristik demografis tertentu
    - **Mengalokasikan sumber daya** lebih efisien berdasarkan pola risiko yang teridentifikasi
    """
    
    st.markdown(background_text)
    
    st.markdown("---")
    st.subheader("🎯 Tujuan Dashboard")
    st.markdown("""
    Dashboard ini menggunakan **K-Means Clustering** untuk mengelompokkan kasus pembunuhan berdasarkan karakteristik korban 
    (usia, jenis kelamin, ras, lokasi, dan lainnya). Dengan mengidentifikasi klaster-klaster korban yang memiliki:
    - **Risiko tinggi** atau **pola tertentu**
    - **Kerentanan khusus** terhadap tindak kejahatan
    - **Profil demografis serupa**
    
    Kita dapat memahami faktor-faktor yang mempengaruhi kerentanan korban dan memberikan rekomendasi untuk pencegahan yang lebih tertarget.
    """)
    
    st.markdown("---")
    st.subheader("📊 Ringkasan Dataset (jika sudah dimuat)")
    df = st.session_state.df_raw
    if df.empty:
        st.info("ℹ️ Belum ada dataset dimuat. Silakan ke **'Input Dataset'** untuk memuat CSV.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            total = len(df)
            st.metric("📌 Jumlah kasus (baris)", f"{total:,}")
        with col2:
            if "victim_age" in df.columns:
                avg_age = pd.to_numeric(df["victim_age"].replace({"kosong": np.nan}), errors="coerce").dropna().mean()
                st.metric("👤 Rata-rata usia korban", f"{avg_age:.1f} tahun")
            else:
                st.metric("👤 Rata-rata usia korban", "—")
        with col3:
            races = df.get("victim_race", pd.Series()).dropna()
            st.metric("🌍 Ras unik", f"{races.nunique()}")
        
        st.markdown("---")
        st.subheader("Preview data (10 baris pertama)")
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Page: Visualisasi & Hasil
# -----------------------------
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

# ===========================================
# PAGE: INPUT DATASET
# ===========================================
elif page == "Input Dataset":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Input Dataset")
    st.write("Anda bisa meng-upload file CSV atau membiarkan aplikasi membaca CSV lokal.")
    
    uploaded = st.file_uploader("Upload CSV (opsional)", type=["csv"])
    if uploaded is not None:
        df0 = robust_read_csv(uploaded)
        st.session_state.df_raw = df0
        st.success("✅ CSV berhasil diupload dan dimuat ke sesi.")
        st.write(f"**Informasi**: {len(df0)} baris, {len(df0.columns)} kolom")
        st.dataframe(df0.head(10), use_container_width=True)
    else:
        st.write("Tidak ada file diupload — mencoba baca CSV lokal:")
        st.code(DEFAULT_CSV)
        if st.button("📂 Muat CSV lokal"):
            df0 = load_csv(DEFAULT_CSV)
            if df0.empty:
                st.error("❌ Gagal membaca CSV lokal. Pastikan path benar atau upload file.")
            else:
                st.session_state.df_raw = df0
                st.success("✅ CSV lokal berhasil dimuat.")
                st.write(f"**Informasi**: {len(df0)} baris, {len(df0.columns)} kolom")
                st.dataframe(df0.head(10), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================
# PAGE: PREPROCESSING DATA
# ===========================================
elif page == "Preprocessing Data":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔧 Pra-proses Data")
    
    df = st.session_state.df_raw
    if df.empty:
        st.warning("⚠️ Dataset belum dimuat. Silakan ke 'Input Dataset'.")
        st.stop()
    
    st.markdown("#### 1️⃣ Deteksi Masalah Kualitas Data")
    issues = detect_data_quality_issues(df)
    if issues:
        st.markdown("<div class='recommendation'>", unsafe_allow_html=True)
        st.write("**Masalah data terdeteksi:**")
        for col, issue in issues.items():
            st.write(f"- **{col}**: {issue['count']} nilai kosong ({issue['pct']:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("✅ Tidak ada nilai kosong terdeteksi.")
    
    st.markdown("#### 2️⃣ Rekomendasi Cleaning")
    recs = recommend_cleaning(df)
    if recs:
        st.markdown("<div class='recommendation'>", unsafe_allow_html=True)
        st.write("**Rekomendasi sistem:**")
        for rec in recs:
            st.write(rec)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("✅ Tidak ada rekomendasi cleaning khusus.")
    
    st.markdown("#### 3️⃣ Opsi Cleaning Manual")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        remove_dup = st.checkbox("Hapus duplikat", value=False)
    with col_opt2:
        remove_miss = st.checkbox("Hapus baris dengan nilai kosong di fitur terpilih", value=False)
    
    fill_choice = st.selectbox("Isi nilai usia kosong dengan:", ["median", "mean", "0"], index=0)
    
    st.markdown("#### 4️⃣ Pilih Fitur untuk Clustering")
    st.write("Kolom terdeteksi:")
    st.dataframe(pd.DataFrame({"columns": df.columns.tolist()}), use_container_width=True)
    
    suggested = [c for c in ["victim_age", "victim_race", "victim_sex", "state", "disposition", "lat", "lon"] if c in df.columns]
    selected = st.multiselect("Pilih fitur untuk clustering", options=list(df.columns), default=suggested)
    st.session_state.selected_features = selected
    
    if st.button("🔍 Preview hasil pra-proses"):
        if len(selected) == 0:
            st.error("❌ Pilih minimal satu fitur.")
        else:
            with st.spinner("Menghitung preview..."):
                dfp, Xsc, feat_cols = preprocess_with_options(df, selected, fill_choice, remove_dup, remove_miss)
                if Xsc is None:
                    st.error("❌ Tidak ada fitur yang dapat diproses. Periksa pilihan fitur.")
                else:
                    st.session_state.df_cleaned = dfp
                    st.session_state.X_scaled = Xsc
                    st.session_state.feature_cols = feat_cols
                    st.success("✅ Pra-proses selesai.")
                    
                    st.write(f"**Hasil**: {len(dfp)} baris (dari {len(df)} baris awal), {len(feat_cols)} fitur terkode")
                    st.subheader("Data setelah cleaning:")
                    st.dataframe(dfp.head(10), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================
# PAGE: ANALISIS DATA
# ===========================================
elif page == "Analisis Data":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Analisis: Elbow & Silhouette untuk Memilih K")
    
    Xscaled = st.session_state.X_scaled
    selected = st.session_state.selected_features
    
    if Xscaled is None:
        if len(selected) == 0:
            st.warning("⚠️ Belum ada fitur terpilih. Pergi ke 'Preprocessing Data' dan pilih fitur.")
        else:
            st.info("ℹ️ Jalankan preview pra-proses di halaman 'Preprocessing Data' terlebih dahulu.")
        st.stop()
    
    st.markdown("#### 1️⃣ Hitung Metrik Elbow & Silhouette")
    max_k = st.slider("Max K untuk diuji", min_value=3, max_value=12, value=8)
    
    compute = st.button("🔬 Compute Elbow & Silhouette")
    if compute:
        with st.spinner("Menghitung metrik untuk setiap K..."):
            ks, inertias, silhouettes = compute_k_metrics(Xscaled, k_min=2, k_max=max_k, random_state=42)
            suggested_k, method_used = suggest_k(ks, inertias, silhouettes)
            st.session_state.suggested_k = suggested_k
            st.session_state.suggested_method = method_used
            
            st.markdown("<div class='insight'>", unsafe_allow_html=True)
            st.success(f"✅ **Saran K terbaik: {suggested_k}** (metode: {method_used})")
            st.markdown("</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(x=ks, y=inertias, markers=True, title="📈 Elbow (Inertia)", 
                             labels={"x": "K", "y": "Inertia"})
                fig.add_vline(x=suggested_k, line_dash="dash", line_color="red", annotation_text=f"K={suggested_k}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.line(x=ks, y=[s if s is not None else np.nan for s in silhouettes], markers=True, 
                              title="📉 Silhouette Score", labels={"x": "K", "y": "Silhouette"})
                fig2.add_vline(x=suggested_k, line_dash="dash", line_color="red", annotation_text=f"K={suggested_k}")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        if st.session_state.suggested_k is not None:
            st.info(f"ℹ️ Saran K sebelumnya: {st.session_state.suggested_k}")
    
    st.markdown("#### 2️⃣ Pilih K untuk Clustering")
    use_auto = st.checkbox("Gunakan K saran otomatis", value=True)
    if not use_auto:
        k_manual = st.number_input("K manual", min_value=2, max_value=20, value=3, step=1)
        final_k = int(k_manual)
    else:
        if st.session_state.suggested_k is not None:
            final_k = st.session_state.suggested_k
        else:
            final_k = 3
            st.warning("⚠️ Belum ada saran K. Gunakan default K=3 atau compute terlebih dahulu.")
    
    st.write(f"**K yang akan digunakan: {final_k}**")
    
    if st.button("✅ Konfirmasi & Lanjut ke Visualisasi"):
        st.session_state.final_k = final_k
        st.success("✅ K dikonfirmasi. Lanjut ke halaman 'Visualisasi' untuk melihat hasil clustering.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================
# PAGE: VISUALISASI
# ===========================================
elif page == "Visualisasi":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Visualisasi Hasil Clustering")
    
    Xscaled = st.session_state.X_scaled
    dfp = st.session_state.df_cleaned if not st.session_state.df_cleaned.empty else st.session_state.df_raw
    final_k = st.session_state.final_k
    
    if Xscaled is None or final_k is None:
        st.warning("⚠️ Pra-proses atau analisis belum selesai. Silakan kerjakan tahap sebelumnya.")
        st.stop()
    
    st.markdown("#### 1️⃣ Pilih Metode Reduksi Dimensi")
    col_dr = st.columns([1, 2])
    with col_dr[0]:
        dr_method = st.selectbox("Metode", options=["PCA", "t-SNE", "UMAP"] if UMAP_AVAILABLE else ["PCA", "t-SNE"])
    if dr_method == "t-SNE":
        with col_dr[1]:
            tsne_perp = st.slider("t-SNE perplexity", 5, 50, 30)
    else:
        tsne_perp = None
    
    if st.button("🎨 Jalankan Clustering & Visualisasi"):
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
        
        st.success("✅ Clustering selesai.")
        
        # Display
        st.markdown("#### 2️⃣ Visualisasi Klaster (2D)")
        left, right = st.columns([2, 1])
        hover_cols = [c for c in ["uid", "report_date", "victim_race", "victim_age", "victim_sex", "state", "disposition", "lat", "lon"] if c in dfp.columns]
        
        with left:
            fig = px.scatter(dfp, x="_x", y="_y", color=dfp["cluster"].astype(str), 
                           hover_data=hover_cols, title="Visualisasi Klaster (2D)", 
                           labels={"_x": "Dimensi 1", "_y": "Dimensi 2"})
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with right:
            st.markdown("**📊 Ringkasan**")
            st.write(f"**K final:** {final_k}")
            if sil is not None:
                st.write(f"**Silhouette:** {sil:.3f}")
            st.table(dfp["cluster"].value_counts().sort_index().rename("count").to_frame())
        
        # Map if coordinates available
        st.markdown("#### 3️⃣ Peta Geografis")
        lat_col = "lat" if "lat" in dfp.columns else ("latitude" if "latitude" in dfp.columns else None)
        lon_col = "lon" if "lon" in dfp.columns else ("longitude" if "longitude" in dfp.columns else None)
        
        if lat_col and lon_col and dfp[lat_col].notna().sum() > 0:
            df_map = dfp.dropna(subset=[lat_col, lon_col])
            fig_map = px.scatter_mapbox(df_map, lat=lat_col, lon=lon_col, color=df_map["cluster"].astype(str),
                                       hover_data=hover_cols, zoom=10, height=600, title="Distribusi Klaster per Lokasi")
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("ℹ️ Kolom lat/lon tidak ada atau kosong — peta tidak ditampilkan.")
        
        # Sample by cluster
        st.markdown("#### 4️⃣ Contoh Baris per Klaster")
        sel = st.selectbox("Pilih klaster untuk contoh baris", options=sorted(dfp["cluster"].unique()))
        st.dataframe(dfp[dfp["cluster"] == sel].head(25), use_container_width=True)
        
        # Download
        csv = dfp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download dataset berlabel (CSV)", data=csv, file_name="homicide_with_clusters.csv", mime="text/csv")
    
    else:
        st.info("ℹ️ Klik 'Jalankan Clustering & Visualisasi' untuk memulai.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================
# PAGE: HASIL
# ===========================================
elif page == "Hasil":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Kesimpulan & Hasil Analisis")
    
    dfp = st.session_state.df_proc
    labels = st.session_state.cluster_labels
    final_k = st.session_state.final_k
    sil = st.session_state.silhouette_score_val
    
    if dfp.empty or labels is None:
        st.warning("⚠️ Belum ada clustering hasil. Silakan jalankan clustering di halaman 'Visualisasi' terlebih dahulu.")
        st.stop()
    
    st.markdown("#### 📊 Statistik Per Klaster")
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
    
    st.markdown("#### 🔍 Insights & Kesimpulan")
    
    # Generate insights
    for cl in sorted(dfp["cluster"].unique()):
        sub = dfp[dfp["cluster"] == cl]
        pct = 100*len(sub)/len(dfp)
        
        st.markdown(f"<div class='insight'>", unsafe_allow_html=True)
        st.write(f"**Klaster {int(cl)}** ({len(sub)} kasus, {pct:.1f}%)")
        
        insights = []
        if "victim_age" in sub.columns:
            avg_age = sub["victim_age"].dropna().mean()
            if avg_age < 25:
                insights.append("👥 Didominasi korban muda (< 25 tahun) — tinggi risiko di kalangan anak-anak & remaja")
            elif avg_age > 60:
                insights.append("👴 Didominasi korban lansia (> 60 tahun) — kelompok rentan dengan mobilitas terbatas")
            else:
                insights.append(f"👤 Korban dewasa (rata-rata {avg_age:.0f} tahun)")
        
        if "victim_race" in sub.columns:
            top_race = sub["victim_race"].mode()
            if not top_race.empty:
                insights.append(f"🌍 Ras dominan: {top_race.iloc[0]}")
        
        if "victim_sex" in sub.columns:
            top_sex = sub["victim_sex"].mode()
            if not top_sex.empty and top_sex.iloc[0] == "Female":
                insights.append("👩 Tingkat korban perempuan tinggi — perlu program perlindungan khusus")
            elif not top_sex.empty:
                insights.append(f"👨 Jenis kelamin dominan: {top_sex.iloc[0]}")
        
        if "state" in sub.columns:
            top_state = sub["state"].mode()
            if not top_state.empty:
                insights.append(f"📍 Lokasi dominan: {top_state.iloc[0]}")
        
        if "disposition" in sub.columns:
            closed_rate = (sub["disposition"].str.contains("Closed", case=False).sum() / len(sub)) * 100
            insights.append(f"📋 Tingkat penyelesaian kasus: {closed_rate:.1f}%")
        
        for insight in insights:
            st.write(f"- {insight}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("#### 💡 Rekomendasi Kebijakan")
    st.markdown("""
    Berdasarkan hasil clustering, berikut rekomendasi untuk pencegahan yang lebih efektif:
    
    1. **Identifikasi Klaster Berisiko Tinggi**: Fokus pada klaster dengan jumlah kasus terbanyak atau tingkat penyelesaian rendah
    2. **Program Perlindungan Khusus**: Untuk kelompok korban tertentu (perempuan, lansia, anak-anak)
    3. **Penguatan Pengawasan Wilayah**: Tingkatkan patroli di lokasi-lokasi dengan konsentrasi kasus tinggi
    4. **Edukasi Masyarakat**: Berbeda untuk setiap segmen demografis klaster
    5. **Alokasi Sumber Daya**: Distribusikan personel & anggaran berdasarkan risiko per klaster
    """)
    
    st.markdown("#### 📊 Metrik Kualitas Clustering")
    if sil is not None:
        st.write(f"**Silhouette Score**: {sil:.3f}")
        if sil > 0.5:
            st.success("✅ Clustering kualitas baik (silhouette > 0.5)")
        elif sil > 0.3:
            st.info("ℹ️ Clustering kualitas sedang (0.3 < silhouette ≤ 0.5)")
        else:
            st.warning("⚠️ Clustering kualitas kurang (silhouette ≤ 0.3)")
    
    st.markdown("</div>", unsafe_allow_html=True)