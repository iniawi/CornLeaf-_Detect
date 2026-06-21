"""
SiCorn — Klasifikasi Penyakit Daun Jagung
Landing page Streamlit satu halaman dengan navigasi scroll, untuk model
MobileNetV2 (model_mobilenetv2_daun_jagung.keras).
"""

import random

import numpy as np
import streamlit as st
from PIL import Image

# =============================================================================
# KONFIGURASI
# =============================================================================

st.set_page_config(
    page_title="SiCorn — Klasifikasi Penyakit Daun Jagung",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "model_mobilenetv2_daun_jagung.keras"
FALLBACK_IMG_SIZE = (224, 224)

PREPROCESS_MODE = "rescale"

CLASS_INFO = [
    {
        "label": "Karat Daun Jagung",
        "latin": "Puccinia sorghi",
        "condition": "Karat Daun disebabkan oleh jamur Puccinia sorghi, ditandai bintik-bintik "
        "kecil berwarna coklat kemerahan menyerupai karat di permukaan daun.",
        "treatment": "Aplikasikan fungisida berbahan aktif triazol, buang daun yang terserang "
        "berat, dan hindari kelembapan berlebih di sekitar tanaman.",
        "tone": "warning",
        "icon": "🟠",
    },
    {
        "label": "Hawar Daun Jagung",
        "latin": "Exserohilum turcicum",
        "condition": "Hawar Daun Utara disebabkan oleh jamur Exserohilum turcicum, ditandai "
        "bercak panjang berbentuk cerutu berwarna coklat keabu-abuan.",
        "treatment": "Lakukan penyemprotan fungisida sistemik, gunakan varietas tahan, dan jaga "
        "jarak tanam agar sirkulasi udara tetap baik.",
        "tone": "warning",
        "icon": "🟠",
    },
    {
        "label": "Bercak Daun Cercospora",
        "latin": "Cercospora zeae-maydis",
        "condition": "Bercak Daun Abu-abu disebabkan oleh jamur Cercospora zeae-maydis, dengan "
        "gejala bercak persegi panjang berwarna abu-abu hingga coklat di antara tulang daun.",
        "treatment": "Lakukan rotasi tanaman dengan non-inang, kelola sisa tanaman setelah "
        "panen, dan gunakan fungisida bila serangan meluas.",
        "tone": "warning",
        "icon": "🟤",
    },
    {
        "label": "Daun Sehat",
        "latin": "Tidak terdeteksi penyakit",
        "condition": "Daun jagung tidak menunjukkan gejala penyakit yang signifikan. Pertahankan "
        "praktik budidaya yang baik agar tanaman tetap produktif.",
        "treatment": "Lanjutkan penyiraman dan pemupukan teratur, serta lakukan pemantauan "
        "rutin setiap minggu untuk deteksi dini.",
        "tone": "success",
        "icon": "🟢",
    },
]

# =============================================================================
# FUNGSI BANTU — MODEL
# =============================================================================


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as exc:
        return None, str(exc)


def get_input_size(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h, w = shape[1], shape[2]
        if h and w:
            return (int(w), int(h))
    except Exception:
        pass
    return FALLBACK_IMG_SIZE


def predict_image(model, image: Image.Image):
    size = get_input_size(model)
    img = image.convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32")

    if PREPROCESS_MODE == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100
    return idx, confidence, preds


# =============================================================================
# HELPER — encode gambar ke base64 agar bisa dipakai di CSS url()
# =============================================================================

import base64
import os

def get_base64_of_bin_file(path: str) -> str:
    """Baca file biner dan kembalikan string base64-nya."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_bg_b64 = get_base64_of_bin_file("static/field_bg.jpg")
# Tentukan tipe MIME berdasarkan ekstensi
_bg_ext  = "jpeg"  # jpg → jpeg
_bg_url  = f"data:image/{_bg_ext};base64,{_bg_b64}" if _bg_b64 else ""

# =============================================================================
# CSS
# =============================================================================

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{{
  --ink:#16261c;
  --gray-600:#5c6b5e;
  --gray-400:#94a398;
  --green-dark:#14532d;
  --green-primary:#2f7a45;
  --green-primary-hover:#256138;
  --green-light:#e8f5e1;
  --green-pale:#f4faf0;
  --gold:#d6a429;
  --gold-light:#fdf2c9;
  --gold-dark:#a8780f;
  --cream:#fbfaf3;
  --white:#ffffff;
  --border:#e7ede4;
  --shadow-md: 0 12px 28px -10px rgba(20,60,30,0.16);
  --shadow-lg: 0 24px 48px -16px rgba(20,60,30,0.20);
  --radius-lg:24px;
  --radius-md:16px;
  --radius-sm:10px;
  --navbar-h:64px;
}}

html{{ scroll-behavior:smooth; }}
[id]{{ scroll-margin-top:calc(var(--navbar-h) + 16px); }}

/* ---- Streamlit chrome cleanup ---- */
#MainMenu{{ visibility:hidden; }}
footer{{ visibility:hidden; }}
header[data-testid="stHeader"]{{
  display:none !important;
  visibility:hidden !important;
  height:0 !important;
}}
.block-container{{
  padding:0 !important;
  max-width:100% !important;
}}

/* Hapus semua padding atas bawaan Streamlit agar navbar nempel ke top */
div[data-testid="stAppViewBlockContainer"]{{
  padding-top:0 !important;
  margin-top:0 !important;
}}
div[data-testid="stAppViewContainer"]{{
  padding-top:0 !important;
}}
section[data-testid="stMain"] > div:first-child{{
  padding-top:0 !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]{{ width:100%; }}

.stApp{{
  background:
    radial-gradient(900px 420px at 85% 0%, rgba(214,164,41,0.10), transparent 60%),
    linear-gradient(180deg, var(--cream) 0%, var(--green-pale) 45%, var(--cream) 100%);
  font-family:'Inter', sans-serif;
  color:var(--ink);
}}

h1,h2,h3,h4{{ font-family:'Plus Jakarta Sans', sans-serif; color:var(--ink); margin:0;  }}
p{{ color:var(--gray-600); line-height:1.65; }}

.container{{ max-width:1140px; margin:0 auto; padding:0 32px; }}

/* ============================================================
   NAVBAR — position:fixed + padding-top pada konten agar
   tidak tertutup navbar
   ============================================================ */
.navbar{{
  position:fixed;
  top:0;
  left:0;
  right:0;
  width:100%;
  height:var(--navbar-h);
  z-index:9999;
  background:rgba(255,255,255,0.95);
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
  box-sizing:border-box;
}}
.nav-inner{{
  max-width:1140px;
  margin:0 auto;
  padding:0 32px;
  height:100%;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:24px;
}}

/* Dorong seluruh konten ke bawah agar tidak ketutupan navbar fixed */
div[data-testid="stAppViewBlockContainer"]{{
  padding-top:var(--navbar-h) !important;
}}

.logo{{ font-family:'Plus Jakarta Sans', sans-serif; font-weight:800; font-size:1.3rem; display:flex; align-items:center; gap:6px; white-space:nowrap; text-decoration:none !important; }}
.logo .si{{ color:var(--ink); }}
.logo .corn{{ color:var(--green-primary); }}
.nav-links{{ display:flex; align-items:center; gap:30px; flex-wrap:wrap; }}
.nav-links a.nav-link{{
  color:var(--ink);
  text-decoration:none !important;
  font-weight:600;
  font-size:0.95rem;
  opacity:0.8;
  transition:opacity .15s ease, color .15s ease;
}}
.nav-links a.nav-link:hover{{ opacity:1; color:var(--green-primary); }}

/* ---- Tombol Primary — tanpa garis bawah sama sekali ---- */
.btn-primary{{
  display:inline-flex;
  align-items:center;
  gap:8px;
  background:var(--green-primary);
  color:#fff !important;
  text-decoration:none !important;
  font-weight:700;
  font-size:0.92rem;
  padding:11px 22px;
  border-radius:999px;
  box-shadow:var(--shadow-md);
  transition:transform .15s ease, background .15s ease;
  white-space:nowrap;
  border:none;
  cursor:pointer;
}}
.btn-primary:hover,
.btn-primary:visited,
.btn-primary:active{{
  background:var(--green-primary-hover);
  transform:translateY(-1px);
  text-decoration:none !important;
  color:#fff !important;
}}
.btn-large{{ padding:15px 30px; font-size:1rem; }}

/* ============================================================
   HERO — background dari field_bg.jpg (di-embed sebagai
   base64 agar tidak tergantung pada URL Streamlit static)
   ============================================================ */
.hero{{
  position:relative;
  display: flex;
  flex-direction: column;
  justify-content: center; /* Tengah vertikal */
  align-items: center;
  background-image:url("{_bg_url}");
  background-size:cover;
  background-position:center center;
  background-repeat:no-repeat;
  min-height: 80vh;        
  padding: 100px 20px;
}}

/* Overlay agar teks terbaca di atas foto ladang */
.hero::before{{
  content:'';
  position:absolute;
  inset:0;
  background:linear-gradient(
    180deg,
    rgba(251,250,243,0.88) 0%,
    rgba(244,250,240,0.78) 50%,
    rgba(251,250,243,0.97) 88%,
    rgba(251,250,243,1.00) 100%
  );
  pointer-events:none;
  z-index:0;
}}

/* Konten hero harus di atas overlay */
.hero .container,
.hero .hero-illustration-wrap{{
  position:relative;
  z-index:1;
}}

.hero-badge{{
  display:inline-flex; align-items:center; gap:6px;
  background:var(--gold-light); color:var(--gold-dark); font-weight:700; font-size:0.82rem;
  padding:7px 16px; border-radius:999px; margin-bottom:22px;
}}
.hero-title{{ font-size:2.9rem; line-height:1.18; font-weight:800; text-align:center; letter-spacing:-0.5px; }}
.hero-gradient{{
  background:linear-gradient(90deg, var(--green-dark), var(--gold));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}}
.hero-subtitle{{ max-width:620px; margin:20px auto 0; text-align:center; font-size:1.08rem; }}
.hero-cta-row{{ display:flex; justify-content:center; margin:30px 0 50px; }}

/* ---- Section headers (shared) ---- */
.section{{ padding:84px 0; }}
.section-eyebrow{{
  display:inline-flex; align-items:center; gap:6px; background:var(--green-light); color:var(--green-dark);
  font-weight:700; font-size:0.8rem; padding:6px 14px; border-radius:999px; margin-bottom:14px;
}}
.section-title{{ font-size:2.1rem; font-weight:800; }}
.section-subtitle{{ max-width:560px; margin-top:12px; font-size:1.02rem; }}
.text-center{{ text-align:center; }}
.mx-auto{{ margin-left:auto; margin-right:auto; }}

/* ---- Analisis section ---- */

.text-center.section-subtitle{{
  margin-left:auto !important;
  margin-right:auto !important;
  text-align:center !important;
}}
.st-key-analisis_wrap{{
  max-width:1140px; margin:0 auto; padding:84px 32px 90px;
}}
.window-chrome{{
  display:flex; gap:7px; padding:14px 18px; background:#eef3ea; border-radius:16px 16px 0 0;
  margin-top:34px; border:1px solid var(--border); border-bottom:none;
}}
.window-chrome .dot{{ width:11px; height:11px; border-radius:50%; }}
.dot.red{{ background:#f06a5e; }} .dot.yellow{{ background:#f0c259; }} .dot.green{{ background:#74c573; }}

.st-key-upload_card, .st-key-result_card{{
  background:var(--white); border:1px solid var(--border); border-top:none;
  border-radius:0 0 var(--radius-md) var(--radius-md);
  padding:34px 30px !important; min-height:330px;
  box-shadow:var(--shadow-md);
}}
.st-key-upload_card{{ text-align:center; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
.upload-icon{{
  width:52px; height:52px; border-radius:50%; background:var(--green-light); color:var(--green-primary);
  display:flex; align-items:center; justify-content:center; font-size:1.4rem; margin:0 auto 16px;
}}
.upload-title{{ font-size:1.3rem; font-weight:700; margin-bottom:6px; }}
.upload-hint{{ font-size:0.88rem; color:var(--gray-400); margin-bottom:18px; }}

div[data-testid="stFileUploaderDropzone"]{{
  background:var(--green-pale) !important; border:1.5px dashed #b7d8b8 !important; border-radius:14px !important;
}}
div[data-testid="stFileUploaderDropzone"] button{{
  background:var(--green-primary) !important; color:#fff !important; border-radius:999px !important;
  border:none !important; font-weight:600 !important;
}}
div[data-testid="stFileUploaderDropzone"] button:hover{{ background:var(--green-primary-hover) !important; }}

.result-placeholder{{ text-align:center; color:var(--gray-400); padding:30px 10px; }}
.result-placeholder .ph-icon{{ font-size:2.4rem; margin-bottom:10px; }}
.result-card-img{{ border-radius:12px; overflow:hidden; margin-bottom:16px; }}
.result-badge{{
  display:inline-flex; align-items:center; gap:6px; font-size:0.78rem; font-weight:700;
  padding:5px 12px; border-radius:999px; margin-bottom:10px;
}}
.result-badge.tone-success{{ background:#e3f6e6; color:#1f7a3d; }}
.result-badge.tone-warning{{ background:#fdeee0; color:#b85c1f; }}
.result-label{{ font-size:1.35rem; font-weight:800; margin-bottom:14px; }}
.confidence-row{{ display:flex; align-items:center; gap:12px; margin-bottom:18px; }}
.confidence-bar-bg{{ flex:1; height:9px; background:#eef1ea; border-radius:999px; overflow:hidden; }}
.confidence-bar-fill{{ height:100%; border-radius:999px; }}
.confidence-bar-fill.tone-success{{ background:linear-gradient(90deg,#3fae5c,#74c573); }}
.confidence-bar-fill.tone-warning{{ background:linear-gradient(90deg,#d97b2b,#e8a951); }}
.confidence-text{{ font-weight:800; font-size:0.95rem; min-width:42px; }}
.result-detail{{ font-size:0.92rem; margin-bottom:10px; }}
.result-detail b{{ color:var(--ink); }}
.model-warning{{
  background:#fdf2e9; border:1px solid #f4d9bb; color:#8a5524; border-radius:12px;
  padding:16px 18px; font-size:0.9rem; text-align:left;
}}

* ---- Fitur ---- */
.features-section{{ background:transparent; }}
.features-section .section-subtitle{{
  margin-left:auto !important;
  margin-right:auto !important;
  text-align:center !important;
}}
.features-grid{{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:22px;
  margin-top:44px;
  text-align:left;
}}
.feature-card{{
  background:var(--white); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:28px 22px; box-shadow:var(--shadow-md); transition:transform .18s ease;
  display:flex; flex-direction:column;
}}
.feature-card:hover{{ transform:translateY(-4px); }}
.feature-icon{{
  width:48px; height:48px; border-radius:14px; background:var(--gold-light);
  display:flex; align-items:center; justify-content:center; font-size:1.3rem; margin-bottom:16px;
  flex-shrink:0;
}}
.feature-card h3{{ font-size:1.08rem; margin-bottom:8px; }}
.feature-card p{{ font-size:0.92rem; margin:0; line-height:1.6; }}
 
/* ---- Cara Kerja ---- */
.steps-section{{ background:linear-gradient(180deg, transparent, var(--green-pale) 40%, transparent); }}
.steps-section .section-subtitle{{
  margin-left:auto !important;
  margin-right:auto !important;
  text-align:center !important;
}}
.steps-grid{{ display:grid; grid-template-columns:repeat(3,1fr); gap:26px; margin-top:46px; }}
.step-card{{ position:relative; background:var(--white); border:1px solid var(--border); border-radius:var(--radius-md); padding:32px 26px 26px; box-shadow:var(--shadow-md); }}
.step-number{{
  position:absolute; top:-18px; left:26px; width:36px; height:36px; border-radius:50%;
  background:var(--green-primary); color:#fff; font-weight:800; font-size:0.95rem;
  display:flex; align-items:center; justify-content:center; box-shadow:var(--shadow-md);
}}
.step-card h3{{ font-size:1.08rem; margin:10px 0 8px; }}
.step-card p{{ font-size:0.92rem; margin:0; }}

/* ---- Tentang ---- */
.tentang-grid{{ display:grid; grid-template-columns:1fr 1fr; gap:50px; align-items:center; margin-top:36px; }}
.tentang-stats{{ display:flex; flex-wrap:wrap; gap:14px; margin-top:26px; }}
.stat-pill{{
  background:var(--white); border:1px solid var(--border); border-radius:999px;
  padding:9px 16px; font-size:0.85rem; font-weight:600; color:var(--green-dark);
  display:inline-flex; align-items:center; gap:6px; box-shadow:var(--shadow-md);
}}
.tentang-card{{
  background:var(--white); border:1px solid var(--border); border-radius:var(--radius-lg);
  padding:34px; box-shadow:var(--shadow-lg);
}}
.tentang-card .big-emoji{{ font-size:2.6rem; margin-bottom:10px; }}

/* ---- Footer ---- */
.footer{{ border-top:1px solid var(--border); margin-top:20px; }}
.footer-inner{{
  display:flex; align-items:center; justify-content:space-between; padding:30px 32px; flex-wrap:wrap; gap:12px;
}}
.footer-inner .logo{{ font-size:1.05rem; }}
.footer-links{{ display:flex; gap:18px; color:var(--gray-600); font-size:0.88rem; }}
.footer-links a{{ color:var(--gray-600); text-decoration:none !important; }}
.footer-copy{{ color:var(--gray-400); font-size:0.85rem; }}

/* ---- Responsive ---- */
@media (max-width:900px){{
  .hero-title{{ font-size:2.1rem; }}
  .nav-links{{ gap:14px; }}
  .nav-links a.nav-link{{ font-size:0.85rem; }}
  .features-grid{{ grid-template-columns:repeat(2,1fr); }}
  .steps-grid{{ grid-template-columns:1fr; }}
  .tentang-grid{{ grid-template-columns:1fr; }}
}}
@media (max-width:600px){{
  .nav-links{{ display:none; }}
  .features-grid{{ grid-template-columns:1fr; }}
  .hero-title{{ font-size:1.7rem; }}
  .section{{ padding:56px 0; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# NAVBAR
# =============================================================================

st.markdown(
    """
<div class="navbar">
  <div class="nav-inner">
    <div class="logo"><span>🌽</span><span class="si">Si</span><span class="corn">Corn</span></div>
    <div class="nav-links">
      <a class="nav-link" href="#beranda">Beranda</a>
      <a class="nav-link" href="#fitur">Fitur</a>
      <a class="nav-link" href="#cara-kerja">Cara Kerja</a>
      <a class="nav-link" href="#tentang">Tentang Kami</a>
      <a class="btn-primary" href="#analisis">Analisis Sekarang</a>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# HERO
# =============================================================================

st.markdown(
    f"""
<section id="beranda" class="hero">
  <div class="container text-center">
    <h1 class="hero-title">Cegah Kegagalan Panen Jagung<br/>dengan <span class="hero-gradient">SiCorn</span></h1>
    <p class="hero-subtitle">Klasifikasi penyakit daun jagung secara instan dan akurat
    menggunakan kecerdasan buatan, langsung dari foto yang Anda ambil di ladang.</p>
    <div class="hero-cta-row">
      <a class="btn-primary btn-large" href="#analisis">Mulai Analisis Daun Jagung →</a>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# ANALISIS — bagian fungsional (upload + prediksi model)
# =============================================================================

with st.container(key="analisis_wrap"):
    st.markdown('<div id="analisis"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="text-center">
          <h2 class="section-title">Coba SiCorn Sekarang</h2>
          <p class="section-subtitle" style="margin:12px auto 0; text-align:center; display:block;">Unggah foto daun jagung Anda dan dapatkan
          diagnosis beserta rekomendasi penanganan dalam hitungan detik.</p>
        </div>
        <div class="window-chrome">
          <span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.05], gap="medium")

    with col1:
        with st.container(key="upload_card"):
            st.markdown(
                """
                <div class="upload-icon">⬆️</div>
                <div class="upload-title" style="margin:12px auto 0; text-align:center; display:block;">Unggah Foto Daun Jagung Anda Disini</div>
                <div class="upload-hint" style="margin:12px auto 0; text-align:center; display:block;" >Format JPG, JPEG, atau PNG</div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Pilih File",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

    with col2:
        with st.container(key="result_card"):
            if uploaded_file is None:
                st.markdown(
                    """
                    <div class="result-placeholder">
                      <div class="ph-icon">🌽</div>
                      <p>Hasil analisis akan muncul di sini setelah Anda<br/>
                      mengunggah foto daun jagung.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                try:
                    image = Image.open(uploaded_file)
                    model, err = load_model()

                    if model is None:
                        st.image(image, use_container_width=True)
                        st.markdown(
                            f"""
                            <div class="model-warning">
                              ⚠️ <b>Model belum ditemukan.</b><br/>
                              Pastikan file <code>{MODEL_PATH}</code> berada di folder yang
                              sama dengan <code>app.py</code>, lalu jalankan ulang aplikasi.
                              <br/><br/><i>Detail teknis: {err}</i>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        with st.spinner("Menganalisis daun jagung..."):
                            idx, confidence, preds = predict_image(model, image)

                        st.image(image, use_container_width=True)

                        with st.expander("🔍 Lihat detail probabilitas tiap kelas (debug)"):
                            for i, p in enumerate(preds):
                                nm = CLASS_INFO[i]["label"] if i < len(CLASS_INFO) else f"Kelas {i}"
                                st.write(f"`indeks {i}` — {nm}: **{p * 100:.2f}%**")

                        if idx < len(CLASS_INFO):
                            info = CLASS_INFO[idx]
                            tone = "tone-success" if info["tone"] == "success" else "tone-warning"
                            st.markdown(
                                f"""
                                <span class="result-badge {tone}">{info['icon']} Terdeteksi</span>
                                <div class="result-label">{info['label']}</div>
                                <div class="confidence-row">
                                  <div class="confidence-bar-bg">
                                    <div class="confidence-bar-fill {tone}" style="width:{confidence:.0f}%;"></div>
                                  </div>
                                  <span class="confidence-text">{confidence:.0f}%</span>
                                </div>
                                <p class="result-detail"><b>Kondisi:</b> {info['condition']}</p>
                                <p class="result-detail"><b>Tindakan:</b> {info['treatment']}</p>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="model-warning">
                                  ⚠️ Jumlah kelas keluaran model ({len(preds)}) tidak sesuai
                                  dengan daftar <code>CLASS_INFO</code> ({len(CLASS_INFO)}).
                                  Silakan sesuaikan konfigurasi kelas di <code>app.py</code>.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                except Exception as exc:
                    st.markdown(
                        f"""
                        <div class="model-warning">
                          ⚠️ Terjadi kesalahan saat memproses gambar.<br/>
                          <i>Detail teknis: {exc}</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# =============================================================================
# FITUR
# =============================================================================
 
st.markdown(
    """
<section id="fitur" class="section features-section">
  <div class="container text-center">
    <h2 class="section-title">Fitur Unggulan</h2>
    <p class="section-subtitle">Dirancang khusus untuk membantu petani jagung
    mengambil keputusan lebih cepat dan tepat.</p>
    <div class="features-grid">
      <div class="feature-card">
        <div class="feature-icon">⏱️</div>
        <h3>Deteksi Dini Cepat</h3>
        <p>Klasifikasi penyakit daun jagung hanya dalam hitungan detik, sehingga deteksi
        dapat dilakukan sejak gejala awal muncul.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <h3>Akurasi Tinggi AI</h3>
        <p>Ditenagai model MobileNetV2 yang dilatih khusus untuk mengenali pola penyakit
        pada citra daun jagung secara konsisten.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🌿</div>
        <h3>Rekomendasi Penanganan</h3>
        <p>Setiap hasil disertai kondisi dan langkah penanganan praktis yang dapat
        langsung diterapkan di lapangan.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon">📱</div>
        <h3>Akses Kapan Saja</h3>
        <p>Cukup unggah foto dari ponsel atau komputer — tanpa instalasi rumit, kapan
        pun Anda membutuhkannya.</p>
      </div>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)
 
# =============================================================================
# CARA KERJA
# =============================================================================

st.markdown(
    """
<section id="cara-kerja" class="section steps-section">
  <div class="container text-center">
    <h2 class="section-title">Cara Kerja SiCorn</h2>
    <p class="section-subtitle mx-auto">Tiga langkah sederhana dari foto hingga
    rekomendasi penanganan.</p>
  </div>
  <div class="container steps-grid">
    <div class="step-card">
      <div class="step-number">01</div>
      <h3>Unggah Foto Daun</h3>
      <p>Ambil atau unggah foto daun jagung yang ingin diperiksa langsung melalui
      halaman analisis.</p>
    </div>
    <div class="step-card">
      <div class="step-number">02</div>
      <h3>AI Menganalisis Citra</h3>
      <p>Model MobileNetV2 memproses gambar dan mengenali pola visual khas tiap
      jenis penyakit daun jagung.</p>
    </div>
    <div class="step-card">
      <div class="step-number">03</div>
      <h3>Dapatkan Hasil & Rekomendasi</h3>
      <p>Hasil klasifikasi, tingkat keyakinan, serta rekomendasi penanganan
      ditampilkan secara instan.</p>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# TENTANG KAMI
# =============================================================================

st.markdown(
    """
<section id="tentang" class="section">
  <div class="container">
    <div class="tentang-grid">
      <div>
        <span class="section-eyebrow">🌽 Tentang Kami</span>
        <h2 class="section-title">Dibangun untuk Petani Jagung Indonesia</h2>
        <p style="margin-top:16px; font-size:1rem;">SiCorn adalah aplikasi klasifikasi
        penyakit daun jagung berbasis kecerdasan buatan. Kami percaya deteksi dini
        adalah kunci untuk mencegah kegagalan panen, dan teknologi seharusnya
        mudah diakses oleh siapa pun yang bekerja di ladang.</p>
        <p style="font-size:1rem;">Model yang digunakan dilatih dengan arsitektur
        MobileNetV2 menggunakan teknik transfer learning, sehingga ringan namun
        tetap mampu mengenali pola visual penyakit secara konsisten.</p>
        <div class="tentang-stats">
          <span class="stat-pill">🧠 Model MobileNetV2</span>
          <span class="stat-pill">🍃 4 Kategori Klasifikasi</span>
          <span class="stat-pill">⚡ Hasil Instan</span>
        </div>
      </div>
      <div class="tentang-card">
        <div class="big-emoji">🌽🤖</div>
        <h3 style="margin-bottom:10px;">Misi Kami</h3>
        <p>Membantu petani mengambil keputusan lebih cepat melalui teknologi AI
        yang sederhana, sehingga kerugian akibat penyakit tanaman dapat ditekan
        sedini mungkin.</p>
      </div>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
<div class="footer">
  <div class="footer-inner">
    <div class="logo"><span>🌽</span><span class="si">Si</span><span class="corn">Corn</span></div>
    <div class="footer-links">
      <a href="#beranda">Beranda</a>
      <a href="#fitur">Fitur</a>
      <a href="#cara-kerja">Cara Kerja</a>
      <a href="#tentang">Tentang Kami</a>
    </div>
    <div class="footer-copy">© 2026 SiCorn — Klasifikasi Penyakit Daun Jagung</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)