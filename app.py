"""
BerryVision - Klasifikasi jenis berry
Proyek Akhir Praktikum Kecerdasan Buatan

"""

import base64
import io
import json
import os
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from info import BERRY_INFO, PER_CLASS, ROBUSTNESS, TEST_ACC, TEST_F1

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "berry_cnn.keras"
CLASS_PATH = BASE_DIR / "class_names.json"
IMG_SIZE = 128            
LOW_CONF = 0.60           
HIGH_CONF = 0.80
GRADCAM_LAYER = "last_conv_relu"
TEAM_NAME = "Proyek Akhir Praktikum Kecerdasan Buatan"   

st.set_page_config(page_title="BerryVision", page_icon="🫐", layout="wide")


def info_for(name):
    """Info kelas; kalau nama kelas tidak dikenal pakai nilai default."""
    return BERRY_INFO.get(name, {"label": name.title(), "emoji": "🍇", "color": "#c084fc",
                                 "latin": "-", "desc": "", "ciri": []})


def html(s):
    # baris dirapikan dulu supaya tidak dibaca markdown sebagai code block
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# ----------------------------- Model ----------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_assets():
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH, compile=False)
    class_names = json.load(open(CLASS_PATH, encoding="utf-8"))
    try:
        grad_model = keras.Model(model.inputs, [model.get_layer(GRADCAM_LAYER).output, model.output])
    except Exception:
        grad_model = None   # Grad-CAM dimatikan kalau layer tidak ketemu
    return model, grad_model, class_names


def to_model_input(img):
    """Resize sama persis seperti training. Normalisasi 1/255 sudah ada di dalam model."""
    import tensorflow as tf
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return tf.image.resize(arr, (IMG_SIZE, IMG_SIZE)).numpy()[None, ...]


def gradcam(grad_model, x, class_idx):
    import tensorflow as tf
    with tf.GradientTape() as tape:
        fmap, pred = grad_model(tf.convert_to_tensor(x, tf.float32), training=False)
        score = pred[:, class_idx]
    grads = tape.gradient(score, fmap)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = Image.fromarray((cam * 255).astype("uint8")).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(cam, dtype=np.float32) / 255.0


# ----------------------------- Gambar ---------------------------------------
def load_image(file):
    img = ImageOps.exif_transpose(Image.open(file))      # betulkan rotasi foto HP
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def to_data_uri(img, max_side=720):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay_cam(img, cam, alpha=0.45):
    """Tempel heatmap (colormap jet) di atas foto."""
    base = img.copy()
    base.thumbnail((720, 720))
    c = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(base.size, Image.BICUBIC),
                   dtype=np.float32) / 255.0
    heat = np.stack([np.clip(1.5 - np.abs(4 * c - k), 0, 1) for k in (3, 2, 1)], axis=-1) * 255
    out = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * heat
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


def quality_report(img):
    """Cek kualitas foto. Blur dicek karena itu kelemahan terbesar model.
    Ambang batas diambil dari variance Laplacian pada gambar dataset (tajam ~1500, buram ~35)."""
    g = np.asarray(img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    sharp, bright, (w, h) = float(lap.var()), float(g.mean()), img.size

    blur = ("bad", "Foto buram") if sharp < 150 else ("warn", "Kurang tajam") if sharp < 400 else ("good", "Foto tajam")
    light = ("warn", "Terlalu gelap") if bright < 60 else ("warn", "Terlalu terang") if bright > 205 else ("good", "Cahaya oke")
    size = ("warn" if min(w, h) < 96 else "good", f"{w}×{h}px")
    return {"blur": blur, "light": light, "size": size}


# ----------------------------- Tampilan -------------------------------------
# Definisi baru (aman dari error, otomatis menyesuaikan)
def callout(kind, icon, text=None):
    if text is None:
        text = icon
        icon = "❌" if kind == "bad" else "⚠️"
    html(f'<div class="callout {kind}"><span class="ic">{icon}</span><div>{text}</div></div>')

def render_hero(class_names):
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:{info_for(c)["color"]}"></span>'
        f'{escape(info_for(c)["label"])}</span>' for c in class_names)
    html(f"""
        <div class="hero">
          <span class="eyebrow">Deep Learning · Custom CNN</span>
          <h1>Berry<span>Vision</span></h1>
          <p>Unggah gambar buah berry untuk mendeteksi jenisnya secara otomatis menggunakan model Convolutional Neural Network (CNN).
          Ada Grad-CAM untuk melihat bagian gambar yang jadi bahan pertimbangan model.</p>
          <div class="chips">{chips}</div>
        </div>""")


def render_empty():
    html("""
        <div class="empty">
          <div class="big">🫐</div>
          <h3>Belum ada foto</h3>
          <p>Upload foto atau ambil lewat kamera di sebelah kiri, hasilnya muncul di sini.</p>
          <div class="tips">
            <div class="tip"><b>Foto yang tajam</b><span>Foto buram bikin akurasi model turun jauh.</span></div>
            <div class="tip"><b>Cahaya cukup</b><span>Jangan terlalu gelap atau terlalu terang.</span></div>
            <div class="tip"><b>Buah kelihatan jelas</b><span>Usahakan buahnya mengisi sebagian besar gambar.</span></div>
          </div>
        </div>""")


def render_result(class_names, probs, blurry=False):
    order = np.argsort(probs)[::-1]
    top, runner = int(order[0]), int(order[1])
    meta, conf = info_for(class_names[top]), float(probs[top])
    runner_label = info_for(class_names[runner])["label"]

    if blurry:
        badge = '<span class="badge low">Foto buram</span>'
    elif conf >= HIGH_CONF:
        badge = '<span class="badge ok">Yakin</span>'
    elif conf >= LOW_CONF:
        badge = '<span class="badge mid">Cukup yakin</span>'
    else:
        badge = '<span class="badge low">Kurang yakin</span>'

    html(f"""
        <div class="result">
          <div class="glow" style="background:{meta['color']}"></div>
          <div class="kicker">Hasil prediksi</div>
          <div class="name"><div class="emoji">{meta['emoji']}</div>{escape(meta['label'])}</div>
          <div class="latin">{escape(meta['latin'])}</div>
          <div class="conf-row"><div class="conf-num">{conf*100:.1f}<small>%</small></div>{badge}</div>
          <div class="meter"><div style="width:{conf*100:.1f}%;background:linear-gradient(90deg,{meta['color']},#ffffff55)"></div></div>
        </div>""")

    # catatan sesuai kondisi hasil
    pair = {class_names[top], class_names[runner]} == {"Tayberry", "Wineberry"}
    if blurry:
        callout("bad", "<b>Fotonya buram.</b> Waktu diuji, akurasi model turun sekitar 36 poin untuk gambar buram, "
                "jadi hasil ini kurang bisa dipercaya. Coba foto ulang dengan fokus yang lebih tajam.")
    elif conf < LOW_CONF:
        callout("bad", f"<b>Model kurang yakin</b> (di bawah {int(LOW_CONF*100)}%). Coba foto yang lebih tajam dan buahnya "
                "lebih jelas. Ingat juga, jadi objek lain pasti ditebak jadi salah satunya.")
    elif pair and probs[runner] >= 0.25:
        callout("warn", f"<b>Tayberry dan Wineberry memang mirip.</b> Model sempat ragu antara keduanya "
                f"({escape(runner_label)} {probs[runner]*100:.0f}%). Dua kelas ini juga yang paling sering tertukar waktu evaluasi.")
    elif probs[runner] >= 0.30:
        callout("warn", f"Tebakan kedua cukup dekat: <b>{escape(runner_label)}</b> ({probs[runner]*100:.0f}%).")

    bars = ""
    for rank, i in enumerate(order):
        m, p = info_for(class_names[i]), float(probs[i]) * 100
        bars += (f'<div class="prob {"first" if rank == 0 else ""}"><div class="top"><span>{m["emoji"]} {escape(m["label"])}</span>'
                 f'<span>{p:.1f}%</span></div><div class="bar"><div style="width:{max(p, 0.6):.1f}%;background:{m["color"]}"></div></div></div>')
    html(f'<div class="card"><div class="sec-title">Peluang tiap kelas</div>{bars}</div>')


def render_gradcam(img, cam, class_names, class_idx):
    name = escape(info_for(class_names[class_idx])["label"])
    html(f"""
        <div class="card"><h3>Grad-CAM: bagian gambar yang dilihat model</h3>
        <div class="sub">Warna merah = bagian yang paling berpengaruh ke prediksi <b>{name}</b>.
        Dari sini bisa dicek apakah model benar-benar melihat buahnya, atau malah latar belakangnya.</div></div>""")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(f'<img class="photo" src="{to_data_uri(img)}"><div class="photo-cap">Foto asli</div>')
    with c2:
        html(f'<img class="photo" src="{to_data_uri(overlay_cam(img, cam))}"><div class="photo-cap">Grad-CAM</div>')


def render_about():
    html(f"""
        <div class="metrics">
          <div class="metric"><div class="v">{TEST_ACC:.2f}%</div><div class="l">Akurasi data test</div></div>
          <div class="metric"><div class="v">{TEST_F1:.2f}%</div><div class="l">Macro-F1</div></div>
          <div class="metric"><div class="v">2.519</div><div class="l">Gambar, 5 kelas</div></div>
          <div class="metric"><div class="v">{IMG_SIZE}×{IMG_SIZE}</div><div class="l">Ukuran input</div></div>
        </div>""")

    left, right = st.columns(2, gap="large")
    with left:
        rows = "".join(
            f'<tr><td><span class="dot" style="background:{info_for(n)["color"]};margin-right:8px"></span>{escape(info_for(n)["label"])}</td>'
            f'<td class="num">{p:.3f}</td><td class="num">{r:.3f}</td><td class="num">{f:.3f}</td><td class="num">{s}</td></tr>'
            for n, p, r, f, s in PER_CLASS)
        html(f"""<div class="card"><h3>Performa per kelas</h3><div class="sub">Dihitung dari 378 gambar data test.</div>
            <table class="tbl"><tr><th>Kelas</th><th class="num">Precision</th><th class="num">Recall</th>
            <th class="num">F1</th><th class="num">Data</th></tr>{rows}</table></div>""")
    with right:
        bars = ""
        for n, a in ROBUSTNESS:
            color = "#34d399" if n == "Normal" else "#ff4d8d" if a < 60 else "#a78bfa"
            bars += (f'<div class="prob"><div class="top"><span>{n}</span><span>{a:.1f}%</span></div>'
                     f'<div class="bar"><div style="width:{a}%;background:{color}"></div></div></div>')
        html(f'<div class="card"><h3>Uji ketahanan (robustness)</h3><div class="sub">Akurasi saat gambar test diberi gangguan.</div>{bars}</div>')

    callout("warn", "", "<b>Kelemahan model: blur.</b> Akurasi turun dari sekitar 88% jadi 52% kalau gambarnya buram. "
            "Dugaan kami, model bergantung pada tekstur halus permukaan buah. Makanya aplikasi ini cek ketajaman foto dulu.")
    callout("", "", "<b>Arsitektur:</b> CNN buatan sendiri, dilatih dari nol tanpa pre-trained weights. Terdiri dari 4 blok konvolusi "
            "(Conv 3×3, BatchNorm, ReLU, MaxPool, Dropout), Global Average Pooling, Dense, lalu softmax 5 kelas.")
    callout("", "", "<b>Yang sering tertukar:</b> Tayberry dan Wineberry, karena sama-sama merah dan berbulir. "
            "Sekitar 64% kesalahan model jatuh ke kelas Wineberry.")


def render_guide(class_names):
    cards = ""
    for c in class_names:
        m = info_for(c)
        lis = "".join(f"<li>{escape(x)}</li>" for x in m["ciri"])
        cards += (f'<div class="berry-card"><div class="strip" style="background:{m["color"]}"></div>'
                  f'<h4>{m["emoji"]} {escape(m["label"])}</h4><div class="latin">{escape(m["latin"])}</div>'
                  f'<p>{escape(m["desc"])}</p><ul>{lis}</ul></div>')
    html(f'<div class="berry-grid">{cards}</div>')
    html('<div class="footer" style="margin-top:22px;text-align:left">Dataset: '
         '<a href="https://www.kaggle.com/datasets/srijitabandyopadhyay/berry-image-dataset" target="_blank">Berry Image Dataset (Kaggle)</a> '
         'oleh Bandopadhyay, Banerjee &amp; Debnath.</div>')


# ----------------------------- Halaman utama --------------------------------
def main():
    st.markdown(f"<style>{(BASE_DIR / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        st.error("File model tidak ketemu. Pastikan berry_cnn.keras dan class_names.json ada di folder yang sama dengan app.py.")
        st.stop()
    try:
        model, grad_model, class_names = load_assets()
    except Exception as e:  # noqa: BLE001
        st.error("Model gagal dimuat. Cek apakah versi TensorFlow sama dengan yang dipakai saat training.")
        st.exception(e)
        st.stop()

    render_hero(class_names)
    st.write("")
    tab_cls, tab_about, tab_guide = st.tabs(["  Klasifikasi", "  Tentang Model", "  Kenali Berry"])

    with tab_cls:
        left, right = st.columns([1, 1.12], gap="large")
        img = None

        with left:
            html('<div class="sec-title">1 · Pilih foto</div>')
            t_up, t_cam = st.tabs(["Upload foto", "Ambil foto"])
            with t_up:
                uploaded = st.file_uploader("Upload foto berry", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                            label_visibility="collapsed")
            with t_cam:
                shot = st.camera_input("Ambil foto berry", label_visibility="collapsed")

            file = uploaded or shot
            if file is not None:
                try:
                    img = load_image(file)
                except Exception:
                    st.error("Gambar tidak bisa dibaca, coba file lain (JPG atau PNG).")

            if img is not None:
                quality = quality_report(img)
                html(f'<img class="photo" src="{to_data_uri(img)}"><div class="photo-cap">Foto yang dianalisis</div>')
                chips = "".join(f'<span class="q {k}">{escape(t)}</span>' for k, t in quality.values())
                html(f'<div class="q-row">{chips}</div>')
                show_cam = st.checkbox("Tampilkan Grad-CAM", value=True, disabled=grad_model is None)

        with right:
            html('<div class="sec-title">2 · Hasil</div>')
            if img is None:
                render_empty()
            else:
                with st.spinner("Menganalisis foto..."):
                    x = to_model_input(img)
                    probs = model(x, training=False).numpy()[0]
                render_result(class_names, probs, blurry=quality["blur"][0] == "bad")

        if img is not None and grad_model is not None and show_cam:
            st.write("")
            idx = int(np.argmax(probs))
            try:
                render_gradcam(img, gradcam(grad_model, x, idx), class_names, idx)
            except Exception:
                st.info("Grad-CAM tidak bisa dibuat untuk gambar ini.")

    with tab_about:
        st.write("")
        render_about()
    with tab_guide:
        st.write("")
        render_guide(class_names)

    html(f'<div class="footer">{escape(TEAM_NAME)}<br>Model hanya mengenal 5 jenis berry. '
         'Hasilnya cuma perkiraan, bukan identifikasi pasti.</div>')


main()
