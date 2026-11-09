"""Data statis: info tiap kelas dan hasil evaluasi dari notebook."""

# Kunci = nama kelas persis seperti di class_names.json
BERRY_INFO = {
    "Tayberry": {
        "label": "Tayberry", "emoji": "", "color": "#ff5d8f",
        "latin": "Rubus fruticosus × R. idaeus",
        "desc": "Hasil silangan blackberry dan raspberry. Buahnya agak lonjong dan merah tua keunguan kalau sudah matang.",
        "ciri": ["Lonjong, merah tua keunguan", "Permukaan berbulir seperti raspberry", "Rasanya manis asam"],
    },
    "Wineberry": {
        "label": "Wineberry", "emoji": "", "color": "#ff3b4e",
        "latin": "Rubus phoenicolasius",
        "desc": "Berry liar dari Asia Timur. Buahnya kecil, merah cerah, dan berbulir halus.",
        "ciri": ["Merah oranye sampai merah cerah", "Berbulir dan sering mengkilap", "Kelopak dan batangnya berbulu kemerahan"],
    },
    "black berry": {
        "label": "Blackberry", "emoji": "", "color": "#060019",
        "latin": "Rubus fruticosus (agg.)",
        "desc": "Buah majemuk berwarna hitam mengkilap yang tumbuh di semak berduri.",
        "ciri": ["Hitam pekat dan mengkilap", "Tersusun dari banyak butir kecil", "Bentuknya bulat memanjang"],
    },
    "blueberry": {
        "label": "Blueberry", "emoji": "", "color": "#035679",
        "latin": "Vaccinium spp.",
        "desc": "Buah bulat kecil berwarna biru keunguan, permukaannya dilapisi lilin keputihan.",
        "ciri": ["Bulat, biru keunguan", "Ada lapisan lilin keputihan", "Ada mahkota kecil di ujung buah"],
    },
    "white mulberry": {
        "label": "White Mulberry", "emoji": "", "color": "#fcfcfa",
        "latin": "Morus alba",
        "desc": "Buah pohon murbei putih. Warnanya putih kekuningan, bisa jadi merah muda atau ungu kalau sudah matang.",
        "ciri": ["Putih, krem, atau kehijauan", "Bisa memerah muda saat matang", "Kecil, lonjong, dan bergerombol"],
    },
}

# Hasil evaluasi di notebook (data test, 378 gambar)
TEST_ACC, TEST_F1 = 88.36, 87.83

# (kelas, precision, recall, f1, jumlah data)
PER_CLASS = [
    ("Tayberry", 0.892, 0.773, 0.829, 75),
    ("Wineberry", 0.705, 0.893, 0.788, 75),
    ("black berry", 0.955, 0.977, 0.966, 87),
    ("blueberry", 0.966, 0.925, 0.945, 93),
    ("white mulberry", 0.950, 0.792, 0.864, 48),
]

# Akurasi test saat gambar diberi gangguan
ROBUSTNESS = [
    ("Normal", 87.83), ("Noise gaussian", 83.60), ("JPEG q=15", 84.13), ("Rotasi 30°", 83.33),
    ("Gelap", 80.95), ("Terang", 79.37), ("Blur", 52.12),
]
