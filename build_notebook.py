import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# SSDS 2026 — Prediksi Tinggi Muka Air (TMA) DAS Bengawan Solo

**Rebuild end-to-end pipeline** untuk kompetisi Sebelas Maret Statistics Data Science 2026.

Target: prediksi `tma_mdpl` untuk 30 pos pemantauan, periode test 2025-09-19 s/d 2026-05-18
(242 hari, forecast murni tanpa ground-truth TMA), dievaluasi dengan RMSE.

Struktur notebook:
1. EDA (15 pemeriksaan) — struktur data, missing value, outlier, distribusi
2. Data cleaning — deteksi & perbaikan spike sensor
3. Feature engineering — domain hidrologi (climatology, seasonal lag, exogenous rolling)
4. Validasi season-matched anti-leakage (mimic distribusi test)
5. Multi-fold robustness check (keselarasan struktural + rezim iklim vs test asli)
6. Model final + submission
""")

# ---------------- 1. EDA ----------------
md("## 1. Exploratory Data Analysis\n\n15 pemeriksaan pada train, test, dan data exogenous.")
code(open(r'D:/Lomba/ssds/model/eda_full.py', encoding='utf-8').read())

# ---------------- 2+3. Feature engineering (rebuild_v4) ----------------
md("""## 2-3. Data Cleaning & Feature Engineering

Temuan EDA kunci: `seasonal_lag_1y` sebelumnya bernilai RMSE=64 karena train
mengandung **spike sensor 1-timestep** (mis. Napel 34.55→325.83→34.55) yang
tidak dibersihkan, bukan karena bug join. Sel berikut:
- Membersihkan spike via deteksi MAD (titik terisolasi jauh dari kedua tetangga)
- Membangun fitur domain: rolling curah hujan/suhu/tanah, kalender siklis,
  atribut statis stasiun, `doy_climatology` (smoothed ±5 hari), `seasonal_lag_1y`
  (merge_asof toleran 2 hari), dan anchor persistence untuk blending far-horizon.
""")
code(open(r'D:/Lomba/ssds/model/rebuild_v4.py', encoding='utf-8').read())

# ---------------- 4. Validation ----------------
md("""## 4. Validasi Season-Matched (Anti-Leakage)

Validasi dirancang meniru distribusi test: cutoff = train_end − 365 hari,
divalidasi pada 242 hari berikutnya (window Sep→Mei, sama persis dengan
horizon test asli). Semua statistik (mean/std stasiun, climatology,
seasonal lag) dihitung ulang **hanya dari data ≤ cutoff** untuk mencegah leakage.
""")
code(open(r'D:/Lomba/ssds/model/validate_v4.py', encoding='utf-8').read())

# ---------------- 5. Multi-fold robustness ----------------
md("""## 5. Multi-Fold Robustness & Keselarasan dengan Test Asli

Satu fold validasi berisiko overfit ke satu window. Sel berikut menjalankan
dua fold season-matched independen (2023-24 dan 2024-25) dengan konfigurasi
model tetap (tidak di-tuning ulang per fold), dan membandingkan:
- **Keselarasan struktural**: jumlah baris/stasiun & horizon vs `test.csv` asli
- **Keselarasan rezim iklim**: `nino_34` fold vs periode test asli (data
  exogenous tersedia penuh hingga Mei 2026, sehingga bisa dicek langsung)

Hasil: fold 2024-25 (rezim iklim paling mirip test asli) → RMSE≈1.20.
Fold 2023-24 (El Niño, mismatch iklim, training lebih sedikit) tetap
RMSE≈1.41 — jauh di bawah floor lama (1.77–1.84), mengonfirmasi perbaikan
ini robust, bukan artefak satu fold.
""")
code(open(r'D:/Lomba/ssds/model/multi_fold_validate.py', encoding='utf-8').read())

# ---------------- 6. Final submission ----------------
md("""## 6. Model Final & Submission

Model final dilatih pada **seluruh** data train (84.396 baris, cleaned),
menggunakan ensemble Ridge+HistGB (bobot dari tuning season-matched) yang
diblend dengan persistence terakhir (tau=20) untuk horizon sangat dekat.
""")
code(open(r'D:/Lomba/ssds/model/final_submission_v4.py', encoding='utf-8').read())

md("""## Ringkasan

| Tahap | RMSE (backtest) |
|---|---|
| Persistence murni | ~2.11–2.26 |
| Pipeline lama (v3, sebelum cleaning) | ~1.77–1.84 (floor) |
| **Rebuild v4 (data cleaning + FE + validasi season-matched)** | **~1.20 (fold representatif iklim)** |
| Stress-test fold iklim-mismatch (El Niño, training minim) | ~1.41 |

Akar perbaikan terbesar: pembersihan spike sensor 1-timestep pada `tma_mdpl`
sebelum dipakai sebagai fitur lag, yang sebelumnya membuat `seasonal_lag_1y`
(anchor far-horizon terpenting) menjadi tidak berguna (RMSE=64 → RMSE≈1.5,
kompetitif dengan anchor lain).

Submission akhir: `submission_v4.csv`.
""")

nb['cells'] = cells
with open(r'D:/Lomba/ssds/model/SSDS2026_rebuild_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("notebook written")
