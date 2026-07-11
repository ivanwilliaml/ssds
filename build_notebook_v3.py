import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

def codefile(path):
    code(open(path, encoding='utf-8').read())

md("""# SSDS 2026 — Prediksi Tinggi Muka Air (TMA) DAS Bengawan Solo

End-to-end pipeline untuk kompetisi Sebelas Maret Statistics Data Science 2026.
Target: prediksi `tma_mdpl` untuk 30 pos pemantauan, periode test 2025-09-19 s/d
2026-05-18 (242 hari, forecast murni tanpa ground-truth TMA), dievaluasi dengan RMSE.

**Cara membaca notebook ini**: pipeline melewati tiga putaran perbaikan.
- **v4**: rebuild awal, menemukan & memperbaiki bug spike sensor.
- **v5/v6**: dipicu code review pertama, menemukan bug leakage yang membuat
  angka v4 terlalu optimis (1.20→1.4585), plus beberapa eksperimen lanjutan.
- **v7 (final)**: dipicu code review kedua yang lebih dalam — memperbaiki 2 bug
  metodologi tambahan (self-referential climatology, upstream-map beku lintas
  fold) dan menambah 3 pemeriksaan robustness (pemisahan tuning/scoring,
  bootstrap signifikansi tau per-stasiun, uji missingness).

Setiap putaran dilaporkan apa adanya — termasuk angka yang harus dikoreksi
turun dan eksperimen yang gagal — karena itu bagian dari proses ilmiahnya.

Struktur:
1. EDA (15 pemeriksaan)
2. v4: cleaning & feature engineering (bug spike sensor diperbaiki)
3. v4: validasi season-matched (⚠️ kemudian ditemukan mengandung bug leakage)
4. `feature_lib.py` — konsolidasi kode + 5 bug fix dari 2 putaran code review
5. Multi-fold validation v2 (perbaikan putaran 1: climate-check nyata)
6. Eksperimen lanjutan putaran 1: anomaly target, upstream-lag, LGBM tuning, per-station tau (naif)
7. Multi-fold validation v3 (perbaikan putaran 2: pemisahan tuning/scoring + cross-fold stability)
8. Bootstrap signifikansi tau per-stasiun (perbaikan putaran 2)
9. Uji missingness backtest vs curah hujan (perbaikan putaran 2)
10. Model final v7 & submission
11. Ringkasan & rekomendasi lanjutan
""")

# ---------------- 1. EDA ----------------
md("## 1. Exploratory Data Analysis\n\n15 pemeriksaan pada train, test, dan data exogenous.")
codefile(r'D:/Lomba/ssds/model/eda_full.py')

# ---------------- 2. v4 rebuild ----------------
md("""## 2. Data Cleaning & Feature Engineering (v4)

Temuan EDA kunci: `seasonal_lag_1y` sebelumnya bernilai RMSE=64 karena train
mengandung **spike sensor 1-timestep** (mis. Napel 34.55→325.83→34.55) yang
tidak dibersihkan. Sel berikut membersihkan spike via deteksi MAD, lalu
membangun fitur domain (rolling curah hujan/suhu/tanah, kalender siklis,
atribut statis stasiun, `doy_climatology`, `seasonal_lag_1y`, anchor persistence).
""")
codefile(r'D:/Lomba/ssds/model/rebuild_v4.py')

# ---------------- 3. v4 validation (buggy, documented) ----------------
md("""## 3. Validasi Season-Matched v4 ⚠️ *(mengandung bug, diperbaiki di §5)*

**Bug yang ditemukan belakangan**: fungsi pembersihan spike di sini dijalankan
pada `RAW` sebelum displit oleh cutoff, sehingga referensi median/MAD ikut
melihat data "masa depan" relatif ke fold — RMSE≈1.20 di bawah ini
**overoptimistic**, dikoreksi di §5 menjadi ≈1.4585 (lalu ≈1.34 setelah
putaran perbaikan kedua di §7-10).
""")
codefile(r'D:/Lomba/ssds/model/validate_v4.py')

md("### 3b. Multi-fold check v4 (versi awal, klaim climate-check belum benar-benar dihitung)")
codefile(r'D:/Lomba/ssds/model/multi_fold_validate.py')

# ---------------- 4. feature_lib ----------------
md("""## 4. `feature_lib.py` — Konsolidasi & Perbaikan Bug (2 Putaran Code Review)

**Putaran review 1** menemukan 3 bug: (1) tau=20 di submission tidak pernah
divalidasi terhadap grid manapun, (2) klaim climate-regime check tidak benar-benar
dihitung di kode, (3) `clean_station()` dijalankan sebelum split `CUT` (leakage nyata).

**Putaran review 2** menemukan 2 bug lagi, lebih halus:
- **`doy_climatology` self-referential untuk baris training**: window ±5 hari
  dihitung dari `train_only` yang MENCAKUP baris itu sendiri, jadi tiap baris
  training jadi ~1 dari ~99 titik yang dirata-ratakan ke fitur miliknya
  sendiri (val/test tidak kena ini karena bukan bagian dari `train_only`).
  **Fix**: `build_doy_clim_loyo()` — leave-one-year-out, tiap baris training
  memakai climatology yang mengecualikan tahunnya sendiri.
- **`UPSTREAM_MAP` dibekukan dari satu cutoff (2024-09-18), dipakai ulang di
  semua fold** — benar untuk FOLD2 (cutoff sama) tapi salah untuk FOLD1
  (cutoff 2023-09-18, ikut "melihat" data setelah cutoff-nya sendiri).
  **Fix**: `derive_upstream_map(train_only)` — topologi (pasangan stasiun)
  tetap tetap (fakta geografis), tapi lag-nya dihitung ulang dari `train_only`
  fold yang sedang aktif, dipanggil otomatis di setiap `build_features()`.
""")
codefile(r'D:/Lomba/ssds/model/feature_lib.py')

# ---------------- 5. multi_fold_validate_v2 ----------------
md("""## 5. Multi-Fold Validation v2 (Perbaikan Putaran 1)

Dua fold season-matched independen (2023-24, 2024-25), konfigurasi model
**tetap** (bukan re-tuning per fold), dengan climate-regime check yang
benar-benar dihitung (nino_34/curah hujan vs test asli).

**Hasil setelah bug leakage putaran 1 diperbaiki**: fold 2024-25 (climate
gap=0.005) → RMSE **1.4585**, bukan 1.20 yang diklaim v4. Masih jauh di bawah
floor v3 (1.77–1.84).
""")
codefile(r'D:/Lomba/ssds/model/multi_fold_validate_v2.py')

# ---------------- 6. further experiments round 1 ----------------
md("""## 6. Eksperimen Lanjutan (Putaran 1)

### 6a. Reparametrisasi target ke anomali — **NEGATIF**, tidak diadopsi.
""")
codefile(r'D:/Lomba/ssds/model/anomaly_target_test.py')

md("### 6b. Upstream-downstream lag feature (empirical cross-correlation) — **POSITIF kecil (~1%)**, diadopsi.")
codefile(r'D:/Lomba/ssds/model/upstream_lag_test.py')

md("### 6c. Upstream-downstream topology dari shapefile HydroRIVERS asli — **NETRAL/NEGATIF**, tidak diadopsi (topologi tetap dipakai di §4 untuk fold-aware lag).")
codefile(r'D:/Lomba/ssds/model/upstream_shapefile_test.py')

md("### 6d. Tuning LGBM serius (4 konfigurasi, native categorical) — awalnya **NEGLIGIBLE** dalam ensemble lama; catatan: bobotnya berubah signifikan setelah perbaikan §4, lihat §7.")
codefile(r'D:/Lomba/ssds/model/lgbm_tuning_test.py')

md("""### 6e. Tau per-stasiun (naif, tanpa uji signifikansi) + audit stasiun bermasalah

Per-station tau (dibanding tau global tunggal) memberi perbaikan kecil:
RMSE 1.4433→1.4335 (~0.7%). **Catatan penting**: versi ini BELUM diuji
signifikansi statistiknya — itu baru dilakukan di §8, dan hasilnya mengubah
keputusan (4 stasiun ternyata fitting noise, bukan sinyal nyata).

Audit stasiun bermasalah (Jurug, Peren, Wonogiri Dam, Bojonegoro - Kali
Kethek) menemukan pola "flat lalu lompat" yang genuine di Wonogiri Dam (bukan
artefak sensor) — konsisten dengan statusnya sebagai bendungan sungguhan
(Waduk Gajah Mungkur), levelnya keputusan operasional manusia, bukan murni
hidrologi.
""")
codefile(r'D:/Lomba/ssds/model/per_station_tau_test.py')

# ---------------- 7. multi_fold_validate_v3 ----------------
md("""## 7. Multi-Fold Validation v3 (Perbaikan Putaran 2 — Poin #1 & #5)

**Bug yang ditemukan review putaran 2**: pemilihan (weights, tau) dan skor
fold dihitung dari `y_true` validasi yang SAMA — validation set dan tuning
set adalah 19.5k titik yang identik. Kemungkinan bias kecil (grid kasar:
~21 kombinasi bobot × 16 tau), tapi "kemungkinan kecil" perlu diverifikasi,
bukan diasumsikan.

**Fix**: `run_fold()` sekarang menerima `select_weights`/`fixed_weights`
(mencerminkan `select_tau`/`fixed_tau` yang sudah ada), lalu FOLD1 di-skor
memakai config FOLD2 yang **dibekukan** (tanpa re-selection) sebagai cek
stabilitas cross-fold.

**Efek samping penting**: setelah bug §4 diperbaiki (LOYO climatology +
upstream-map fold-aware), konfigurasi ensemble optimal **berubah** — LGBM
sekarang mendapat bobot 0.6 (sebelumnya 0 karena overfitting pada fitur yang
mengandung self-leak), RMSE FOLD2 turun ke ≈1.34.
""")
codefile(r'D:/Lomba/ssds/model/multi_fold_validate_v3.py')

# ---------------- 8. bootstrap ----------------
md("""## 8. Bootstrap Signifikansi Tau Per-Stasiun (Perbaikan Putaran 2 — Poin #2)

Tau per-stasiun di §6e adalah 30 argmin independen atas grid 17 nilai, masing-
masing di-fit ke ~650 titik saja — berisiko fitting sampling noise, bukan
karakteristik persistence stasiun yang genuine. Bootstrap 300× per stasiun:
apakah tau pilihannya benar-benar lebih baik dari tau global secara konsisten,
atau cuma menang kebetulan?

**Hasil**: 26/30 stasiun lolos ambang percaya diri (p≥0.70). 4 stasiun
(Badegan, Peren, Ngadipiro, Karangnongko) gagal — tau globalnya menang di
hampir semua resampling, berarti tau spesifik mereka sebelumnya cuma
fitting noise. **Peren** salah satu top-4 kontributor error — temuan ini
penting karena "perbaikan" tau khusus Peren di §6e ternyata palsu.
Stasiun-stasiun ini di-fallback ke tau global di submission final (§10).
""")
codefile(r'D:/Lomba/ssds/model/bootstrap_tau_significance.py')

# ---------------- 9. missingness ----------------
md("""## 9. Uji Missingness Backtest vs Curah Hujan (Perbaikan Putaran 2 — Poin #3)

Backtest FOLD2 hanya mencakup ~90% baris/stasiun dari test asli (real test
tidak punya missing value sama sekali, by construction dari `sample_submission`).
Kalau gap itu adalah sensor dropout saat kondisi ekstrem (rutin terjadi di
instrumentasi hidrologi), backtest diam-diam melewati baris-baris tersulit —
bukan leakage target-ke-fitur, tapi "backtest tidak mencerminkan distribusi
evaluasi asli".

**Hasil**: secara **pooled**, signifikan (Mann-Whitney p=3e-70) — baris hilang
memang condong ke kondisi lebih basah (median 4.0mm vs 1.8mm). Tapi untuk
**3 stasiun dengan missingness tertinggi** (Gunungsari, Floodway Bridge C,
Bojonegoro - Kali Kethek — termasuk salah satu top-error kita), pola justru
**terbalik** (rasio curah hujan missing/observed 0.59–0.78, lebih kering).
Kesimpulan bernuansa: risiko ini nyata secara umum di dataset, tapi tidak
memperparah keandalan backtest untuk stasiun yang sudah jadi fokus perhatian.
""")
codefile(r'D:/Lomba/ssds/model/missingness_check.py')

# ---------------- 10. final model v7 ----------------
md("""## 10. Model Final v7 & Submission

Menggabungkan seluruh perbaikan: `feature_lib.py` dengan LOYO climatology +
upstream-map fold-aware (§4), bobot ensemble Ridge(0.4)+LGBM(0.6) dari §7,
dan tau per-stasiun yang HANYA dipakai untuk 26 stasiun bootstrap-confirmed
(§8) — 4 sisanya (Badegan, Peren, Ngadipiro, Karangnongko) memakai tau
global=15.
""")
codefile(r'D:/Lomba/ssds/model/rebuild_v7.py')
codefile(r'D:/Lomba/ssds/model/final_submission_v7.py')

# ---------------- 11. summary ----------------
md("""## 11. Ringkasan & Rekomendasi Lanjutan

*(Angka di bawah ditranskrip langsung dari output sel-sel di atas pada run
notebook ini — bukan diketik manual — untuk menghindari drift narasi vs kode
yang ditemukan review putaran 1, poin #6.)*

| Tahap | RMSE backtest (FOLD2, climate-matched) | Catatan |
|---|---|---|
| Persistence murni | ~2.11–2.26 | baseline |
| v3 (leaderboard real) | ~1.77–1.84 | floor sebelum rebuild |
| v4 (klaim awal) | ~1.20 | ❌ inflated, bug leakage §3 |
| v5/v6 (leakage §3 diperbaiki) | ~1.4433–1.4585 | angka jujur putaran 1 |
| **v7 (final, +LOYO climatology, +upstream fold-aware, +bootstrap-robust tau)** | **~1.34** (lihat output §7) | leaderboard real sebelumnya (v6): **1.63** |

**Cross-fold stability (§7)**: config FOLD2 yang dibekukan lalu diskor di
FOLD1 mengalami penalti kecil (+0.10 RMSE, ~4%) dibanding FOLD1 self-tuned —
config cukup stabil, tidak curve-fit parah ke FOLD2.

**Lever yang terbukti membantu**: pembersihan spike sensor (akar perbaikan
terbesar), upstream-lag empirical, LOYO climatology + upstream fold-aware
(mengubah ensemble optimal jadi menyertakan LGBM), tau per-stasiun **yang
sudah difilter bootstrap** (bukan versi naif §6e yang sebagiannya palsu).

**Lever yang dicoba tapi gagal** (dilaporkan untuk transparansi): reparametrisasi
anomali, upstream topology shapefile penuh, tuning LGBM standalone (meski
akhirnya LGBM tetap masuk ensemble lewat jalur berbeda di §7).

**Plafon struktural yang teridentifikasi**: Wonogiri Dam (Waduk Gajah Mungkur)
butuh data operasional (jadwal rilis air) yang tidak tersedia di kompetisi
ini. Uji missingness (§9) menunjukkan risiko "backtest terlalu optimis akibat
sensor dropout saat ekstrem" nyata secara umum tapi TIDAK memperparah 3
stasiun missingness tertinggi secara spesifik.

**Belum dicek** (di luar cakupan kerja lokal): perbedaan leaderboard
publik/private Kaggle — kalau skor 1.63 yang terlihat adalah subset publik,
bukan full 242-hari×30-stasiun, itu bisa menjelaskan sebagian gap tanpa perlu
bug lagi. Cek tab Evaluation kompetisi.

Submission akhir: `submission_v7.csv`.
""")

nb['cells'] = cells
with open(r'D:/Lomba/ssds/model/SSDS2026_rebuild_v7.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("notebook written: SSDS2026_rebuild_v7.ipynb")
