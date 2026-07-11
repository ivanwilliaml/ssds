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

**Cara membaca notebook ini**: pipeline ini melewati dua putaran perbaikan besar.
Putaran pertama (v4) menemukan dan memperbaiki bug data (spike sensor) yang
membuat satu fitur kunci tidak berguna. Putaran kedua (v5/v6), dipicu oleh code
review independen, menemukan bug lain yang membuat angka RMSE v4 **terlalu
optimis** — notebook ini melaporkan kedua putaran apa adanya, termasuk angka
yang harus dikoreksi turun, karena itu bagian penting dari proses ilmiahnya.

Struktur:
1. EDA (15 pemeriksaan)
2. Data cleaning & feature engineering v4 — perbaikan bug spike sensor
3. Validasi season-matched v4 (mengandung bug leakage, lihat §5)
4. `feature_lib.py` — konsolidasi kode + perbaikan 3 bug dari code review
5. Multi-fold validation v2 — versi benar, dengan climate-regime check nyata
6. Eksperimen lanjutan: anomaly target, upstream-lag (empirical & shapefile), LGBM tuning, per-station tau
7. Model final v6 & submission
8. Ringkasan & rekomendasi lanjutan
""")

# ---------------- 1. EDA ----------------
md("## 1. Exploratory Data Analysis\n\n15 pemeriksaan pada train, test, dan data exogenous.")
codefile(r'D:/Lomba/ssds/model/eda_full.py')

# ---------------- 2. v4 rebuild ----------------
md("""## 2. Data Cleaning & Feature Engineering (v4)

Temuan EDA kunci: `seasonal_lag_1y` sebelumnya bernilai RMSE=64 karena train
mengandung **spike sensor 1-timestep** (mis. Napel 34.55→325.83→34.55) yang
tidak dibersihkan, bukan bug join seperti dugaan awal. Sel berikut membersihkan
spike via deteksi MAD, lalu membangun fitur domain (rolling curah hujan/suhu/
tanah, kalender siklis, atribut statis stasiun, `doy_climatology`,
`seasonal_lag_1y`, anchor persistence).
""")
codefile(r'D:/Lomba/ssds/model/rebuild_v4.py')

# ---------------- 3. v4 validation (contains bug, documented) ----------------
md("""## 3. Validasi Season-Matched v4 ⚠️ *(mengandung bug, diperbaiki di §4-5)*

Validasi season-matched pertama: cutoff = train_end − 365 hari, divalidasi 242
hari berikutnya (window Sep→Mei, meniru horizon test asli). **Bug yang baru
diketahui belakangan** (lihat §4): fungsi pembersihan spike di sini dijalankan
pada `RAW` sebelum displit oleh cutoff, sehingga referensi median/MAD ikut
melihat data "masa depan" relatif ke fold — hasil RMSE≈1.20 di bawah ini
**overoptimistic** dan sudah dikoreksi di §5 menjadi ≈1.44.
""")
codefile(r'D:/Lomba/ssds/model/validate_v4.py')

md("""### 3b. Multi-fold check v4 (versi awal, klaim climate-check belum benar-benar dihitung)

Versi pertama dari multi-fold check ini mengklaim ada pengecekan rezim iklim
(nino_34) di docstring-nya, tapi kodenya **tidak pernah benar-benar menghitung
itu** — baru diperbaiki di §5.
""")
codefile(r'D:/Lomba/ssds/model/multi_fold_validate.py')

# ---------------- 4. feature_lib ----------------
md("""## 4. `feature_lib.py` — Konsolidasi & Perbaikan Bug (dari Code Review)

Code review independen terhadap notebook v4 menemukan 3 bug nyata (diverifikasi
satu-satu terhadap kode, bukan asumsi):

1. **tau=20 di final submission tidak pernah divalidasi** — grid tau yang diuji
   di §3 adalah `[30..300]` dan `[90..240]`, angka `20` tidak ada di grid manapun;
   ternyata berasal dari eksperimen ad-hoc terpisah yang tidak pernah masuk ke
   pipeline resmi.
2. **Klaim climate-regime check di §3b tidak benar-benar dihitung** di kode
   (hanya klaim di docstring).
3. **Leakage nyata**: `clean_station()` dijalankan di atas `RAW` sebelum displit
   `CUT`, sehingga statistik pembersihan spike (median/MAD) ikut melihat data
   sesudah cutoff fold — bukan cuma isu pelabelan, ini benar-benar mengubah
   angka RMSE (§5 menunjukkan 1.20 → 1.4585 setelah diperbaiki).

`feature_lib.py` mengonsolidasikan seluruh logika feature engineering yang
sebelumnya diduplikasi 3× (akar penyebab bug #1, karena versi-versi itu
drift satu sama lain), dan memperbaiki ketiga bug di atas.
""")
codefile(r'D:/Lomba/ssds/model/feature_lib.py')

# ---------------- 5. multi_fold_validate_v2 ----------------
md("""## 5. Multi-Fold Validation v2 (Versi Benar)

Menjalankan dua fold season-matched independen (2023-24 dan 2024-25) dengan
konfigurasi model **tetap** (bukan di-tuning ulang per fold), dan benar-benar
menghitung:
- **Keselarasan struktural**: baris/stasiun & horizon vs `test.csv` asli
- **Keselarasan rezim iklim**: `nino_34`/curah hujan fold vs periode test asli
  (dihitung dari `data_lingkungan.csv`, yang sudah mencakup penuh periode test)

**Hasil jujur setelah bug leakage diperbaiki**: fold 2024-25 (rezim iklim
paling mirip test asli, nino_34 gap=0.005) → RMSE **1.4585**, bukan 1.20 yang
diklaim sebelumnya. Fold 2023-24 (El Niño, climate mismatch) → RMSE 2.56.
Tetap jauh di bawah floor lama v3 (1.77–1.84), tapi perbaikannya lebih kecil
dari klaim awal.
""")
codefile(r'D:/Lomba/ssds/model/multi_fold_validate_v2.py')

# ---------------- 6. further experiments ----------------
md("""## 6. Eksperimen Lanjutan

Setelah pipeline diperbaiki, diuji beberapa lever tambahan yang diusulkan
review — dilaporkan apa adanya termasuk yang **gagal**, bukan hanya yang berhasil.

### 6a. Reparametrisasi target ke anomali (`y' = tma_mdpl - doy_climatology`)
**Hasil: NEGATIF.** RMSE 1.5087 vs 1.4629 (level target) pada fold & fitur
yang sama. `station_mean`/`nama_pos` kategorikal sudah cukup efisien menangkap
offset elevasi per stasiun, sehingga reparametrisasi ini justru membuang sinyal
yang berguna, bukan menambah. **Tidak diadopsi.**
""")
codefile(r'D:/Lomba/ssds/model/anomaly_target_test.py')

md("""### 6b. Upstream-downstream lag feature (empirical, cross-correlation)

Daripada langsung parsing shapefile HydroRIVERS, dicoba dulu pendekatan
data-driven: cross-correlation antar semua pasangan stasiun di berbagai lag
untuk menemukan hubungan hulu-hilir secara empiris (lag=0 dikecualikan karena
biasanya cuma korelasi cuaca bersama, bukan travel-time aliran nyata).
6/30 stasiun punya kandidat kuat (corr>0.5, lag>0), termasuk **Bojonegoro -
Kali Kethek ← Cepu (lag 6 jam, corr 0.96)** — salah satu kontributor error
terbesar. **Hasil: POSITIF kecil** (RMSE 1.4585 → 1.4433, ~1%). Diadopsi
sebagai fitur `upstream_lag_value`.
""")
codefile(r'D:/Lomba/ssds/model/upstream_lag_test.py')

md("""### 6c. Upstream-downstream topology dari shapefile HydroRIVERS asli

Untuk memvalidasi/memperluas 6c, dicoba parsing `HydroRIVERS_v10_au_shp`
sungguhan: setiap stasiun di-snap ke segmen sungai terdekat, lalu diurutkan
hulu→hilir memakai `DIST_DN_KM` (jarak ke muara) pada `MAIN_RIV` yang sama.
26/30 stasiun berada di satu mainstem yang sama, dan urutannya **cocok persis**
dengan geografi asli Bengawan Solo (Wonogiri di hulu → Karanggeneng dekat
muara). Ini menghasilkan 23 pasangan hulu-hilir yang **fisik nyata** (bukan
sekadar korelasi), beberapa dengan korelasi sangat tinggi (Cepu←Karangnongko
corr=0.98).

**Hasil: NETRAL/NEGATIF.** Replace penuh (23 pasangan) → RMSE **memburuk**
(1.4433→1.4707, kemungkinan HistGB overfitting pada kolom tambahan di fold
training yang terbatas). Versi hybrid (hanya tambah Jurug & Peren, kontributor
error terbesar yang belum ter-cover) → RMSE nyaris sama (1.4433→1.4442,
dalam batas noise). **Tidak diadopsi** — sinyal yang ada tampaknya sudah
tertangkap oleh fitur lain (rolling exogenous, seasonal_lag_1y, doy_climatology).
Data topologi tetap disimpan (`upstream_shapefile_map.csv`) untuk referensi laporan.
""")
codefile(r'D:/Lomba/ssds/model/upstream_shapefile_test.py')

md("""### 6d. Tuning LGBM serius (native categorical, 4 konfigurasi)

LGBM selalu mendapat bobot 0 di ensemble karena belum pernah dituning
sungguhan. Dicoba 4 konfigurasi (termasuk pohon lebih dalam + subsampling)
dengan native categorical handling. **Hasil: NEGLIGIBLE.** LGBM terbaik
standalone (1.577) masih kalah dari ensemble Ridge+HistGB (1.4482); ditambahkan
ke ensemble hanya menggeser RMSE dari 1.4482→1.4470 (~0.08%). **Tidak diadopsi**
— tidak sepadan dengan kompleksitas tambahan.
""")
codefile(r'D:/Lomba/ssds/model/lgbm_tuning_test.py')

md("""### 6e. Tau per-stasiun + audit stasiun bermasalah

Diagnosa lanjutan setelah leaderboard real (1.63) dibandingkan dengan backtest
(1.44): 4/30 stasiun (Jurug, Peren, Wonogiri Dam, Bojonegoro - Kali Kethek)
menyumbang **~47% dari total error**. Audit menunjukkan tidak ada lagi data
kotor di stasiun-stasiun ini (variansi normal, tanpa spike tersisa) — tapi
ditemukan pola **"flat lalu lompat"** yang genuine (bukan artefak resolusi
sensor, presisi float utuh dipertahankan): Wonogiri Dam punya plateau sampai
21 langkah observasi berturutan (~7 hari rata). Ini konsisten dengan Wonogiri
Dam sebagai **bendungan sungguhan** — levelnya keputusan operasional manusia,
bukan murni hidrologi, dan tidak ada fitur di dataset yang menangkap jadwal
operasi bendungan. Ini plafon struktural yang sulit ditembus tanpa data
operasional tambahan.

Tau per-stasiun (dibanding satu tau global) memberi perbaikan kecil tapi nyata:
RMSE 1.4433 → 1.4335 (~0.7%). **Diadopsi** ke submission final.
""")
codefile(r'D:/Lomba/ssds/model/per_station_tau_test.py')

# ---------------- 7. final model ----------------
md("""## 7. Model Final v5/v6 & Submission

`rebuild_v5.py` membangun fitur final (train+test) memakai `feature_lib.py`
yang sudah diperbaiki. `final_submission_v6.py` melatih model pada **seluruh**
data train, memakai ensemble Ridge(0.8)+HistGB(0.2) yang diblend dengan
persistence terakhir memakai **tau per-stasiun** (§6e).
""")
codefile(r'D:/Lomba/ssds/model/rebuild_v5.py')
codefile(r'D:/Lomba/ssds/model/final_submission_v6.py')

# ---------------- 8. summary ----------------
md("""## 8. Ringkasan & Rekomendasi Lanjutan

| Tahap | RMSE backtest | Catatan |
|---|---|---|
| Persistence murni | ~2.11–2.26 | baseline |
| v3 (leaderboard real) | ~1.77–1.84 | floor sebelum rebuild |
| v4 (klaim awal) | ~1.20 | ❌ inflated oleh bug leakage (§3→§5) |
| v4 setelah leakage diperbaiki | ~1.4585 | angka jujur pembanding |
| **v6 (final, upstream-lag + per-station tau)** | **~1.4335** | leaderboard real: **1.63** |

**Gap backtest vs leaderboard (1.44 vs 1.63, ~13%)** jauh lebih sehat dari gap
awal (0.19 vs 1.89, 10×) — konsisten dengan optimisme wajar validasi satu-fold,
bukan tanda metodologi salah.

**Lever yang terbukti membantu**: pembersihan spike sensor (akar perbaikan
terbesar), upstream-lag empirical (+1%), tau per-stasiun (+0.7%).

**Lever yang dicoba tapi gagal** (dilaporkan untuk transparansi, bukan
disembunyikan): reparametrisasi anomali, upstream topology shapefile penuh,
tuning LGBM.

**Plafon struktural yang teridentifikasi**: Wonogiri Dam & stasiun terkontrol
lain butuh data operasional (jadwal rilis air) yang tidak tersedia di
kompetisi ini — potensi perbaikan lanjutan paling besar adalah model/fitur
khusus untuk stasiun-stasiun ini, bukan lagi tuning global.

Submission akhir: `submission_v6.csv`.
""")

nb['cells'] = cells
with open(r'D:/Lomba/ssds/model/SSDS2026_rebuild_v6.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("notebook written: SSDS2026_rebuild_v6.ipynb")
