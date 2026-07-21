"""
Petha — Retrain Hybrid Recommendation Model
=============================================
Menggantikan alur manual "export CSV dari MySQL → upload ke Colab →
jalankan notebook → download models.pkl → upload ke server".

Script ini konek LANGSUNG ke MySQL, menjalankan preprocessing yang PERSIS
SAMA dengan notebook Colab (AI_Petha_CLEAN.ipynb), lalu menyimpan ulang:
  - models.pkl
  - train_data.csv
  - data_restoran.csv

Cara pakai:
  1. Install dependency (sekali saja):
       pip install pandas scikit-learn sqlalchemy pymysql --break-system-packages

  2. Sesuaikan konfigurasi DB_* di bawah kalau perlu (default cocok untuk
     XAMPP/phpMyAdmin lokal: host=127.0.0.1, user=root, password kosong).

  3. Jalankan dari folder yang sama dengan main.py (FastAPI):
       python retrain.py

  4. Restart FastAPI (uvicorn) supaya model baru ke-load ke memory.

Jalankan script ini setiap kali ada user baru mengisi onboarding/preferensi,
atau ada ulasan/rating baru masuk, supaya AI langsung "mengenal" data terbaru.
"""

import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# ── KONFIGURASI DATABASE ──────────────────────────────────────
# Sesuaikan kalau setup MySQL Anda berbeda dari default XAMPP
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "petahalal2"

# ── OUTPUT FILES (harus sama persis dengan yang dibaca main.py) ──
OUT_MODELS_PKL = "models.pkl"
OUT_TRAIN_DATA = "train_data.csv"
OUT_DATA_RESTORAN = "data_restoran.csv"


def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


def load_data(engine):
    """Ambil data mentah langsung dari MySQL — pengganti CSV export manual."""
    print("📥 Mengambil data dari MySQL...")

    df_ulasan = pd.read_sql(
        "SELECT user_id, id_restoran, rating FROM ulasan", engine
    )

    df_restoran = pd.read_sql(
        """
        SELECT id_restoran, nama_restoran, deskripsi, kota, status_halal,
               harga_rata_rata_min, harga_rata_rata_max, rating
        FROM restoran
        """,
        engine,
    )

    df_users = pd.read_sql("SELECT id, name, email FROM users", engine)

    df_pencarian = pd.read_sql(
        "SELECT id_pencari, keyword, lokasi FROM pencarian", engine
    )

    # Kategori 'skip' sengaja DIKECUALIKAN — itu bukan preferensi asli,
    # hanya penanda "user sudah ditanya tapi memilih skip" (lihat
    # OnboardingController::store). Kalau ikut masuk, kata "skip" akan
    # mengotori TF-IDF preferensi user.
    df_preferensi = pd.read_sql(
        "SELECT user_id, kategori FROM preferensi_users WHERE kategori != 'skip'",
        engine,
    )

    print(f"   Ulasan    : {len(df_ulasan)} baris")
    print(f"   Restoran  : {len(df_restoran)} baris")
    print(f"   Users     : {len(df_users)} baris")
    print(f"   Pencarian : {len(df_pencarian)} baris")
    print(f"   Preferensi: {len(df_preferensi)} baris (kategori 'skip' dikecualikan)")

    return df_ulasan, df_restoran, df_users, df_pencarian, df_preferensi


def clean_data(df_ulasan, df_restoran, df_pencarian, df_preferensi):
    """Persis mengikuti Cell 16 (Data Cleaning) & Cell 12.5 di notebook."""
    print("\n🧹 Membersihkan data...")

    # -- Ulasan --
    df_ulasan = df_ulasan.drop_duplicates(subset=['user_id', 'id_restoran'])
    df_ulasan = df_ulasan.dropna(subset=['user_id', 'id_restoran', 'rating'])
    df_ulasan = df_ulasan[df_ulasan['rating'].between(1, 5)]
    df_ulasan['user_id'] = df_ulasan['user_id'].astype(int)
    df_ulasan['id_restoran'] = df_ulasan['id_restoran'].astype(int)
    df_ulasan['rating'] = df_ulasan['rating'].astype(float)

    # -- Restoran --
    df_restoran['rating'] = df_restoran['rating'].fillna(0.0)
    jumlah_ulasan_per_resto = df_ulasan.groupby('id_restoran').size()
    df_restoran['jumlah_ulasan'] = (
        df_restoran['id_restoran'].map(jumlah_ulasan_per_resto).fillna(0).astype(int)
    )

    # -- Pencarian --
    df_pencarian = df_pencarian.dropna(subset=['id_pencari', 'keyword'])
    df_pencarian['keyword'] = df_pencarian['keyword'].astype(str).str.strip()
    df_pencarian = df_pencarian[df_pencarian['keyword'] != '']

    # -- Preferensi --
    df_preferensi = df_preferensi.dropna(subset=['user_id', 'kategori'])
    df_preferensi['kategori'] = df_preferensi['kategori'].astype(str).str.strip()
    df_preferensi = df_preferensi[df_preferensi['kategori'] != '']

    print(f"   Ulasan bersih    : {len(df_ulasan)} baris")
    print(f"   Pencarian bersih : {len(df_pencarian)} baris")
    print(f"   Preferensi bersih: {len(df_preferensi)} baris")

    return df_ulasan, df_restoran, df_pencarian, df_preferensi


def build_fitur_restoran(df_restoran):
    """Persis Cell 20 di notebook — fitur konten restoran untuk CBF (Model 2)."""
    print("\n🏗️  Membangun fitur konten restoran (CBF)...")

    df_fitur = df_restoran[
        ['id_restoran', 'kota', 'status_halal',
         'harga_rata_rata_min', 'harga_rata_rata_max', 'rating']
    ].copy()

    df_fitur['status_halal'] = df_fitur['status_halal'].fillna('none')
    df_fitur['kota'] = df_fitur['kota'].fillna('Tidak Diketahui')

    df_fitur = pd.get_dummies(
        df_fitur, columns=['kota', 'status_halal'], prefix=['kota', 'halal']
    )

    scaler = MinMaxScaler()
    num_cols = ['harga_rata_rata_min', 'harga_rata_rata_max', 'rating']
    df_fitur[num_cols] = scaler.fit_transform(df_fitur[num_cols].fillna(0))

    fitur_cols = [c for c in df_fitur.columns if c != 'id_restoran']
    df_fitur_final = df_fitur[['id_restoran'] + fitur_cols]

    print(f"   Jumlah fitur: {len(fitur_cols)}")
    return df_fitur_final


def build_text_profiles(df_restoran, df_preferensi, df_pencarian):
    """Persis Cell 20.5 — TF-IDF konten resto + profil teks user (CBF sinyal 2 & 3)."""
    print("\n📝 Membangun profil teks (preferensi & pencarian)...")

    df_restoran = df_restoran.copy()
    df_restoran['teks_konten'] = (
        df_restoran['nama_restoran'].fillna('') + ' ' +
        df_restoran['deskripsi'].fillna('') + ' ' +
        df_restoran['kota'].fillna('') + ' ' +
        df_restoran['status_halal'].fillna('')
    )

    tfidf = TfidfVectorizer(max_features=200)
    resto_tfidf_matrix = tfidf.fit_transform(df_restoran['teks_konten'])
    resto_id_to_row = {rid: i for i, rid in enumerate(df_restoran['id_restoran'])}

    # Preferensi eksplisit — bobot 3x (sinyal kuat, sengaja diisi user)
    user_pref_text = defaultdict(str)
    for _, row in df_preferensi.iterrows():
        user_pref_text[row['user_id']] += (' ' + str(row['kategori'])) * 3

    # Riwayat pencarian
    user_search_text = defaultdict(str)
    for _, row in df_pencarian.iterrows():
        lokasi = row['lokasi'] if pd.notna(row['lokasi']) else ''
        user_search_text[row['id_pencari']] += ' ' + str(row['keyword']) + ' ' + str(lokasi)

    print(f"   Profil preferensi: {len(user_pref_text)} user")
    print(f"   Profil pencarian : {len(user_search_text)} user")

    return tfidf, resto_tfidf_matrix, resto_id_to_row, user_pref_text, user_search_text


def build_models(df_ulasan, df_fitur_final):
    """Persis Cell 22, 27, 28, 29 — interaction matrix, CF, CBF, Trend."""
    print("\n🤖 Melatih ulang model (CF, CBF, Trend)...")

    # ── Model 1: Collaborative Filtering ──
    # Catatan: di production kita pakai SEMUA data ulasan bersih (bukan
    # hasil train-test split 80% seperti di notebook), karena tidak ada
    # kebutuhan evaluasi di sini — hanya butuh model paling lengkap/akurat.
    train_data = df_ulasan.copy()

    interaction_matrix = train_data.pivot_table(
        index='user_id', columns='id_restoran', values='rating', fill_value=0
    )
    user_similarity = cosine_similarity(interaction_matrix)
    user_sim_df = pd.DataFrame(
        user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index
    )

    # ── Model 2: Content-Based Filtering ──
    fitur_cols = [c for c in df_fitur_final.columns if c != 'id_restoran']
    fitur_matrix = df_fitur_final[fitur_cols].values
    resto_similarity = cosine_similarity(fitur_matrix)
    resto_sim_df = pd.DataFrame(
        resto_similarity,
        index=df_fitur_final['id_restoran'],
        columns=df_fitur_final['id_restoran'],
    )

    # ── Model 3: Popularity-Based Trend ──
    ulasan_count = (
        df_ulasan.groupby('id_restoran').size().reset_index(name='jumlah_ulasan')
    )
    rating_avg = (
        df_ulasan.groupby('id_restoran')['rating'].mean().reset_index(name='avg_rating')
    )
    df_trend = df_fitur_final[['id_restoran']].copy()
    df_trend = df_trend.merge(ulasan_count, on='id_restoran', how='left')
    df_trend = df_trend.merge(rating_avg, on='id_restoran', how='left')
    df_trend = df_trend.fillna(0)

    scaler_trend = MinMaxScaler()
    kolom_norm = ['jumlah_ulasan', 'avg_rating']
    df_trend[kolom_norm] = scaler_trend.fit_transform(df_trend[kolom_norm])
    df_trend['skor_trend'] = df_trend[kolom_norm].mean(axis=1)
    trend_scores = dict(zip(df_trend['id_restoran'], df_trend['skor_trend']))

    print(f"   Interaction matrix: {interaction_matrix.shape[0]} user × {interaction_matrix.shape[1]} restoran")
    print(f"   User similarity   : {user_sim_df.shape}")
    print(f"   Resto similarity  : {resto_sim_df.shape}")
    print(f"   Trend scores      : {len(trend_scores)} restoran")

    return train_data, interaction_matrix, user_sim_df, resto_sim_df, trend_scores


def main():
    engine = get_engine()

    df_ulasan, df_restoran, df_users, df_pencarian, df_preferensi = load_data(engine)
    df_ulasan, df_restoran, df_pencarian, df_preferensi = clean_data(
        df_ulasan, df_restoran, df_pencarian, df_preferensi
    )

    df_fitur_final = build_fitur_restoran(df_restoran)
    tfidf, resto_tfidf_matrix, resto_id_to_row, user_pref_text, user_search_text = (
        build_text_profiles(df_restoran, df_preferensi, df_pencarian)
    )

    train_data, interaction_matrix, user_sim_df, resto_sim_df, trend_scores = (
        build_models(df_ulasan, df_fitur_final)
    )

    # ── SIMPAN SEMUA ARTEFAK ──────────────────────────────────
    print("\n💾 Menyimpan models.pkl, train_data.csv, data_restoran.csv...")

    with open(OUT_MODELS_PKL, 'wb') as f:
        pickle.dump({
            'user_sim_df': user_sim_df,
            'resto_sim_df': resto_sim_df,
            'interaction_matrix': interaction_matrix,
            'df_fitur_final': df_fitur_final,
            'trend_scores': trend_scores,
            'tfidf': tfidf,
            'resto_tfidf_matrix': resto_tfidf_matrix,
            'resto_id_to_row': resto_id_to_row,
            'user_pref_text': dict(user_pref_text),
            'user_search_text': dict(user_search_text),
        }, f)

    train_data.to_csv(OUT_TRAIN_DATA, index=False)
    df_restoran.to_csv(OUT_DATA_RESTORAN, index=False)

    print("\n" + "=" * 55)
    print("  ✅ RETRAIN SELESAI")
    print("=" * 55)
    print(f"  Users     : {len(df_users)}")
    print(f"  Restoran  : {len(df_restoran)}")
    print(f"  Ulasan    : {len(train_data)}")
    print(f"  Preferensi: {len(user_pref_text)} user punya profil")
    print(f"  Pencarian : {len(user_search_text)} user punya profil")
    print("\n  ⚠️  Jangan lupa RESTART FastAPI (uvicorn) supaya")
    print("      model baru ke-load ke memory!")
    print("=" * 55)


if __name__ == "__main__":
    main()