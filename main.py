from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Petha Recommendation API")

# Izinkan Laravel mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LOAD MODEL & DATA ─────────────────────────────────────────
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

user_sim_df = models['user_sim_df']
resto_sim_df = models['resto_sim_df']
interaction_matrix = models['interaction_matrix']
df_fitur_final = models['df_fitur_final']
trend_scores = models['trend_scores']

# ── BARU: artefak untuk fitur riwayat pencarian & preferensi ──
tfidf = models['tfidf']
resto_tfidf_matrix = models['resto_tfidf_matrix']
resto_id_to_row = models['resto_id_to_row']
user_pref_text = models['user_pref_text']
user_search_text = models['user_search_text']

train_data = pd.read_csv('train_data.csv')
df_restoran = pd.read_csv('data_restoran.csv')

print(f"✅ Model loaded. Users: {len(user_sim_df)}, Restoran: {len(df_fitur_final)}")
print(f"✅ Profil preferensi: {len(user_pref_text)} user, Profil pencarian: {len(user_search_text)} user")


# ── HELPER: Bersihkan nilai NaN sebelum dikirim sebagai JSON ──

def safe_value(val, default="none"):
    """Ganti NaN/None dengan nilai default agar JSON-compliant."""
    if pd.isna(val):
        return default
    return val


# ── FUNGSI PREDIKSI (PERSIS DARI COLAB) ──────────────────────

def predict_cf(user_id, resto_id, n_similar=10):
    if user_id not in user_sim_df.index:
        return 3.0
    if resto_id not in interaction_matrix.columns:
        return 3.0
    sim_scores = user_sim_df[user_id].drop(user_id)
    top_users = sim_scores.nlargest(n_similar)
    ratings_sim = interaction_matrix.loc[top_users.index, resto_id]
    mask = ratings_sim > 0
    if mask.sum() == 0:
        mean_val = interaction_matrix.loc[user_id].replace(0, np.nan).mean()
        return mean_val if not np.isnan(mean_val) else 3.0
    num = (top_users[mask] * ratings_sim[mask]).sum()
    den = top_users[mask].abs().sum()
    return round(float(num / den), 2) if den != 0 else 3.0


def get_text_sim_score(user_id, resto_id, text_dict):
    """Cosine similarity antara profil teks user (preferensi/pencarian) dan konten resto → skor 1-5."""
    text = text_dict.get(user_id, '')
    if not text or not str(text).strip():
        return None
    if resto_id not in resto_id_to_row:
        return None
    user_vec = tfidf.transform([text])
    resto_vec = resto_tfidf_matrix[resto_id_to_row[resto_id]]
    sim = cosine_similarity(user_vec, resto_vec)[0][0]
    return round(1 + sim * 4, 2)


def predict_cbf(user_id, resto_id, w_hist=0.5, w_pref=0.3, w_search=0.2):
    """
    CBF gabungan 3 sinyal:
    - Riwayat rating (similarity antar resto yang disukai)
    - Preferensi eksplisit user (tabel preferensi_users)
    - Riwayat pencarian user (tabel pencarian)
    Bobot dinormalisasi ulang jika ada sinyal yang kosong (user baru dsb).
    """
    scores, weights = {}, {}

    # 1. Riwayat rating
    ur = train_data[train_data['user_id'] == user_id]
    if len(ur) > 0 and resto_id in resto_sim_df.columns:
        liked = ur[ur['rating'] >= 4]['id_restoran'].tolist()
        if not liked:
            liked = ur['id_restoran'].tolist()
        sims = [resto_sim_df.loc[lid, resto_id]
                for lid in liked if lid in resto_sim_df.index]
        if sims:
            scores['hist'] = 1 + np.mean(sims) * 4
            weights['hist'] = w_hist

    # 2. Preferensi eksplisit
    pref_score = get_text_sim_score(user_id, resto_id, user_pref_text)
    if pref_score is not None:
        scores['pref'] = pref_score
        weights['pref'] = w_pref

    # 3. Riwayat pencarian
    search_score = get_text_sim_score(user_id, resto_id, user_search_text)
    if search_score is not None:
        scores['search'] = search_score
        weights['search'] = w_search

    if not scores:
        return 3.0

    total_w = sum(weights.values())
    final = sum(scores[k] * weights[k] for k in scores) / total_w
    return round(final, 2)


def predict_hybrid(user_id, resto_id, w_cf=0.4, w_cbf=0.4, w_trend=0.2):
    total = w_cf + w_cbf + w_trend
    w_cf, w_cbf, w_trend = w_cf / total, w_cbf / total, w_trend / total
    cf_n = (max(1.0, min(5.0, predict_cf(user_id, resto_id))) - 1) / 4
    cbf_n = (max(1.0, min(5.0, predict_cbf(user_id, resto_id))) - 1) / 4
    tr_n = trend_scores.get(resto_id, 0.0)
    return round((w_cf * cf_n) + (w_cbf * cbf_n) + (w_trend * tr_n), 4)


def rec_hybrid(user_id, n_recommend=5, w_cf=0.4, w_cbf=0.4, w_trend=0.2):
    already = set(train_data[train_data['user_id'] == user_id]['id_restoran'])
    candidates = list(set(df_fitur_final['id_restoran']) - already)
    scores = {r: predict_hybrid(user_id, r, w_cf, w_cbf, w_trend) for r in candidates}
    return [(r, s) for r, s in
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommend]]


def build_rekomendasi_list(hasil):
    """Bangun list rekomendasi dengan nilai yang sudah aman untuk JSON."""
    rekomendasi = []
    for resto_id, score in hasil:
        info = df_restoran[df_restoran['id_restoran'] == resto_id]
        if info.empty:
            continue
        row = info.iloc[0]
        rekomendasi.append({
            "id_restoran": int(resto_id),
            "nama_restoran": safe_value(row['nama_restoran'], "-"),
            "status_halal": safe_value(row['status_halal'], "none"),
            "score": float(score)
        })
    return rekomendasi


# ── SCHEMA REQUEST ────────────────────────────────────────────

class RekomendasiRequest(BaseModel):
    user_id: int
    n_recommend: int = 5
    w_cf: float = 0.4
    w_cbf: float = 0.4
    w_trend: float = 0.2


# ── ENDPOINTS ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Petha Recommendation API aktif", "total_users": len(user_sim_df)}


@app.post("/rekomendasi")
def get_rekomendasi(req: RekomendasiRequest):
    try:
        hasil = rec_hybrid(
            req.user_id,
            n_recommend=req.n_recommend,
            w_cf=req.w_cf,
            w_cbf=req.w_cbf,
            w_trend=req.w_trend
        )
        rekomendasi = build_rekomendasi_list(hasil)
        return {
            "user_id": req.user_id,
            "rekomendasi": rekomendasi
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/rekomendasi/{user_id}")
def get_rekomendasi_simple(user_id: int, n: int = 5):
    """Versi GET sederhana untuk testing cepat lewat browser"""
    try:
        hasil = rec_hybrid(user_id, n_recommend=n)
        rekomendasi = build_rekomendasi_list(hasil)
        return {"user_id": user_id, "rekomendasi": rekomendasi}
    except Exception as e:
        return {"error": str(e)}