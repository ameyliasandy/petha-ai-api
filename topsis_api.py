"""
Petha TOPSIS API — Flask
Port: 5001

Endpoint:
  POST /topsis/terdekat  → ranking restoran terdekat pakai TOPSIS
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────
#  CORE TOPSIS
# ─────────────────────────────────────────────

def topsis(matrix: list[list[float]], weights: list[float], impacts: list[str]) -> list[float]:
    """
    matrix  : list of alternatives, each alternative = list of criteria values
    weights : bobot tiap kriteria (akan dinormalisasi otomatis)
    impacts : '+' = benefit (lebih besar lebih baik)
              '-' = cost   (lebih kecil lebih baik)
    return  : list skor TOPSIS (0–1), urutan sama dengan matrix
    """
    X = np.array(matrix, dtype=float)
    n_alt, n_crit = X.shape

    # 1. Normalisasi
    col_norms = np.sqrt((X ** 2).sum(axis=0))
    col_norms[col_norms == 0] = 1          # hindari bagi nol
    R = X / col_norms

    # 2. Bobot
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    V = R * w

    # 3. Ideal positif & negatif
    A_pos = np.where(np.array(impacts) == '+', V.max(axis=0), V.min(axis=0))
    A_neg = np.where(np.array(impacts) == '+', V.min(axis=0), V.max(axis=0))

    # 4. Jarak Euclidean
    D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    # 5. Skor kedekatan
    denom = D_pos + D_neg
    denom[denom == 0] = 1e-9
    scores = D_neg / denom

    return scores.tolist()


# ─────────────────────────────────────────────
#  ENDPOINT: TERDEKAT
# ─────────────────────────────────────────────
#
#  Kriteria:
#    C1  jarak_km      → cost  (-)  bobot 0.35
#    C2  rating        → benefit (+) bobot 0.30
#    C3  halal_score   → benefit (+) bobot 0.25  (certified=3, self_claimed=2, none=1)
#    C4  jumlah_menu   → benefit (+) bobot 0.10
#
#  Request body (JSON):
#  {
#    "restorans": [
#      {
#        "id_restoran": 1,
#        "nama_restoran": "Warung Sate",
#        "kota": "Batam",
#        "rating": 4.5,
#        "status_halal": "certified",   // certified | self_claimed | none
#        "jarak_km": 1.2,
#        "jumlah_menu": 12
#      }, ...
#    ]
#  }
#
#  Response:
#  {
#    "ranked": [
#      { ...restoran fields..., "topsis_score": 0.87, "rank": 1 }, ...
#    ]
#  }

@app.post("/topsis/terdekat")
def topsis_terdekat():
    data = request.get_json(force=True)
    restorans = data.get("restorans", [])

    if not restorans:
        return jsonify({"ranked": [], "message": "Tidak ada data restoran"}), 200

    HALAL_MAP = {"certified": 3, "self_claimed": 2, "none": 1}

    matrix = []
    for r in restorans:
        jarak    = float(r.get("jarak_km", 999))
        rating   = float(r.get("rating") or 0)
        halal    = HALAL_MAP.get(r.get("status_halal", "none"), 1)
        n_menu   = float(r.get("jumlah_menu", 0))
        matrix.append([jarak, rating, halal, n_menu])

    weights = [0.35, 0.30, 0.25, 0.10]
    impacts = ['-',  '+',  '+',  '+']

    scores = topsis(matrix, weights, impacts)

    ranked = []
    for i, r in enumerate(restorans):
        ranked.append({**r, "topsis_score": round(scores[i], 4)})

    ranked.sort(key=lambda x: x["topsis_score"], reverse=True)
    for idx, item in enumerate(ranked):
        item["rank"] = idx + 1

    return jsonify({"ranked": ranked})


# ─────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return jsonify({
        "status": "Petha TOPSIS API aktif",
        "endpoint": "POST /topsis/terdekat"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)