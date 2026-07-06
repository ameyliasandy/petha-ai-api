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

@app.post("/topsis/terdekat")
def topsis_terdekat():
    data = request.get_json(force=True)
    restorans = data.get("restorans", [])

    if not restorans:
        return jsonify({"ranked": [], "message": "Tidak ada data restoran"}), 200

    matrix = []

    for r in restorans:

        jarak = float(r.get("jarak_km",999))

        harga = float(r.get("harga",999999))

        rating = float(r.get("rating",0))

        jam = float(r.get("jam_operasional",0))

        matrix.append([
            jarak,
            harga,
            rating,
            jam
        ])

        weights = [
            0.40,
            0.25,
            0.20,
            0.15
        ]

        impacts = [
            '-',
            '-',
            '+',
            '+'
        ]