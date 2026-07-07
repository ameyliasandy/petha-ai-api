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

    print("\n====================================")
    print("Request TOPSIS diterima")
    print(f"Jumlah restoran : {len(restorans)}")
    print("====================================")

    if not restorans:
        print("  Tidak ada data restoran")
        return jsonify({"ranked": [], "message": "Tidak ada data restoran"}), 200

    # 1. Kumpulkan semua data
    matrix = []

    for r in restorans:
        jarak = float(r.get("jarak_km", 999))
        harga = float(r.get("harga", 999999))
        rating = float(r.get("rating", 0))
        jam = float(r.get("jam_operasional", 0))

        print(
            f"{r.get('nama_restoran')} | "
            f"Jarak={jarak} km | "
            f"Harga={harga} | "
            f"Rating={rating} | "
            f"Jam={jam}"
        )

        matrix.append([
            jarak,
            harga,
            rating,
            jam
        ])

    print(f"\n Matrix terkumpul: {len(matrix)} data")

    # 2. Baru hitung TOPSIS
    print("\nMenghitung TOPSIS...")

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

    scores = topsis(matrix, weights, impacts)

    # 3. Gabungkan hasil
    ranked = []

    for restoran, score in zip(restorans, scores):
        item = dict(restoran)
        item["skor_topsis"] = round(float(score), 6)
        ranked.append(item)
        
        print(f"{restoran['nama_restoran']} -> {score:.6f}")

    ranked.sort(
        key=lambda x: x["skor_topsis"],
        reverse=True
    )

    print("\n=== Ranking ===")
    for i, r in enumerate(ranked, start=1):
        print(f"{i}. {r['nama_restoran']} ({r['skor_topsis']:.6f})")
    print("============================\n")

    # 4. Return DI PALING BAWAH
    return jsonify({
        "ranked": ranked
    })


if __name__ == "__main__":
    print("=" * 50)
    print("PETHA AI TOPSIS API")
    print("=" * 50)
    print("Server berjalan...")
    print("URL        : http://127.0.0.1:5001")
    print("Endpoint   : POST /topsis/terdekat")
    print("Menunggu request dari Laravel...")
    print("=" * 50)

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True,
        use_reloader=False
    )