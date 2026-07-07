"""
Script untuk cek apakah restoran dengan trend score tinggi
lebih sering muncul di rekomendasi banyak user.

Cara pakai:
1. Pastikan FastAPI sudah jalan di http://127.0.0.1:8001
2. Jalankan: python cek_trend.py
"""

import requests
from collections import Counter

API_URL = "http://127.0.0.1:8001"

# Daftar user_id yang mau dicek (sesuaikan dengan user yang ada di database kamu)
USER_IDS = list(range(1, 33))  # cek user id 1 sampai 32

# Resto dengan trend score tertinggi (dari hasil Colab kemarin)
TOP_TREND_RESTO = {7: 0.8815, 23: 0.7778, 35: 0.7478, 20: 0.7333, 33: 0.7125}

def main():
    semua_rekomendasi = []
    error_count = 0

    print(f"Mengecek rekomendasi untuk {len(USER_IDS)} user...\n")

    for uid in USER_IDS:
        try:
            resp = requests.get(f"{API_URL}/rekomendasi/{uid}", params={"n": 5}, timeout=5)
            data = resp.json()

            if "error" in data or "rekomendasi" not in data:
                error_count += 1
                continue

            for item in data["rekomendasi"]:
                semua_rekomendasi.append(item["id_restoran"])

        except Exception as e:
            print(f"  User {uid}: gagal -> {e}")
            error_count += 1

    print(f"Total user berhasil dicek : {len(USER_IDS) - error_count}")
    print(f"Total user error          : {error_count}")
    print(f"Total slot rekomendasi    : {len(semua_rekomendasi)}\n")

    counter = Counter(semua_rekomendasi)

    print("=" * 60)
    print("FREKUENSI KEMUNCULAN SETIAP RESTORAN DI TOP-5 REKOMENDASI")
    print("=" * 60)
    print(f"{'ID Resto':<10}{'Muncul':<10}{'Trend Score':<15}{'Kategori'}")
    print("-" * 60)

    for resto_id, jumlah in counter.most_common():
        trend = TOP_TREND_RESTO.get(resto_id, "-")
        kategori = "🔥 TOP TREND" if resto_id in TOP_TREND_RESTO else ""
        print(f"{resto_id:<10}{jumlah:<10}{str(trend):<15}{kategori}")

    print("\n" + "=" * 60)
    print("ANALISIS")
    print("=" * 60)

    avg_muncul_top_trend = []
    avg_muncul_lainnya = []

    for resto_id, jumlah in counter.items():
        if resto_id in TOP_TREND_RESTO:
            avg_muncul_top_trend.append(jumlah)
        else:
            avg_muncul_lainnya.append(jumlah)

    if avg_muncul_top_trend:
        rata_top = sum(avg_muncul_top_trend) / len(avg_muncul_top_trend)
        print(f"Rata-rata kemunculan resto TOP TREND     : {rata_top:.2f} kali")
    else:
        print("Tidak ada resto top-trend yang muncul sama sekali di rekomendasi siapapun.")
        rata_top = 0

    if avg_muncul_lainnya:
        rata_lain = sum(avg_muncul_lainnya) / len(avg_muncul_lainnya)
        print(f"Rata-rata kemunculan resto LAINNYA        : {rata_lain:.2f} kali")
    else:
        rata_lain = 0

    if rata_top > rata_lain:
        print("\n✅ Trend score TERBUKTI berpengaruh — resto top-trend lebih sering direkomendasikan.")
    elif rata_top == rata_lain:
        print("\n⚠️  Tidak ada perbedaan signifikan antara resto top-trend dan lainnya.")
    else:
        print("\n❌ Resto top-trend justru LEBIH JARANG muncul — perlu investigasi lebih lanjut.")


if __name__ == "__main__":
    main()
