import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from topsis_api import app

client = app.test_client()


def test_endpoint_status_200():
    """Endpoint mengembalikan HTTP 200"""

    response = client.post(
        "/topsis/terdekat",
        json={
            "restorans": [
                {
                    "nama_restoran": "Resto A",
                    "jarak_km": 1,
                    "harga": 10000,
                    "rating": 5,
                    "jam_operasional": 12
                }
            ]
        }
    )

    assert response.status_code == 200


def test_response_memiliki_key_ranked():
    """Response harus memiliki key ranked"""

    response = client.post(
        "/topsis/terdekat",
        json={
            "restorans": [
                {
                    "nama_restoran": "Resto A",
                    "jarak_km": 1,
                    "harga": 10000,
                    "rating": 5,
                    "jam_operasional": 12
                }
            ]
        }
    )

    data = response.get_json()

    assert "ranked" in data


def test_jumlah_hasil_sama_dengan_input():
    """Jumlah output harus sama dengan jumlah restoran yang dikirim"""

    response = client.post(
        "/topsis/terdekat",
        json={
            "restorans": [
                {
                    "nama_restoran": "A",
                    "jarak_km": 1,
                    "harga": 10000,
                    "rating": 5,
                    "jam_operasional": 12
                },
                {
                    "nama_restoran": "B",
                    "jarak_km": 2,
                    "harga": 15000,
                    "rating": 4,
                    "jam_operasional": 10
                }
            ]
        }
    )

    data = response.get_json()

    assert len(data["ranked"]) == 2


def test_hasil_sudah_diurutkan():
    """Restoran terbaik harus berada di urutan pertama"""

    response = client.post(
        "/topsis/terdekat",
        json={
            "restorans": [
                {
                    "nama_restoran": "Buruk",
                    "jarak_km": 5,
                    "harga": 50000,
                    "rating": 3,
                    "jam_operasional": 8
                },
                {
                    "nama_restoran": "Bagus",
                    "jarak_km": 1,
                    "harga": 10000,
                    "rating": 5,
                    "jam_operasional": 12
                }
            ]
        }
    )

    data = response.get_json()

    assert data["ranked"][0]["nama_restoran"] == "Bagus"


def test_input_kosong():
    """Input kosong harus menghasilkan ranked kosong"""

    response = client.post(
        "/topsis/terdekat",
        json={
            "restorans": []
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["ranked"] == []