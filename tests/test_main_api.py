import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "total_users" in data
def test_get_rekomendasi():

    response = client.get("/rekomendasi/1")

    assert response.status_code == 200

    data = response.json()

    assert "rekomendasi" in data

def test_post_rekomendasi():

    response = client.post(
        "/rekomendasi",
        json={
            "user_id":1,
            "n_recommend":5,
            "w_cf":0.4,
            "w_cbf":0.4,
            "w_trend":0.2
        }
    )

    assert response.status_code==200

    data=response.json()

    assert "rekomendasi" in data

def test_total_rekomendasi():

    response=client.get("/rekomendasi/1?n=5")

    data=response.json()

    assert len(data["rekomendasi"])<=5

def test_struktur_json():

    response=client.get("/rekomendasi/1")

    data=response.json()

    if len(data["rekomendasi"])>0:

        resto=data["rekomendasi"][0]

        assert "id_restoran" in resto
        assert "nama_restoran" in resto
        assert "status_halal" in resto
        assert "score" in resto
