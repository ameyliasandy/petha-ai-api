import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from topsis_api import topsis

weights = [0.40, 0.25, 0.20, 0.15]
impacts = ['-', '-', '+', '+']


def test_jumlah_skor_sesuai_jumlah_alternatif():
    matrix = [
        [1, 10000, 5, 12],
        [2, 15000, 4, 10],
        [3, 20000, 3, 8]
    ]

    hasil = topsis(matrix, weights, impacts)

    assert len(hasil) == 3

def test_skor_berada_pada_rentang_0_1():

    matrix = [
        [1,10000,5,12],
        [2,15000,4,10],
        [3,20000,3,8]
    ]

    hasil=topsis(matrix,weights,impacts)

    assert all(0<=x<=1 for x in hasil)

def test_restoran_terbaik_memiliki_skor_tertinggi():

    matrix=[
        [1,10000,5,12],
        [5,40000,3,8]
    ]

    hasil=topsis(matrix,weights,impacts)

    assert hasil[0]>hasil[1]

