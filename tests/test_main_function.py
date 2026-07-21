import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import (
    safe_value,
    build_rekomendasi_list,
    predict_hybrid
)
def test_safe_value_normal():

    assert safe_value("Halal") == "Halal"

def test_safe_value_none():

    assert safe_value(None) == "none"

import numpy as np

def test_safe_value_nan():

    assert safe_value(np.nan) == "none"

def test_predict_hybrid_return_type():

    score = predict_hybrid(1, 1)

    assert isinstance(score, float)

def test_predict_hybrid_range():

    score = predict_hybrid(1,1)

    assert 0 <= score <= 1

def test_build_rekomendasi_list():

    hasil = [
        (1,0.95)
    ]

    rekom = build_rekomendasi_list(hasil)

    assert isinstance(rekom,list)