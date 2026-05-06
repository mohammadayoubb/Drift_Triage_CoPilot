"""
generate_predictions.py

This script sends multiple prediction requests to the model service.

It simulates live traffic so drift detection can run.
"""

import requests
import time

URL = "http://127.0.0.1:8000/predict"

PAYLOAD = {
    "age": 35,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
    "pdays_was_999": 1,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "poutcome": "nonexistent",
}


def generate(n=120):
    """
    Send N prediction requests.
    """

    for i in range(n):
        response = requests.post(URL, json=PAYLOAD)

        if response.status_code != 200:
            print("Error:", response.text)
            return

        if i % 10 == 0:
            print(f"Sent {i} requests")

        time.sleep(0.02)


if __name__ == "__main__":
    generate()