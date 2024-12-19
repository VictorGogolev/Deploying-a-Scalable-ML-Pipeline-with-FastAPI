import json
import requests

# Base URL for the API
base_url = "http://127.0.0.1:8000"

# TODO: send a GET using the URL http://127.0.0.1:8000
r = requests.get(base_url)

# TODO: print the status code
print(f"GET Status Code: {r.status_code}")
# TODO: print the welcome message
print(f"Welcome Message: {r.json()}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
post_url = f"{base_url}/data/"
r = requests.post(post_url, json=data)

# TODO: print the status code
print(f"POST Status Code: {r.status_code}")
# TODO: print the result
print(f"Prediction Result: {r.json()}")

