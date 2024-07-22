import requests

def query(filename, model_url, headers):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(model_url, headers=headers, data=data, json={'wait_for_model':'True'})
    return response.json()
