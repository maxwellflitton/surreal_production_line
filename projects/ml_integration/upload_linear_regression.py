from surrealml import SurMlFile
import requests
import json


test_load = SurMlFile.load("./linear_test.surml")
url = "http://0.0.0.0:8000/ml/import"

SurMlFile.upload("./linear_test.surml", url, 5)

headers = {
  'Accept': 'application/json',
  'NS': 'test',
  'DB': 'test',
  'Content-Type': 'text/plain',
  'Authorization': 'Basic cm9vdDpyb290'
}

url = "http://0.0.0.0:8000/ml/compute/raw"

payload = json.dumps({
    "id": "Prediction-0.0.1",
    "input": [3200.0, 2.0]
})

response = requests.request("GET", url, headers=headers, data=payload)
print(response.text)

url = "http://0.0.0.0:8000/ml/compute/buffered"

payload = json.dumps({
    "id": "Prediction-0.0.1",
    "input": {
      "squarefoot": 3200.0,
      "num_floors": 2.0
    }
})

response = requests.request("GET", url, headers=headers, data=payload)
print(response.text)


if __name__ == "__main__":
    pass
