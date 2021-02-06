import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://08733603-df61-45b2-9ba1-1e267c88d389.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = '1ZFLDS0VQ1tHyZRfT96v8kGs5haaQ48s'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
           "age": 75, 
           "anaemia": 0, 
           "creatinine_phosphokinase": 582, 
           "diabetes": 0, 
           "ejection_fraction": 20, 
           "high_blood_pressure": 1, 
           "platelets": 265000, 
           "serum_creatinine": 1.9, 
           "serum_sodium": 130, 
           "sex": 1, 
           "smoking": 0,
           "time": 4
          },
          {
           "age": 51, 
           "anaemia": 0, 
           "creatinine_phosphokinase": 1380, 
           "diabetes": 0, 
           "ejection_fraction": 25, 
           "high_blood_pressure": 1, 
           "platelets": 271000, 
           "serum_creatinine": 0.9, 
           "serum_sodium": 130, 
           "sex": 1, 
           "smoking": 0,
           "time": 38
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
print("++++++++++++++++++++++++++++++")
print("Expected result: [true, true], where 'true' means '1' as result in the 'DEATH_EVENT' column")


