import numpy as np
import pandas as pd
import requests
import json

# Paths
INDEX = 2
INPUT_FILEPATH = f'./inputs/json_input_{INDEX}.json'
RESULT_FILEPATH = f'./true/json_trueOutput_{INDEX}.json'

# Load input
with open(INPUT_FILEPATH) as o1:
    parsed = json.loads(o1.read())
input_df = pd.json_normalize(parsed)
input = input_df.values

# Serialize the data into json and send the request to the model
data_string = json.dumps(input.tolist(), ensure_ascii=False)
response = requests.put('http://127.0.0.1:5000/apartments/' + data_string)

# Print response
print('response: ', response.json()['data'])

# Show true values
# with open(RESULT_FILEPATH) as o1:
#     true = json.loads(o1.read())
# print(true)