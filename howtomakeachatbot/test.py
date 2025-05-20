
import requests
import json

def getBembedingVector(textInput ="Ma trận Phim được phát hành vào năm 2000. Nội dung nói về …"):
    url = "http://103.141.141.143:11434/api/embed"

    payload = json.dumps({
    "model": "gemma2:27b",
    "input": textInput
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    
    return json.loads(response.text)["embeddings"][0]



#ussage print(getBembedingVector('Hello'))