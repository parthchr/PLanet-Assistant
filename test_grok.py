import requests
import os
from dotenv import load_dotenv

load_dotenv()
PLANET_API_KEY = os.getenv("PLANET_API_KEY")

url = "https://api.planet.com/data/v1/quick-search"
headers = {"Content-Type": "application/json"}
auth = (PLANET_API_KEY, "")

body = {
    "item_types": ["PSScene"],
    "filter": {
        "type": "AndFilter",
        "config": [
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": "2024-01-01T00:00:00Z",
                    "lte": "2024-01-10T23:59:59Z"
                }
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {
                    "lte": 0.1
                }
            }
        ]
    }
}

r = requests.post(url, auth=auth, headers=headers, json=body)
print(r.status_code, r.text)
