import requests
import json

# The URL where your API is running
url = "http://127.0.0.1:8000/predict"

# Sample Data (simulating a new ticket coming in)
payload = {
    "ticket_id": "TEST-999",
    "subject": "Login failed",
    "description": "I cannot access my dashboard using the enterprise login.",
    "email": "john.doe@enterprise.org",
    "channel": "Web Form"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("\n✅ API Response Received:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n❌ Could not connect. Is 'api_app.py' running?")