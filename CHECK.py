import requests

api_key = "579b464db66ec23bdd000001d9306f97fa70436a780988d427cb8a91"
resource_id = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
url = f"https://api.data.gov.in/resource/{resource_id}"

params = {
    "api-key": api_key,
    "format": "json",
    "limit": 10
}

resp = requests.get(url, params=params)
print("Status code:", resp.status_code)
try:
    records = resp.json()['records']
    print("Sample columns:", records[0].keys() if records else "No records found.")
except Exception as e:
    print("Error or unexpected schema:", e)
    print("Full response:", resp.text)