import json

import httpx

url = "http://localhost:8000/query"
query = "hi"

print(f"\n[Query]: {query}\n[Answer]: ", end="", flush=True)

with httpx.Client(timeout=30.0) as client:
    with client.stream("POST", url, json={"query": query}) as response:
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "token" in data:
                    print(data["token"], end="", flush=True)
                elif "pipeline_total_ms" in data or "routing" in data:
                    print(f"\n\n[Metadata]: {json.dumps(data, indent=2)}")
print()
