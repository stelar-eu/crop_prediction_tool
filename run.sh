#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <token> <endpoint_url> <id> <signature>"
    exit 2
fi

token="$1"
endpoint_url="$2"
id="$3"
signature="$4"

echo "[STELAR INFO] Performing cURL to fetch tool input..."
response=$(curl -s -L -X GET -H "Authorization: $token" "$endpoint_url/api/v2/task/$id/$signature/input")

success=$(echo "$response" | jq -r '.success')
if [ "$success" != "true" ]; then
    echo "[STELAR INFO] Failed to fetch input from API!"
    exit 3
fi

result=$(echo "$response" | jq '.result')
echo "$result" > input.json
echo "[STELAR INFO] Input fetched and written to input.json"

# Run tool
python -u main.py input.json output.json
output_json=$(<output.json)
echo "$output_json"

echo "[STELAR INFO] Posting tool output to KLMS API..."
response=$(curl -s -X POST -H "Content-Type: application/json" "$endpoint_url/api/v2/task/$id/$signature/output" -d "$output_json")

echo "[STELAR INFO] Output upload response:"
echo "$response"
success=$(echo "$response" | jq -r '.success')
if [ "$success" != "true" ]; then
    echo "[STELAR INFO] Failed to push output to API!"
    exit 4
fi

echo "[STELAR INFO] Output successfully posted!"
