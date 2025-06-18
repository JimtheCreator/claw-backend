import os
import json
import requests

# Ensure you have the environment variables set
if 'DEEPSEEK_R1_OPENROUTER_API_KEY' not in os.environ:
    raise EnvironmentError("Please set the DEEPSEEK_R1_OPENROUTER_API_KEY environment variable.")

if 'DEEPSEEK_R1_OPENROUTER_API_URL' not in os.environ:
    raise EnvironmentError("Please set the DEEPSEEK_R1_OPENROUTER_API_URL environment variable.")

response = requests.post(
    url=os.getenv('DEEPSEEK_R1_OPENROUTER_API_URL'),
    headers={
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_R1_OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    },
    data=json.dumps({
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ]
    })
)

# Only print the model's response content
if response.status_code == 200:
    result = response.json()
    message = result['choices'][0]['message']['content']
    print(message)
else:
    print("Error:", response.text) 
