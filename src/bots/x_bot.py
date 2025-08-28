import os
import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1

# Load secrets from .env
load_dotenv()

# Get values from environment
X_API_KEY = os.getenv("X_API_KEY")
X_API_KEY_SECRET = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_SECRET")

def post_to_x(message: str):
    """
    Posts a tweet (text only) to X using API v2 with OAuth 1.0a User Context.
    """
    if not all([X_API_KEY, X_API_KEY_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET]):
        raise ValueError("Missing one or more X API credentials in .env file")

    url = "https://api.twitter.com/2/tweets"
    
    auth = OAuth1(
        X_API_KEY,
        X_API_KEY_SECRET,
        X_ACCESS_TOKEN,
        X_ACCESS_TOKEN_SECRET
    )

    payload = {
        "text": message
    }

    response = requests.post(url, auth=auth, json=payload)

    if response.status_code in [200, 201]:
        print("✅ Successfully posted to X!")
        print("Response:", response.json())
    else:
        print("❌ Failed to post on X.")
        print("Status Code:", response.status_code)
        print("Response:", response.text)


if __name__ == "__main__":
    # Example usage
    post_to_x("Hello, this is an automated post from my X bot! 🚀")