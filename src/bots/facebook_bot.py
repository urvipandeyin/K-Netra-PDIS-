import os
import requests
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()

# Get values from environment
PAGE_ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")
PAGE_ID = os.getenv("FB_PAGE_ID")

def post_to_facebook(message: str, image_url: str = None):
    """
    Posts a message (and optional image) to a Facebook Page using Graph API.
    """
    if not PAGE_ACCESS_TOKEN or not PAGE_ID:
        raise ValueError("Missing PAGE_ACCESS_TOKEN or PAGE_ID in .env file")

    try:
        if image_url:
            # Posting with image
            url = f"https://graph.facebook.com/{PAGE_ID}/photos"
            payload = {
                "caption": message,
                "url": image_url,
                "access_token": PAGE_ACCESS_TOKEN
            }
        else:
            # Posting text-only
            url = f"https://graph.facebook.com/{PAGE_ID}/feed"
            payload = {
                "message": message,
                "access_token": PAGE_ACCESS_TOKEN
            }

        response = requests.post(url, data=payload)
        response_data = response.json()

        if response.status_code == 200 and "id" in response_data:
            print("✅ Successfully posted to Facebook Page!")
            print("Post ID:", response_data["id"])
        else:
            print("❌ Failed to post on Facebook.")
            print("Status Code:", response.status_code)
            print("Response:", response_data)

    except Exception as e:
        print("🔥 Exception occurred while posting:", str(e))


if __name__ == "__main__":
    # Example usage
    post_to_facebook(
        "Hello, this is an automated post from my bot! 🚀",
        image_url="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    )