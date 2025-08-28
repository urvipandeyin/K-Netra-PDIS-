import os
import requests
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()

# Get values from environment
INSTAGRAM_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN")
INSTAGRAM_USER_ID = os.getenv("IG_USER_ID")  # numeric Instagram Business/Creator ID

def post_to_instagram(message: str, image_url: str):
    """
    Posts an image with a caption to Instagram using Graph API.
    (Instagram requires media upload -> publish step.)
    """
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_USER_ID:
        raise ValueError("Missing INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_USER_ID in .env file")

    # Step 1: Create media object (upload image URL + caption)
    create_url = f"https://graph.facebook.com/v21.0/{INSTAGRAM_USER_ID}/media"
    create_payload = {
        "image_url": image_url,
        "caption": message,
        "access_token": INSTAGRAM_ACCESS_TOKEN
    }
    create_resp = requests.post(create_url, data=create_payload)
    if create_resp.status_code != 200:
        print("❌ Failed to create media object.")
        print("Status Code:", create_resp.status_code)
        print("Response:", create_resp.text)
        return
    
    media_id = create_resp.json().get("id")
    print("✅ Media object created:", media_id)

    # Step 2: Publish the media object
    publish_url = f"https://graph.facebook.com/v21.0/{INSTAGRAM_USER_ID}/media_publish"
    publish_payload = {
        "creation_id": media_id,
        "access_token": INSTAGRAM_ACCESS_TOKEN
    }
    publish_resp = requests.post(publish_url, data=publish_payload)

    if publish_resp.status_code == 200:
        print("✅ Successfully posted to Instagram!")
        print("Response:", publish_resp.json())
    else:
        print("❌ Failed to publish media.")
        print("Status Code:", publish_resp.status_code)
        print("Response:", publish_resp.text)


if __name__ == "__main__":
    # Example usage
    post_to_instagram(
        "Hello Instagram 🚀 Posting via my bot!",
        image_url="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    )
