import os
import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1

# Load secrets
load_dotenv()

# Facebook
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")
# Instagram
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN")
# X / Twitter (OAuth 1.0a User Context)
X_CONSUMER_KEY = os.getenv("X_CONSUMER_KEY")
X_CONSUMER_SECRET = os.getenv("X_CONSUMER_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_SECRET")


def comment_facebook(post_id: str, message: str):
    url = f"https://graph.facebook.com/{post_id}/comments"
    payload = {"message": message, "access_token": FB_PAGE_ACCESS_TOKEN}
    resp = requests.post(url, data=payload)
    if resp.status_code == 200:
        print(f"✅ Commented on Facebook post {post_id}")
    else:
        print(f"❌ Failed FB comment on {post_id}: {resp.text}")


def comment_instagram(media_id: str, message: str):
    url = f"https://graph.facebook.com/v21.0/{media_id}/comments"
    payload = {"message": message, "access_token": IG_ACCESS_TOKEN}
    resp = requests.post(url, data=payload)
    if resp.status_code == 200:
        print(f"✅ Commented on Instagram media {media_id}")
    else:
        print(f"❌ Failed IG comment on {media_id}: {resp.text}")

def comment_x(tweet_id: str, message: str):
    if not tweet_id:
        print("❌ Tweet ID is missing — cannot reply on X")
        return

    url = "https://api.twitter.com/1.1/statuses/update.json"
    auth = OAuth1(
        X_CONSUMER_KEY,
        X_CONSUMER_SECRET,
        X_ACCESS_TOKEN,
        X_ACCESS_TOKEN_SECRET
    )
    payload = {
        "status": message,
        "in_reply_to_status_id": str(tweet_id),
        "auto_populate_reply_metadata": True
    }

    resp = requests.post(url, auth=auth, params=payload)
    if resp.status_code in [200, 201]:
        print(f"✅ Commented on X tweet {tweet_id}")
    else:
        print(f"❌ Failed X comment on {tweet_id}: {resp.text}")