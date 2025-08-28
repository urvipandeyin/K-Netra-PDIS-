import os
import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1

# Load environment variables
load_dotenv()

# Facebook
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")
# Instagram
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN")
# X / Twitter
X_CONSUMER_KEY = os.getenv("X_CONSUMER_KEY")
X_CONSUMER_SECRET = os.getenv("X_CONSUMER_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_SECRET")


def takedown_facebook(post_id: str):
    """
    Deletes a Facebook post using Graph API.
    """
    url = f"https://graph.facebook.com/{post_id}"
    params = {"access_token": FB_PAGE_ACCESS_TOKEN}
    resp = requests.delete(url, params=params)
    if resp.status_code == 200:
        print(f"✅ Deleted Facebook post {post_id}")
    else:
        print(f"❌ Failed FB takedown on {post_id}: {resp.text}")


def takedown_instagram(media_id: str):
    """
    Deletes an Instagram media post using Graph API.
    """
    url = f"https://graph.facebook.com/v21.0/{media_id}"
    params = {"access_token": IG_ACCESS_TOKEN}
    resp = requests.delete(url, params=params)
    if resp.status_code == 200:
        print(f"✅ Deleted Instagram media {media_id}")
    else:
        print(f"❌ Failed IG takedown on {media_id}: {resp.text}")


def takedown_x(tweet_id: str):
    """
    Deletes a tweet using X API v2 with OAuth 1.0a User Context.
    """
    url = f"https://api.x.com/2/tweets/{tweet_id}"
    auth = OAuth1(
        X_CONSUMER_KEY,
        X_CONSUMER_SECRET,
        X_ACCESS_TOKEN,
        X_ACCESS_TOKEN_SECRET
    )
    resp = requests.delete(url, auth=auth)
    if resp.status_code in [200, 204]:
        print(f"✅ Deleted X tweet {tweet_id}")
    else:
        print(f"❌ Failed X takedown on {tweet_id}: {resp.text}")


if __name__ == "__main__":
    # Example usage
    takedown_facebook("1234567890123456")
    takedown_instagram("987654321098765")
    takedown_x("1357913579135791")