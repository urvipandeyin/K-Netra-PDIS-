import argparse
from src.bots.comment_bot import comment_facebook, comment_instagram, comment_x

parser = argparse.ArgumentParser(description="Send automated comment alerts to flagged posts.")
parser.add_argument("--message", required=True, type=str, help="Comment message to post")
parser.add_argument(
    "--tier",
    required=True,
    choices=["AUTO_INTERVENTION", "HUMAN_REVIEW", "LOG_ONLY"],
    help="Tier of alert"
)
parser.add_argument(
    "--platform_ids",
    required=True,
    nargs="+",
    help="List of platform:id to comment, e.g., fb:12345 ig:98765 x:13579 tg:-1001234567890"
)

args = parser.parse_args()

# Only act if tier is flagged for action
if args.tier != "LOG_ONLY":
    for pid in args.platform_ids:
        try:
            platform, identifier = pid.split(":")
        except ValueError:
            print(f"❌ Invalid platform_id format: {pid}")
            continue

        if platform.lower() == "fb":
            comment_facebook(identifier, args.message)
        elif platform.lower() == "ig":
            comment_instagram(identifier, args.message)
        elif platform.lower() == "x":
            comment_x(identifier, args.message)
        else:
            print(f"❌ Unknown platform: {platform}")
else:
    print("Tier is LOG_ONLY — skipping comment alerts.")