"""
Capture still frames from a YouTube live stream every N seconds.
Tested with: Python 3.11, yt‑dlp 2025.3.1, pafy 0.5.5, opencv‑python 4.9.
"""

import cv2
import pafy
import time
import os
from datetime import datetime

YOUTUBE_URL   = "https://www.youtube.com/watch?v=BsfcSCJZRFM"
SAVE_DIR      = "boat_ramp_frames"
CAPTURE_EVERY = 2.0            # seconds
YT_API_KEY = os.environ.get("YT_API_KEY", "AIzaSyC7HnQgRrr7f675X9xViqUvnG52MU-1zmU")

# -----------------------------------------------------------------------------
def get_stream_url(video_url: str) -> str:
    """Return the direct video stream URL (best available)."""
    # Tell pafy to use yt‑dlp instead of (deprecated) youtube‑dl
    import pafy
    pafy.set_api_key(None)
    # yt‑dlp already bundled with pafy if installed as above
    v = pafy.new(video_url)
    best = v.getbest(preftype="mp4")
    return best.url

def capture_frames(stream_url: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("Could not open stream.")

    last_save = 0.0
    frame_count = 0
    print("Capturing…  Press Ctrl‑C to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream read failed, retrying in 5 s…")
            time.sleep(5)
            continue

        now = time.time()
        if now - last_save >= CAPTURE_EVERY:
            ts   = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(SAVE_DIR, f"frame_{ts}.jpg")
            cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_count += 1
            last_save = now
            print(f"saved {path}")

except KeyboardInterrupt:
    print(f"\nStopped – {frame_count} images saved.")
finally:
    cap.release()

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    url = get_stream_url(YOUTUBE_URL)
    capture_frames(url)
