"""
Handles the extraction of frames from a video stream and saves them to disk.
"""
import cv2
import time
import os
from datetime import datetime
import sys
import io
import hashlib  # Added for hashing frames

class FrameExtractor:
    """Extracts and saves frames from a video stream at specified intervals."""
    def __init__(self, save_dir: str = "boat_ramp_frames", capture_every_sec: float = 2.0):
        """Initializes the FrameExtractor.

        Args:
            save_dir: Directory to save the extracted frames.
            capture_every_sec: Interval in seconds at which to capture frames.
        """
        self.save_dir = save_dir
        self.capture_every_sec = capture_every_sec
        self.last_saved_frame_hash = None  # Initialize hash for comparison
        os.makedirs(self.save_dir, exist_ok=True)

    def _calculate_frame_hash(self, frame) -> str:
        """Calculates a SHA256 hash for a given frame."""
        if frame is None:
            return ""
        return hashlib.sha256(frame.tobytes()).hexdigest()

    def capture_frames_from_stream(self, stream_url: str):
        """Captures frames from the given stream URL and saves them.

        Handles FFmpeg messages by printing a user-friendly note for common CDN warnings
        and printing other FFmpeg messages to stderr.

        Args:
            stream_url: The URL of the video stream to capture frames from.

        Raises:
            RuntimeError: If the stream cannot be opened.
        """
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open stream: {stream_url}")

        last_save_time = 0.0
        frame_count = 0
        print("Capturing… Press Ctrl-C to stop.")

        original_stderr = sys.stderr
        stderr_capture = io.StringIO()

        try:
            while True:
                sys.stderr = stderr_capture
                ok, frame = cap.read()
                sys.stderr = original_stderr

                captured_stderr_output = stderr_capture.getvalue()
                if captured_stderr_output:
                    if "Cannot reuse HTTP connection" in captured_stderr_output:
                        print("Info: Video stream source may have shifted (common CDN behavior). Continuing capture.")
                    else:
                        print(f"FFmpeg message: {captured_stderr_output.strip()}", file=sys.stderr)
                
                stderr_capture.seek(0)
                stderr_capture.truncate(0)

                if not ok:
                    print("Stream read failed, retrying in 5s…")
                    self.last_saved_frame_hash = None  # Reset hash on stream error to avoid false positives on reconnect
                    time.sleep(5)
                    continue

                current_time = time.time()
                if current_time - last_save_time >= self.capture_every_sec:
                    current_frame_hash = self._calculate_frame_hash(frame)
                    
                    if current_frame_hash == self.last_saved_frame_hash:
                        print(f"Skipped saving duplicate frame at {datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
                        last_save_time = current_time
                        continue

                    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    file_path = os.path.join(self.save_dir, f"frame_{timestamp}.jpg")
                    cv2.imwrite(file_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    frame_count += 1
                    last_save_time = current_time
                    self.last_saved_frame_hash = current_frame_hash
                    print(f"Saved {file_path}")

        except KeyboardInterrupt:
            print(f"\nStopped – {frame_count} images saved.")
        finally:
            sys.stderr = original_stderr  # Ensure stderr is restored
            cap.release()
            print("Video stream capture released.")
