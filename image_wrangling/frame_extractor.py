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
    def __init__(self, save_dir: str = "boat_ramp_frames", capture_every_sec: float = 2.0, cv_processor=None):
        """Initializes the FrameExtractor.

        Args:
            save_dir: Directory to save the extracted frames.
            capture_every_sec: Interval in seconds at which to capture frames.
            cv_processor: An optional instance of a CV processor class (e.g., LocalCVProcessor).
                          If provided, its `process_image` method will be called after saving a frame.
        """
        self.save_dir = save_dir
        self.capture_every_sec = capture_every_sec
        self.cv_processor = cv_processor  # Store the CV processor instance
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
        It also implements an exponential backoff strategy for retrying stream reads.

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

        # Exponential backoff parameters
        initial_retry_delay_sec = 1.0
        max_retry_delay_sec = 60.0
        current_retry_delay_sec = initial_retry_delay_sec

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
                    print(f"Stream read failed. Retrying in {current_retry_delay_sec:.1f}s…")
                    self.last_saved_frame_hash = None  # Reset hash on stream error
                    time.sleep(current_retry_delay_sec)
                    # Increase delay for next time, up to max
                    current_retry_delay_sec = min(current_retry_delay_sec * 2, max_retry_delay_sec)
                    # Re-open the capture object as it might be in an unrecoverable state
                    cap.release()
                    cap = cv2.VideoCapture(stream_url)
                    if not cap.isOpened():
                        print(f"Failed to re-open stream after {current_retry_delay_sec:.1f}s of retrying. Waiting before next attempt...")
                        time.sleep(current_retry_delay_sec) # Wait again before trying to re-initialize
                        current_retry_delay_sec = min(current_retry_delay_sec * 2, max_retry_delay_sec)
                    else:
                        print("Successfully re-opened stream.")
                    continue
                
                # If frame read was successful, reset retry delay
                current_retry_delay_sec = initial_retry_delay_sec

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

                    # If a CV processor is provided, process the newly saved image
                    if self.cv_processor and self.cv_processor.model_trained_or_loaded:
                        try:
                            print(f"  Processing with CV: {file_path}")
                            cv_results = self.cv_processor.process_image(file_path)
                            print(f"  CV Results: {cv_results}")
                        except Exception as e:
                            print(f"  Error during CV processing for {file_path}: {e}")
                    elif self.cv_processor and not self.cv_processor.model_trained_or_loaded:
                        print(f"  CV processor available but model not ready. Skipping processing for {file_path}")

        except KeyboardInterrupt:
            print(f"\nStopped – {frame_count} images saved.")
        finally:
            sys.stderr = original_stderr  # Ensure stderr is restored
            cap.release()
            print("Video stream capture released.")
