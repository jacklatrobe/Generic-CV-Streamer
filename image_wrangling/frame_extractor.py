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
    def __init__(self, save_dir: str = "raw_frames", capture_every_sec: float = 2.0, cv_processor=None, detections_save_dir: str = "cv_detections"):
        """Initializes the FrameExtractor.

        Args:
            save_dir: Directory to save the extracted raw frames.
            capture_every_sec: Interval in seconds at which to capture frames.
            cv_processor: An optional instance of a CV processor class.
            detections_save_dir: Directory to save processed detections (e.g., cropped objects).
        """
        self.save_dir = save_dir
        self.capture_every_sec = capture_every_sec
        self.cv_processor = cv_processor
        self.detections_save_dir = detections_save_dir # Store detections save directory
        self.last_saved_frame_hash = None
        os.makedirs(self.save_dir, exist_ok=True)
        # Ensure the base detections directory exists if a cv_processor is provided
        # The CV processor itself will handle its specific subdirectories.
        if self.cv_processor and self.detections_save_dir:
            os.makedirs(self.detections_save_dir, exist_ok=True)

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
                    if self.cv_processor:
                        is_ready = False
                        backend_name = "Unknown CV"
                        # Check for AutoKeras
                        if hasattr(self.cv_processor, 'model_ready_for_inference'): 
                            is_ready = self.cv_processor.model_ready_for_inference
                            backend_name = "Local (AutoKeras)"
                        # Check for GoogleCVProcessor or other cloud processors that might use 'api_ready_for_inference'
                        elif hasattr(self.cv_processor, 'api_ready_for_inference'): 
                            is_ready = self.cv_processor.api_ready_for_inference
                            # Attempt to get a more specific name if available (e.g. from GoogleCVProcessor)
                            if hasattr(self.cv_processor, 'backend_name'):
                                backend_name = self.cv_processor.backend_name
                            else:
                                backend_name = "Google CV" # Default if more specific name not found
                        # Check for AzureCVInferencer or similar using 'api_ready'
                        elif hasattr(self.cv_processor, 'api_ready'): 
                            is_ready = self.cv_processor.api_ready
                            # Try to get a more specific name, e.g. by checking class name
                            if self.cv_processor.__class__.__name__ == "AzureCVInferencer":
                                backend_name = "Azure CV"
                            elif hasattr(self.cv_processor, 'backend_name'): # Fallback to a backend_name attribute
                                backend_name = self.cv_processor.backend_name
                            # else it remains "Unknown CV" if it's a new type with just 'api_ready'

                        if is_ready:
                            try:
                                # print(f"  Processing with {backend_name}: {file_path}") # Already logged by CV module usually
                                # Pass the original frame (image_np) to process_frame for CV modules that need it for cropping
                                cv_results = self.cv_processor.process_frame(frame) # Pass frame for potential cropping
                                
                                # Check if the cv_processor has a save_processed_frame method
                                if hasattr(self.cv_processor, 'save_processed_frame') and callable(getattr(self.cv_processor, 'save_processed_frame')):
                                    # AutoKeras saves the whole frame, others save cropped objects from the 'objects' list
                                    if backend_name == "Local (AutoKeras)":
                                        # AutoKeras save_processed_frame expects the result dict and frame_data
                                        if cv_results and cv_results.get("tags") and cv_results["tags"][0] not in ["no_confident_match", "error_model_not_ready", "error_processing_frame"] and cv_results.get("confidence", 0.0) >= self.cv_processor.confidence_threshold:
                                            self.cv_processor.save_processed_frame(frame, cv_results, frame_path=file_path)
                                    elif cv_results and cv_results.get("objects"):
                                        # For object detectors (Azure, Google), pass the frame and the list of objects
                                        # The save_processed_frame method in these inferencers will handle cropping and saving
                                        self.cv_processor.save_processed_frame(frame, cv_results.get("objects"))
                                
                                # Logging results based on the structure of cv_results
                                # This part is for console logging of the main tags/confidence summary
                                confidence_threshold_to_use = 0.7 # Default
                                if hasattr(self.cv_processor, 'confidence_threshold'): # General case for inferencers
                                    confidence_threshold_to_use = self.cv_processor.confidence_threshold
                                elif hasattr(self.cv_processor, 'inferencer') and self.cv_processor.inferencer and \
                                   hasattr(self.cv_processor.inferencer, 'confidence_threshold'): # For AutoKerasCVProcessor
                                    confidence_threshold_to_use = self.cv_processor.inferencer.confidence_threshold
                                
                                if cv_results and cv_results.get("tags") and cv_results["tags"]:
                                    first_tag = cv_results["tags"][0]
                                    # Consolidate negative/error tags if possible, or ensure they are consistently named
                                    negative_tags = ["no_confident_match", "no_confident_match_google", "no_labels_from_google", "no_confident_match_azure", "no_labels_from_azure"]
                                    is_error_tag = first_tag.startswith("error_")

                                    if not is_error_tag and first_tag not in negative_tags and \
                                       cv_results.get("confidence", 0.0) >= confidence_threshold_to_use:
                                        print(f"  {backend_name} Results: Tags: {cv_results['tags']}, Confidence: {cv_results['confidence']:.4f}")
                                        # Detailed objects are saved by save_processed_frame, log count here if useful
                                        if cv_results.get("objects") is not None:
                                            print(f"  {backend_name}: Found {len(cv_results.get('objects'))} objects. Confident detections are saved.")
                                    else:
                                        actual_confidence = cv_results.get('confidence', 0.0)
                                        if first_tag in negative_tags:
                                            print(f"  {backend_name}: No target class met threshold (Tag: {first_tag}, Confidence: {actual_confidence:.4f})")
                                        elif is_error_tag:
                                            print(f"  {backend_name}: Error processing (Tag: {first_tag}, Confidence: {actual_confidence:.4f})")
                                        else: # Low confidence for a target class
                                            print(f"  {backend_name}: Low confidence for match (Tag: {first_tag}, Confidence: {actual_confidence:.4f})")
                                else:
                                    print(f"  {backend_name} processing returned no tags or unexpected result format: {cv_results}")
                                    
                            except Exception as e:
                                print(f"  Error during {backend_name} processing for {file_path}: {e}")
                        else:
                            print(f"  CV processor ({backend_name}) available but not ready. Skipping processing for {file_path}")

        except KeyboardInterrupt:
            print(f"\nStopped – {frame_count} images saved.")
        finally:
            sys.stderr = original_stderr  # Ensure stderr is restored
            cap.release()
            print("Video stream capture released.")
