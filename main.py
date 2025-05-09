"""
Main application script for YouTube Stream Frame Capturer.
Orchestrates downloading, frame extraction, and (eventually) CV processing.
"""

import os
# Removed unused imports like cv2, time, datetime, sys, io, yt_dlp as they are now in their respective classes

from downloading.downloader import YouTubeDownloader
from image_wrangling.frame_extractor import FrameExtractor
# from computer_vision.cv_processor import LocalCVProcessor # Import when ready to use

YOUTUBE_URL   = "https://www.youtube.com/watch?v=BsfcSCJZRFM"
SAVE_DIR      = "boat_ramp_frames"  # This can be passed to FrameExtractor
CAPTURE_EVERY = 2.0             # This can be passed to FrameExtractor

# Removed get_stream_url function (now in YouTubeDownloader)
# Removed capture_frames function (now in FrameExtractor)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the components
    downloader = YouTubeDownloader()
    # Pass configuration to FrameExtractor. SAVE_DIR and CAPTURE_EVERY are used here.
    frame_extractor = FrameExtractor(save_dir=SAVE_DIR, capture_every_sec=CAPTURE_EVERY)
    # local_cv_processor = LocalCVProcessor() # Initialize when ready

    print(f"Attempting to capture frames from: {YOUTUBE_URL}")
    print(f"Frames will be saved to: {os.path.abspath(SAVE_DIR)}")
    print(f"Capturing one frame every {CAPTURE_EVERY} seconds.")

    try:
        stream_url = downloader.get_stream_url(YOUTUBE_URL)
        print(f"Successfully obtained stream URL.")
        
        # Start capturing frames from the obtained stream URL
        frame_extractor.capture_frames_from_stream(stream_url)
        
        # Future step: Once frames are saved, or if processing live frames:
        # for frame_path in collected_frame_paths: # Or for frame_data in live_frames:
        #     results = local_cv_processor.process_image(frame_path) # or local_cv_processor.process_frame(frame_data)
        #     print(f"CV Results for {frame_path}: {results}")

    except RuntimeError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Application finished.")
