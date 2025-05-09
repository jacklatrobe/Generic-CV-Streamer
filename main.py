"""
Main application script for YouTube Stream Frame Capturer.
Orchestrates downloading, frame extraction, and (eventually) CV processing.
"""

import os
import argparse # For command-line arguments

# Modules to be lazy-loaded will be imported inside main()

# Constants can remain global if they don't trigger heavy imports
YOUTUBE_URL   = "https://www.youtube.com/watch?v=BsfcSCJZRFM"
SAVE_DIR      = "boat_ramp_frames"
CAPTURE_EVERY = 2.0

# -----------------------------------------------------------------------------
def main(args):
    # Suppress TensorFlow INFO and WARNING messages
    # These are placed here so they only run when main() is executed, not on -h
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Import heavy modules here (lazy loading)
    import numpy as np # Not strictly needed in main directly, but good practice if other main-level logic used it
    import cv2 # Not strictly needed in main directly
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR') # Suppress TensorFlow Python-level warnings

    from downloading.downloader import YouTubeDownloader
    from image_wrangling.frame_extractor import FrameExtractor
    # Conditionally import AutoKerasCVProcessor only if not --no-cv
    if not args.no_cv:
        from computer_vision import AutoKerasCVProcessor

    # Initialize the components
    downloader = YouTubeDownloader()
    
    cv_processor_instance = None
    if not args.no_cv:
        print("Initializing Computer Vision processor...")
        # Updated instantiation and attribute check
        cv_processor_instance = AutoKerasCVProcessor(force_train=args.force_train_cv) 
        if not cv_processor_instance.model_ready_for_inference:
            print("CV Model is not ready. Frame processing will not occur if model training failed or data was unavailable.")
    else:
        print("Computer Vision processing is disabled by command-line argument.")

    # Pass configuration to FrameExtractor. SAVE_DIR and CAPTURE_EVERY are used here.
    # Also pass the cv_processor_instance (it will be None if --no-cv is used)
    frame_extractor = FrameExtractor(
        save_dir=SAVE_DIR, 
        capture_every_sec=CAPTURE_EVERY,
        cv_processor=cv_processor_instance # Pass the CV processor here
    )
    
    print(f"Attempting to capture frames from: {YOUTUBE_URL}")
    print(f"Frames will be saved to: {os.path.abspath(SAVE_DIR)}")
    print(f"Capturing one frame every {CAPTURE_EVERY} seconds.")

    try:
        stream_url = downloader.get_stream_url(YOUTUBE_URL)
        print("Successfully obtained stream URL.")
        
        # Start capturing frames from the obtained stream URL
        frame_extractor.capture_frames_from_stream(stream_url)
        
        # CV processing is now handled by FrameExtractor if cv_processor_instance is provided

    except RuntimeError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Application finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Stream Frame Capturer and CV Processor.")
    parser.add_argument("--no-cv", action="store_true", 
                        help="Disable Computer Vision processing. Only captures and saves frames.")
    parser.add_argument("--force-train-cv", action="store_true",
                        help="Force retraining of the CV model even if a model file exists.")
    
    # Potentially add other arguments here later, e.g., for YOUTUBE_URL, SAVE_DIR, etc.
    
    parsed_args = parser.parse_args()
    
    main(parsed_args) # Call main with parsed arguments
