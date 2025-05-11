"""
Main application script for Stream Frame Capturer.
Orchestrates downloading, frame extraction, and (eventually) CV processing.
"""

import os
import argparse # For command-line arguments

# Modules to be lazy-loaded will be imported inside main()

# Constants can remain global if they don't trigger heavy imports
SAVE_DIR      = "boat_ramp_frames"
CAPTURE_EVERY = 2.0

# -----------------------------------------------------------------------------
def main(args):
    # Suppress TensorFlow INFO and WARNING messages
    # These are placed here so they only run when main() is executed, not on -h
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Import heavy modules here (lazy loading)
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR') # Suppress TensorFlow Python-level warnings

    from downloading.downloader import YouTubeDownloader # Keep for YouTube
    from downloading.earthcam_downloader import EarthCamDownloader # Add for EarthCam
    from image_wrangling.frame_extractor import FrameExtractor

    # Conditionally import CV processors only if not --no-cv
    if not args.no_cv:
        if args.cv_backend == "local":
            from computer_vision import AutoKerasCVProcessor
        elif args.cv_backend == "google":
            from computer_vision import GoogleCVProcessor
        # Add other backends here if needed
        else: # Should not happen due to argparse choices
            raise ValueError(f"Invalid CV backend specified: {args.cv_backend}")

    # Initialize the components
    youtube_downloader = YouTubeDownloader()
    earthcam_downloader = EarthCamDownloader()
    
    # Determine the source URL
    source_url = args.url if args.url else None
    if not source_url:
        raise ValueError("No URL provided. Please specify a YouTube or EarthCam URL using --url.")
    is_youtube_url = "youtube.com" in source_url or "youtu.be" in source_url
    is_earthcam_url = "earthcam.com" in source_url

    cv_processor_instance = None
    if not args.no_cv:
        print(f"Initializing Computer Vision processor with backend: {args.cv_backend}...")
        if args.cv_backend == "local":
            # Pass the retrain_cv flag to the AutoKeras processor
            cv_processor_instance = AutoKerasCVProcessor(retrain_model=args.retrain_cv)
            if not cv_processor_instance.model_ready_for_inference:
                print("Local CV Model (AutoKeras) is not ready. Frame processing will not occur if model training failed or data was unavailable.")
        elif args.cv_backend == "google":
            # Pass credentials path if provided via command line
            credentials_path = args.google_credentials if hasattr(args, 'google_credentials') else None
            cv_processor_instance = GoogleCVProcessor(credentials_path=credentials_path) 
            if not cv_processor_instance.api_ready_for_inference: # Check the correct attribute
                print("Google Cloud Vision API is not ready. Frame processing will not occur. Check credentials and API status.")
        # Add other backend initializations here
    else:
        print("Computer Vision processing is disabled by command-line argument.")

    # Pass configuration to FrameExtractor. SAVE_DIR and CAPTURE_EVERY are used here.
    # Also pass the cv_processor_instance (it will be None if --no-cv is used)
    frame_extractor = FrameExtractor(
        save_dir=SAVE_DIR, 
        capture_every_sec=CAPTURE_EVERY,
        cv_processor=cv_processor_instance # Pass the CV processor here
    )
    
    print(f"Attempting to capture frames from: {source_url}")
    print(f"Frames will be saved to: {os.path.abspath(SAVE_DIR)}")
    print(f"Capturing one frame every {CAPTURE_EVERY} seconds.")

    try:
        stream_url_to_capture = None
        if is_youtube_url:
            print("YouTube URL detected, getting stream URL via yt-dlp...")
            stream_url_to_capture = youtube_downloader.get_stream_url(source_url)
            print("Successfully obtained stream URL for YouTube video.")
        elif is_earthcam_url:
            print("EarthCam URL detected, attempting to extract HLS stream...")
            stream_url_to_capture = earthcam_downloader.get_stream_url(source_url)
            print("Successfully obtained HLS stream URL for EarthCam.")
        else:
            print("Non-YouTube/EarthCam URL detected, attempting to use directly.")
            stream_url_to_capture = source_url

        if not stream_url_to_capture:
            raise RuntimeError("Failed to obtain a valid stream URL to capture.")

        # Start capturing frames from the obtained stream URL
        frame_extractor.capture_frames_from_stream(stream_url_to_capture)
        
        # CV processing is now handled by FrameExtractor if cv_processor_instance is provided

    except RuntimeError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Application finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Stream Frame Capturer and CV Processor.")
    parser.add_argument("--url", type=str, default=None, 
                        help="URL of the video stream.")
    parser.add_argument("--no-cv", action="store_true", 
                        help="Disable Computer Vision processing. Only captures and saves frames.")
    parser.add_argument("--retrain-cv", action="store_true",
                        help="Retrain the CV model. If not set, uses an existing model or errors if none exists.")
    parser.add_argument("--cv-backend", type=str, default="local", choices=["local", "google"],
                        help="Specify the computer vision backend to use: 'local' for AutoKeras or 'google' for Google Cloud Vision API.")
    parser.add_argument("--google-credentials", type=str, default=None,
                        help="Path to Google Cloud service account JSON file. If not provided, will try to use gptactions-424000-8f24fedcc786.json in the project root.")
    
    # Potentially add other arguments here later, e.g., for YOUTUBE_URL, SAVE_DIR, etc.
    
    parsed_args = parser.parse_args()
    
    main(parsed_args) # Call main with parsed arguments
