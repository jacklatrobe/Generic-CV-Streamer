"""
Main application script for Stream Frame Capturer.
Orchestrates downloading, frame extraction, and (eventually) CV processing.
"""

import os
import argparse # For command-line arguments
import json # Import json for the exception handler
from config_manager import ConfigManager # Import the new ConfigManager

# Constants previously defined globally will now be loaded from config.json
# SAVE_DIR      = "boat_ramp_frames"
# CAPTURE_EVERY = 2.0

# -----------------------------------------------------------------------------
def main(args):
    # Load configuration using ConfigManager
    try:
        config_manager = ConfigManager(config_path=args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except json.JSONDecodeError as e: # ConfigManager might re-raise this
        print(f"Error: {e}")
        return
    except IOError as e: # ConfigManager might raise this for other read errors
        print(f"Error: {e}")
        return
    except Exception as e: # Catch any other init errors from ConfigManager
        print(f"An unexpected error occurred while loading configuration: {e}")
        return

    # Get settings from ConfigManager
    SAVE_DIR = config_manager.get_save_dir()
    CAPTURE_EVERY = config_manager.get_capture_every_sec()
    default_credentials_path = config_manager.get_credentials_path(default="service_account.json")
    # Get AutoKeras specific settings for later use
    try:
        class_names = config_manager.get_class_names()
    except (KeyError, ValueError) as e:
        print(f"Configuration Error: {e}")
        return
    
    autokeras_model_dir = config_manager.get_autokeras_model_dir()
    autokeras_model_filename = config_manager.get_autokeras_model_filename()
    # Potentially get image_data_dir for trainer if needed here or within AutoKerasCVProcessor
    autokeras_image_data_dir = config_manager.get_autokeras_image_data_dir() # Assuming you add this to ConfigManager

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
            # Import needs to be here for GoogleCVProcessor
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
            # Pass the retrain_cv flag and config values to the AutoKeras processor
            cv_processor_instance = AutoKerasCVProcessor(
                retrain_model=args.retrain_cv,
                class_names=class_names,
                model_dir=autokeras_model_dir,
                model_filename=autokeras_model_filename,
                image_data_dir=autokeras_image_data_dir # Pass image_data_dir
            )
            if not cv_processor_instance.model_ready_for_inference:
                print("Local CV Model (AutoKeras) is not ready. Frame processing will not occur if model training failed or data was unavailable.")
        elif args.cv_backend == "google":
            # Determine credentials path: command line > config file > hardcoded default (now from config)
            credentials_path = args.google_credentials if args.google_credentials is not None else default_credentials_path
            cv_processor_instance = GoogleCVProcessor(credentials_path=credentials_path) 
            if not cv_processor_instance.api_ready_for_inference: # Check the correct attribute
                print(f"Google Cloud Vision API is not ready (credentials: {credentials_path}). Frame processing will not occur. Check credentials and API status.")
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
    parser.add_argument("--config", type=str, default="config.json", # Default path for ConfigManager
                        help="Path to the configuration JSON file (default: config.json in project root).")
    parser.add_argument("--no-cv", action="store_true", 
                        help="Disable Computer Vision processing. Only captures and saves frames.")
    parser.add_argument("--retrain-cv", action="store_true",
                        help="Retrain the CV model. If not set, uses an existing model or errors if none exists.")
    parser.add_argument("--cv-backend", type=str, default="local", choices=["local", "google"],
                        help="Specify the computer vision backend to use: 'local' for AutoKeras or 'google' for Google Cloud Vision API.")
    parser.add_argument("--credentials-path", type=str, default=None, # Default is now handled by config or a fallback in main()
                        help="Path to Cloud service account JSON file (e.g., Google, Azure). Overrides path in config file. If not provided, uses path from config or a default.")
    
    # Potentially add other arguments here later, e.g., for YOUTUBE_URL, SAVE_DIR, etc.
    
    parsed_args = parser.parse_args()
    
    main(parsed_args) # Call main with parsed arguments
