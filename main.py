"""
Main application script for Stream Frame Capturer.
Orchestrates downloading, frame extraction, and (eventually) CV processing.
"""

import os
import argparse # For command-line arguments
import json # Import json for the exception handler
import logging # Added for logging
from datetime import datetime # Added for log filename timestamp
from config_manager import ConfigManager # Import the new ConfigManager

# -----------------------------------------------------------------------------
# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime(f"app_log_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(log_dir, log_filename)

# --- Logging Configuration --- Start
# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Set root logger level

# Remove any existing handlers to avoid duplicates if this script is re-run in some contexts
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# File Handler - logs everything from INFO level for all loggers
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')
file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO) # Ensure file handler captures INFO and above
root_logger.addHandler(file_handler)

# Console Handler - logs application INFO messages, but only WARNING and above for Azure SDK
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s') # Simpler format for console
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)

# Filter for console handler to suppress Azure SDK INFO logs
class AzureSdkInfoFilter(logging.Filter):
    def filter(self, record):
        # Allow logs not from Azure HTTP policy or computer_vision.azure_cv_inferencer,
        # or allow if level is WARNING or higher
        if record.name == "azure.core.pipeline.policies.http_logging_policy" and record.levelno < logging.WARNING:
            return False
        if record.name == "computer_vision.azure_cv_inferencer" and record.levelno < logging.WARNING: # Added this condition
            return False
        return True

console_handler.addFilter(AzureSdkInfoFilter())

# NEW: Filter for console handler to make network/OpenCV Python exceptions friendlier
class ConsoleFriendlyErrorFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.ERROR:
            # Scenario 1: Specifically for logger.exception("An unexpected error occurred...")
            if "An unexpected error occurred" in record.msg and record.exc_info:
                 record.msg = "An unexpected critical error occurred. Please check the log file for details and stack trace."
                 record.exc_info = None
                 record.exc_text = None
                 record.args = () # msg is now a literal string
                 return True # Indicates this record was handled regarding console output modification

            # Scenario 2: For RuntimeErrors or cv2.errors related to stream/network/OpenCV issues
            if record.exc_info:
                exc_type, exc_value, _ = record.exc_info
                error_message_str = str(exc_value).lower() # For case-insensitive keyword matching

                is_targeted_for_friendly_console_msg = False
                # Check for RuntimeErrors with specific keywords
                if isinstance(exc_value, RuntimeError):
                    stream_error_keywords = [
                        "could not open stream",
                        "failed to obtain a valid stream url",
                        "stream timeout", # General for timeout messages
                        "unable to read from socket",
                        "failed to send close message",
                        "error when loading first segment", # For HLS/stream parsing errors
                    ]
                    if any(keyword in error_message_str for keyword in stream_error_keywords):
                        is_targeted_for_friendly_console_msg = True
                
                # Check if it's a cv2.error (Python exception from OpenCV)
                # These are often verbose and can be simplified on the console.
                # Ensure cv2 module name is in the type's module path.
                if exc_type.__name__ == 'error' and hasattr(exc_type, '__module__') and 'cv2' in exc_type.__module__:
                    is_targeted_for_friendly_console_msg = True

                if is_targeted_for_friendly_console_msg:
                    record.msg = "A network or video processing error occurred. Please check the log file for details."
                    record.exc_info = None # Suppress stack trace on console
                    record.exc_text = None # Suppress formatted exception text on console
                    record.args = () # msg is now a literal string, so no args needed
                    return True # Indicates this record was handled

        return True # Allow all other records to pass through to the console normally

console_handler.addFilter(ConsoleFriendlyErrorFilter())
console_handler.setLevel(logging.INFO) # Console handler itself processes INFO and above, filters do the specific suppression
root_logger.addHandler(console_handler)

# Set Azure SDK logger level to INFO to ensure its messages *can* reach handlers,
# but the console handler's filter will prevent INFO from appearing on console.
# The file handler will still log them.
azure_http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
azure_http_logger.setLevel(logging.INFO) 
# Prevent Azure SDK logs from propagating to the root logger's console handler if they are INFO
# This is an alternative to filtering at the handler if we want to be more direct with the Azure logger
# azure_http_logger.propagate = False # If used, would need a dedicated handler for azure_http_logger to log to file

logger = logging.getLogger(__name__) # Main application logger for this module
logger.info("Logging initialized. Azure SDK HTTP INFO logs will be in the file but not on console.")
# --- Logging Configuration --- End
# -----------------------------------------------------------------------------
def main(args):
    logger.info("Application starting.") # Example of using logger
    # Load configuration using ConfigManager
    try:
        config_path_to_use = args.config # Use the one from args first
        if not os.path.exists(config_path_to_use):
            logger.warning(f"Config file specified via --config '{args.config}' not found. Attempting to use default 'config.json' in current directory: {os.path.join(os.getcwd(), 'config.json')}")
            config_path_to_use = "config.json" # Try default name in CWD
            if not os.path.exists(config_path_to_use): # Check again for the default in CWD
                 logger.error(f"Default config file 'config.json' also not found in current working directory: {os.getcwd()}")
                 raise FileNotFoundError(f"Neither specified config file '{args.config}' nor default 'config.json' found.")
        
        logger.info(f"Loading configuration from: {os.path.abspath(config_path_to_use)}")
        config_manager = ConfigManager(config_path=config_path_to_use)
    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
        return
    except json.JSONDecodeError as e: # ConfigManager might re-raise this
        # print(f"Error: {e}")
        logger.error(f"Error decoding configuration JSON: {e}")
        return
    except IOError as e: # ConfigManager might raise this for other read errors
        # print(f"Error: {e}")
        logger.error(f"IOError reading configuration file: {e}")
        return
    except Exception as e: # Catch any other init errors from ConfigManager
        # print(f"An unexpected error occurred while loading configuration: {e}")
        logger.exception("An unexpected error occurred while loading configuration.") # Logs stack trace
        return

    # Get settings from ConfigManager
    RAW_SAVE_DIR = config_manager.get_raw_save_dir() # Updated from get_save_dir()
    DETECTIONS_SAVE_DIR = config_manager.get_detections_save_dir() # Get the new setting
    CAPTURE_EVERY = config_manager.get_capture_every_sec()
    default_credentials_path = config_manager.get_credentials_path(default="service_account.json")
    confidence_threshold = config_manager.get_confidence_threshold() # Get the confidence threshold
    logger.info(f"Raw frames save directory set to: {RAW_SAVE_DIR}")
    logger.info(f"Detections save directory set to: {DETECTIONS_SAVE_DIR}")
    logger.info(f"Capture interval set to: {CAPTURE_EVERY} seconds")

    # Get AutoKeras specific settings for later use
    try:
        original_class_names = config_manager.get_class_names() 
        logger.info(f"Original class names loaded from config: {original_class_names}")
        expansion_map = config_manager.get_expansion_map() # Load expansion map
        if not expansion_map: # Add a default empty map if not in config, to prevent errors
            logger.info("'expansion_map' not found in config or is empty. Using no expansions.")
            expansion_map = {}
        else:
            logger.info(f"Expansion map loaded from config: {expansion_map}")

    except (KeyError, ValueError) as e:
        logger.error(f"Configuration Error for class_names or expansion_map: {e}")
        return

    # --- Quick and dirty class name expansion --- Start
    # expansion_map is now loaded from config
    expanded_class_names = []
    if original_class_names: # Ensure original_class_names is not None and is iterable
        for name in original_class_names:
            if isinstance(name, str):
                # Get the list of expanded names, defaulting to the original name if not in map
                # Ensure all names are lowercased for consistent matching
                terms_to_add = [term.lower() for term in expansion_map.get(name.lower(), [name])]
                expanded_class_names.extend(terms_to_add)
            else:
                logger.warning(f"Encountered non-string class name '{name}' in config, skipping expansion for it.")
                expanded_class_names.append(str(name).lower()) # Add as string, lowercased

    # Remove duplicates and sort for consistency (optional, but good for logging)
    if expanded_class_names:
        expanded_class_names = sorted(list(set(expanded_class_names)))
    logger.info(f"Expanded class names for CV processing: {expanded_class_names}")
    # --- Quick and dirty class name expansion --- End
    
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

    # Attempt to suppress OpenCV native warnings (e.g., [ WARN:0@...])
    try:
        import cv2 # Import cv2 earlier specifically for log level setting
        if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
            # cv2.utils.logging.LOG_LEVEL_SILENT or cv2.utils.logging.LOG_LEVEL_ERROR
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR) 
            logger.info("Attempted to set OpenCV native log level to ERROR to reduce console noise.")
        else:
            logger.info("cv2.utils.logging not available with this OpenCV version. Native OpenCV warnings might still appear on console.")
    except ImportError:
        # This might happen if cv2 is not installed, though it's a dependency for FrameExtractor
        logger.warning("OpenCV (cv2) could not be imported directly in main.py for log level setting. FrameExtractor will handle its import.")
    except Exception as e_cv_log:
        logger.warning(f"Could not set OpenCV native log level in main.py: {e_cv_log}")

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
        elif args.cv_backend == "azure": # Add this block
            from computer_vision import AzureCVInferencer
        else: # Should not happen due to argparse choices
            raise ValueError(f"Invalid CV backend specified: {args.cv_backend}")

    # Initialize the components
    youtube_downloader = YouTubeDownloader()
    earthcam_downloader = EarthCamDownloader()
    
    # Determine the source URL
    source_url = args.url if args.url else None
    if not source_url:
        raise ValueError("No URL provided. Please specify a YouTube or EarthCam URL using --url.")
    from urllib.parse import urlparse
    parsed_url = urlparse(source_url)
    is_earthcam_url = parsed_url.hostname and "earthcam.com" in parsed_url.hostname
    is_youtube_url = ( parsed_url.hostname and parsed_url.hostname.endswith("youtube.com") ) or ( parsed_url.hostname and parsed_url.hostname.endswith("youtu.be") )
    cv_processor_instance = None
    if not args.no_cv:
        logger.info(f"Initializing Computer Vision processor with backend: {args.cv_backend}...")
        if args.cv_backend == "local":
            # Pass the retrain_cv flag and config values to the AutoKeras processor
            cv_processor_instance = AutoKerasCVProcessor(
                retrain_model=args.retrain_cv,
                class_names=expanded_class_names, # Use expanded_class_names
                model_dir=autokeras_model_dir,
                model_filename=autokeras_model_filename,
                image_data_dir=autokeras_image_data_dir, # Pass image_data_dir
                confidence_threshold=confidence_threshold # Pass confidence_threshold
            )
            if not cv_processor_instance.model_ready_for_inference:
                logger.warning("Local CV Model (AutoKeras) is not ready. Frame processing will not occur if model training failed or data was unavailable.")
        elif args.cv_backend == "google":
            # Determine credentials path: command line > config file > hardcoded default (now from config)
            credentials_path = args.credentials_path if args.credentials_path is not None else default_credentials_path
            logger.info(f"Using Google CV credentials from: {credentials_path}")
            cv_processor_instance = GoogleCVProcessor(credentials_path=credentials_path, confidence_threshold=confidence_threshold) # Pass confidence_threshold
            if not cv_processor_instance.api_ready_for_inference: # Check the correct attribute
                logger.warning(f"Google Cloud Vision API is not ready (credentials: {credentials_path}). Frame processing will not occur. Check credentials and API status.")
        elif args.cv_backend == "azure": # Add this block
            credentials_path = args.credentials_path if args.credentials_path is not None else config_manager.get_setting("azure_credentials_path", "azure_cv_credentials.json") # Or some other default
            logger.info(f"Using Azure CV credentials from: {credentials_path}")
            cv_processor_instance = AzureCVInferencer(
                credentials_path=credentials_path, 
                confidence_threshold=confidence_threshold,
                class_names_override=expanded_class_names # Pass expanded list
            )
            if not cv_processor_instance.api_ready:
                logger.warning(f"Azure Computer Vision API is not ready (credentials: {credentials_path}). Frame processing will not occur. Check credentials and API status.")
        # Add other backend initializations here
    else:
        logger.info("Computer Vision processing is disabled by command-line argument.")

    # Pass configuration to FrameExtractor. SAVE_DIR and CAPTURE_EVERY are used here.
    # Also pass the cv_processor_instance (it will be None if --no-cv is used)
    frame_extractor = FrameExtractor(
        save_dir=RAW_SAVE_DIR,  # Updated to RAW_SAVE_DIR
        capture_every_sec=CAPTURE_EVERY,
        cv_processor=cv_processor_instance, # Pass the CV processor here
        detections_save_dir=DETECTIONS_SAVE_DIR # Pass the detections save directory
    )
    
    logger.info(f"Attempting to capture frames from: {source_url}")
    # print(f"Frames will be saved to: {os.path.abspath(SAVE_DIR)}") # Covered by logger
    # print(f"Capturing one frame every {CAPTURE_EVERY} seconds.") # Covered by logger

    try:
        stream_url_to_capture = None
        if is_youtube_url:
            logger.info("YouTube URL detected, getting stream URL via yt-dlp...")
            stream_url_to_capture = youtube_downloader.get_stream_url(source_url)
            logger.info("Successfully obtained stream URL for YouTube video.")
        elif is_earthcam_url:
            logger.info("EarthCam URL detected, attempting to extract HLS stream...")
            stream_url_to_capture = earthcam_downloader.get_stream_url(source_url)
            logger.info("Successfully obtained HLS stream URL for EarthCam.")
        else:
            logger.info("Non-YouTube/EarthCam URL detected, attempting to use directly.")
            stream_url_to_capture = source_url

        if not stream_url_to_capture:
            logger.error("Failed to obtain a valid stream URL to capture.")
            raise RuntimeError("Failed to obtain a valid stream URL to capture.")

        # Start capturing frames from the obtained stream URL
        frame_extractor.capture_frames_from_stream(stream_url_to_capture)
        
        # CV processing is now handled by FrameExtractor if cv_processor_instance is provided

    except RuntimeError as e:
        # print(f"An error occurred: {e}")
        logger.error(f"A runtime error occurred: {e}")
    except Exception as e:
        # print(f"An unexpected error occurred: {e}")
        logger.exception("An unexpected error occurred during main execution.") # Logs stack trace

    # print("Application finished.")
    logger.info("Application finished.")

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
    parser.add_argument("--cv-backend", type=str, default="local", choices=["local", "google", "azure"],
                        help="Specify the computer vision backend to use: \'local\' for AutoKeras, \'google\' for Google Cloud Vision API, or \'azure\' for Microsoft Azure Vision API.")
    parser.add_argument("--credentials-path", type=str, default=None, 
                        help="Path to Cloud service account JSON file (e.g., Google Cloud, Azure). Overrides path in config file. If not provided, uses path from config or a default (e.g., service_account.json for Google, azure_cv_credentials.json for Azure).")
    
    # Potentially add other arguments here later, e.g., for YOUTUBE_URL, SAVE_DIR, etc.
    
    parsed_args = parser.parse_args()
    
    main(parsed_args) # Call main with parsed arguments
