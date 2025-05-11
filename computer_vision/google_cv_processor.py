# filepath: c:\Users\Jack\Generic-CV-Streamer\computer_vision\google_cv_processor.py
"""
Orchestrates the Google Cloud Vision API-based computer vision processing.
"""
import os
import numpy as np
from .google_cv_inferencer import GoogleCVInferencer

DATA_DIR_RELATIVE = "../data" 

class GoogleCVProcessor:
    """
    High-level interface for using Google Cloud Vision API for image classification.
    Manages the initialization of the GoogleCVInferencer.
    """

    def __init__(self, config_manager, credentials_path=None):
        """
        Initializes the GoogleCVProcessor.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager to access settings.
            credentials_path (str, optional): Path to Google Cloud service account JSON file.
                                          Overrides credentials_path from config if provided.
        """
        self.config_manager = config_manager
        self.class_names = self.config_manager.get_class_names(default=[])
        self.detections_save_dir = self.config_manager.get_detections_save_dir(default="detections/google_cv")
        self.confidence_threshold = self.config_manager.get_confidence_threshold(default=0.7) # Get general confidence
        
        # Determine credentials_path: command-line arg > config file > default
        # The command-line override for credentials_path is handled in main.py before this is called.
        # Here, we prioritize the direct credentials_path argument if provided to __init__,
        # then fall back to what's in config_manager (which itself has a default).
        actual_credentials_path = credentials_path if credentials_path is not None else self.config_manager.get_setting("google_credentials_path")

        # If still no path, and a default general 'credentials_path' exists in config, consider using it.
        # However, it's better to have a specific 'google_credentials_path' for clarity.
        if actual_credentials_path is None:
            actual_credentials_path = self.config_manager.get_credentials_path(default=None) # General one
            if actual_credentials_path:
                print(f"Warning: Using general 'credentials_path' for Google CV. Consider setting 'google_credentials_path' in config.")

        # Ensure detections_save_dir is specific for Google CV if not already handled by config
        # This creates a subdirectory like "detections/google_cv"
        base_detections_dir = self.config_manager.get_detections_save_dir(default="detections")
        self.detections_save_dir = os.path.join(base_detections_dir, "google_cv")
        os.makedirs(self.detections_save_dir, exist_ok=True)

        self.inferencer = None
        self.api_ready_for_inference = False

        print("Initializing Google Cloud Vision Processor...")
        self.inferencer = GoogleCVInferencer(
            credentials_path=actual_credentials_path,
            class_names=self.class_names,
            detections_save_dir=self.detections_save_dir, # Pass the specific dir
            confidence_threshold=self.confidence_threshold
        )

        if self.inferencer and self.inferencer.api_ready: # Check inferencer's api_ready status
            self.api_ready_for_inference = True
            print("Google Cloud Vision Processor initialized and API is ready.")
        else:
            self.api_ready_for_inference = False
            print("Google Cloud Vision Processor: API is not ready. Inference will not be available.")
            print("Please ensure 'google-cloud-vision' is installed (pip install google-cloud-vision) "
                  "and your Google Cloud credentials are set up correctly (e.g., GOOGLE_APPLICATION_CREDENTIALS environment variable).")


    def _determine_class_names(self):
        """
        Determines class names by listing subdirectories in the `autokeras_image_data_dir` from config.
        Populates `self.class_names` if it wasn't already populated from `class_names` in config.
        This method is primarily a fallback or for autokeras, GoogleCV should rely on `class_names` directly.
        For GoogleCV, class_names are now directly taken from config_manager in __init__.
        This method might be redundant for GoogleCVProcessor if class_names are always defined in config.
        Kept for potential compatibility or if class_names might be dynamically determined in some scenarios.
        """
        if not self.class_names: # Only run if class_names weren't provided/found in config
            image_data_dir_for_classes = self.config_manager.get_autokeras_image_data_dir(default=None)
            if image_data_dir_for_classes and os.path.exists(image_data_dir_for_classes) and os.path.isdir(image_data_dir_for_classes):
                excluded_dirs = {'models', 'detections', 'detections_google_cv', 'google_cv'} # Added 'google_cv'
                self.class_names = sorted([
                    d for d in os.listdir(image_data_dir_for_classes)
                    if os.path.isdir(os.path.join(image_data_dir_for_classes, d)) and d not in excluded_dirs
                ])
                if not self.class_names:
                    print(f"Warning: Data directory {image_data_dir_for_classes} found, but it does not contain any "
                          "valid class subdirectories (or they are all excluded). "
                          "Google CV will report all labels it finds.")
                else:
                    print(f"Target class names determined from data directory: {self.class_names}")
            else:
                print(f"Warning: '{self.config_manager.config_path}' does not define 'class_names' and "
                      f"autokeras_image_data_dir ('{image_data_dir_for_classes}') not found or is not a directory. "
                      "Google CV will report all labels it finds if no class_names are specified.")
        # else: class_names already loaded by __init__ from config_manager


    def process_image(self, image_path: str):
        """
        Processes an image from a file path using the GoogleCVInferencer.
        """
        if self.inferencer and self.inferencer.api_ready:
            return self.inferencer.process_image(image_path)
        else:
            print("Google CV Inferencer not available or API not ready. Cannot process image.")
            return {"tags": ["error_inferencer_not_ready_google"], "confidence": 0.0}

    def process_frame(self, frame_data: np.ndarray):
        """
        Processes a raw image frame (NumPy array) using the GoogleCVInferencer.
        """
        if self.inferencer and self.inferencer.api_ready:
            return self.inferencer.process_frame(frame_data)
        else:
            print("Google CV Inferencer not available or API not ready. Cannot process frame.")
            return {"tags": ["error_inferencer_not_ready_google"], "confidence": 0.0}

    def save_processed_frame(self, frame_data: np.ndarray, result: dict, frame_path: str = None):
        """
        Saves a processed frame using the GoogleCVInferencer's save method.
        """
        if self.inferencer and self.inferencer.api_ready:
            return self.inferencer.save_processed_frame(frame_data, result, frame_path)
        else:
            print("Google CV Inferencer not available. Cannot save frame.")
            return None

# Example usage (for testing purposes when running this file directly)
if __name__ == '__main__':
    print("Initializing GoogleCVProcessor for testing...")
    # For testing, create a dummy ConfigManager
    class DummyConfigManager:
        def get_class_names(self, default=None): return ['boat', 'car', 'trailer']
        def get_detections_save_dir(self, default=None): return "detections_test"
        def get_confidence_threshold(self, default=None): return 0.5
        def get_setting(self, key, default=None):
            if key == "google_credentials_path": return None # Simulate not set, to test fallback
            return default
        def get_credentials_path(self, default=None): return "dummy_google_creds.json" # General fallback
        def get_autokeras_image_data_dir(self, default=None): return "image_data_test" # For _determine_class_names fallback
        @property
        def config_path(self): return "dummy_config.json"

    dummy_config = DummyConfigManager()
    # Create a dummy credentials file if it doesn't exist, as inferencer might try to use it.
    if not os.path.exists("dummy_google_creds.json"):
        with open("dummy_google_creds.json", 'w') as f:
            import json
            json.dump({"type": "service_account"}, f) # Minimal valid JSON

    processor = GoogleCVProcessor(config_manager=dummy_config)

    if processor.api_ready_for_inference:
        dummy_image_dir = os.path.join(os.path.dirname(__file__), "sample_frames") 
        os.makedirs(dummy_image_dir, exist_ok=True)
        dummy_image_path = os.path.join(dummy_image_dir, "google_cv_test_image.jpg")

        if not os.path.exists(dummy_image_path):
            try:
                import cv2 
                dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
                cv2.putText(dummy_img, "Test", (50,128), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 2)
                cv2.imwrite(dummy_image_path, dummy_img)
                print(f"Created dummy image for testing: {dummy_image_path}")
            except Exception as e:
                print(f"Could not create dummy image {dummy_image_path}: {e}. Please create it manually for testing.")

        if os.path.exists(dummy_image_path):
            print(f"\nTesting process_image with {dummy_image_path}:")
            result_image = processor.process_image(dummy_image_path)
            print(f"Result for image: {result_image}")
        else:
            print(f"\nSkipping process_image test, dummy image not found at {dummy_image_path}")

        print("\nTesting process_frame with a dummy frame:")
        try:
            import cv2 
            dummy_frame_data = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8) # BGR
            cv2.putText(dummy_frame_data, "Frame Test", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            result_frame = processor.process_frame(dummy_frame_data)
            print(f"Result for dummy frame: {result_frame}")

            if result_frame and result_frame.get("tags") and \
               result_frame["tags"][0] not in ["error_api_not_ready", "no_confident_match_google", "no_labels_from_google"] and \
               result_frame.get("confidence", 0) >= (processor.inferencer.confidence_threshold if processor.inferencer else 0.7) :
                print("\nTesting save_processed_frame...")
                saved_path = processor.save_processed_frame(dummy_frame_data, result_frame, "dummy_frame_capture.jpg")
                if saved_path:
                    print(f"Dummy frame saved to: {saved_path}")
                else:
                    print("Dummy frame not saved (either low confidence or other issue).")
            else:
                print("\nSkipping save_processed_frame test as frame was not confidently processed or had errors.")

        except ImportError:
            print("cv2 import failed in test, cannot create/process dummy frame.")
        except Exception as e:
            print(f"Error during frame processing test: {e}")
    else:
        print("\nSkipping GoogleCVProcessor tests as the API is not ready.")

    print("\nGoogleCVProcessor test finished.")