# filepath: c:\Users\jlatrobe\BoatRampTagger\computer_vision\google_cv_processor.py
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

    def __init__(self, class_names=None, credentials_path=None):
        """
        Initializes the GoogleCVProcessor.

        Args:
            class_names (list, optional): A list of specific class names the application
                                          is interested in. If None, it will try to
                                          determine them from the data directory structure.
            credentials_path (str, optional): Path to Google Cloud service account JSON file.
        """
        current_script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
        self.data_dir = os.path.abspath(os.path.join(current_script_dir, DATA_DIR_RELATIVE))
        
        # Look for Google service account JSON in project root if not specified
        if credentials_path is None:
            default_credentials = os.path.join(project_root, "gptactions-424000-8f24fedcc786.json")
            if os.path.exists(default_credentials):
                credentials_path = default_credentials
                print(f"Found Google service account credentials at: {credentials_path}")
        
        self.class_names = class_names if class_names is not None else []
        if not self.class_names:
            self._determine_class_names() 

        self.inferencer = None
        self.api_ready_for_inference = False # Changed from model_ready_for_inference

        print("Initializing Google Cloud Vision Processor...")
        self.inferencer = GoogleCVInferencer(
            class_names=self.class_names,
            credentials_path=credentials_path
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
        Determines class names by listing subdirectories in `self.data_dir`.
        Populates `self.class_names`.
        """
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            excluded_dirs = {'models', 'detections', 'detections_google_cv'}
            self.class_names = sorted([
                d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d)) and d not in excluded_dirs
            ])
            if not self.class_names:
                print(f"Warning: Data directory {self.data_dir} found, but it does not contain any "
                      "valid class subdirectories (or they are all excluded). "
                      "Google CV will report all labels it finds.")
            else:
                print(f"Target class names determined from data directory: {self.class_names}")
        else:
            print(f"Warning: Data directory {self.data_dir} not found or is not a directory. "
                  "Cannot automatically determine target class names. Google CV will report all labels.")


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
    current_script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
    dummy_image_dir = os.path.join(project_root, "boat_ramp_frames") 
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

    processor = GoogleCVProcessor(class_names=['boat', 'car', 'trailer'])

    if processor.api_ready_for_inference:
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