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

    def __init__(self, credentials_path=None, confidence_threshold=0.7, class_names=None, expansion_map=None, unexpand_detections=True):
        """
        Initializes the GoogleCVProcessor.

        Args:
            credentials_path (str, optional): Path to Google Cloud service account JSON file.
            confidence_threshold (float, optional): Minimum confidence for detection.
            class_names (list[str], optional): List of original target class names.
            expansion_map (dict, optional): Dictionary mapping original class names to lists of expanded terms.
            unexpand_detections (bool): Whether to unexpand detected names back to original class names.
        """
        self.credentials_path = credentials_path
        self.original_class_names = class_names if class_names else []
        self.expansion_map = expansion_map if expansion_map else {}
        self.unexpand_detections = unexpand_detections
        self.detections_save_dir = "detections"
        self.confidence_threshold = confidence_threshold
        
        # Make sure detections directory exists
        os.makedirs(self.detections_save_dir, exist_ok=True)

        self.inferencer = None
        self.api_ready_for_inference = False

        print("Initializing Google Cloud Vision Processor...")
        print(f"Class names: {self.original_class_names}")
        print(f"Expansion map: {self.expansion_map}")
        print(f"Unexpand detections: {self.unexpand_detections}")
        
        self.inferencer = GoogleCVInferencer(
            credentials_path=self.credentials_path,
            class_names=self.original_class_names,
            detections_save_dir=self.detections_save_dir,
            confidence_threshold=self.confidence_threshold,
            expansion_map=self.expansion_map,
            unexpand_detections=self.unexpand_detections
        )

        if self.inferencer and self.inferencer.api_ready:
            self.api_ready_for_inference = True
            print("Google Cloud Vision Processor initialized and API is ready.")
        else:
            self.api_ready_for_inference = False
            print("Google Cloud Vision Processor: API is not ready. Inference will not be available.")
            print("Please ensure 'google-cloud-vision' is installed (pip install google-cloud-vision) "
                  "and your Google Cloud credentials are set up correctly.")

    def process_image(self, image_path: str):
        """
        Processes an image from a file path using the GoogleCVInferencer.
        """
        if self.inferencer and self.inferencer.api_ready:
            return self.inferencer.process_image(image_path)
        else:
            print("Google CV Inferencer not available or API not ready. Cannot process image.")
            return {"tags": ["error_inferencer_not_ready_google"], "confidence": 0.0, "objects": []}

    def process_frame(self, frame_data: np.ndarray):
        """
        Processes a raw image frame (NumPy array) using the GoogleCVInferencer.
        """
        if self.inferencer and self.inferencer.api_ready:
            return self.inferencer.process_frame(frame_data)
        else:
            print("Google CV Inferencer not available or API not ready. Cannot process frame.")
            return {"tags": ["error_inferencer_not_ready_google"], "confidence": 0.0, "objects": []}

    def save_processed_frame(self, frame_data: np.ndarray, result: dict):
        """
        Saves a processed frame with detected objects using the GoogleCVInferencer.
        
        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array.
            result (dict): The result from process_frame containing 'objects' list.
            
        Returns:
            list: Paths to saved detection files.
        """
        if self.inferencer and self.inferencer.api_ready:
            objects_to_save = result.get("objects", [])
            return self.inferencer.save_processed_frame(frame_data, objects_to_save)
        else:
            print("Google CV Inferencer not available. Cannot save frame.")
            return []