"""
Handles inference using a pre-trained AutoKeras image classification model.
"""
import os
import numpy as np
import tensorflow
import autokeras as ak
import cv2
import shutil
from datetime import datetime
import json 
import uuid # Added for unique filenames

class AutoKerasInferencer:
    """
    Manages loading a pre-trained AutoKeras model and performing inference
    on images or image frames.
    """
    def __init__(self, model_path): # Removed class_names from parameters
        """
        Initializes the AutoKerasInferencer.

        Args:
            model_path (str): Path to the saved Keras model file.
        """
        self.model_path = model_path
        self.clf = None
        self.model_loaded = False
        # These will be loaded from config:
        self.class_names = []
        self.confidence_threshold = 0.7  # Default, will be overridden by config
        self.detections_save_dir = "detections" # Default, will be overridden by config

        self._load_config() # Load settings from config.json

        if not self.class_names:
            # This can be a non-fatal warning if we allow saving with index as class name
            print("Warning: class_names list is empty after loading config. Model might not have defined classes or config is missing.")

        self._create_detection_directories() # Uses self.detections_save_dir
        self._load_model()

    def _load_config(self):
        """
        Loads settings from config.json located at the project root.
        Assumes this script is in computer_vision, so config is one level up.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(project_root, "config.json")

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.class_names = config.get("class_names", [])
                self.confidence_threshold = float(config.get("autokeras_confidence_threshold", self.confidence_threshold))
                self.detections_save_dir = config.get("detections_save_dir", self.detections_save_dir)
                
                if not self.class_names:
                    print(f"Warning: 'class_names' not found or empty in {config_path}.")
                else:
                    print(f"AutoKeras: Loaded class_names: {self.class_names} from {config_path}")
                print(f"AutoKeras: Confidence threshold set to: {self.confidence_threshold}")
                print(f"AutoKeras: Detections will be saved to: {self.detections_save_dir}")
        except FileNotFoundError:
            print(f"Error: {config_path} not found. Using default settings for AutoKerasInferencer.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode {config_path}. Using default settings for AutoKerasInferencer.")
        except ValueError:
            print(f"Error: Invalid 'autokeras_confidence_threshold' in {config_path}. Using default {self.confidence_threshold}.")
        except Exception as e:
            print(f"An unexpected error occurred while loading config for AutoKerasInferencer: {e}. Using defaults.")
        self.class_names = [name.lower() for name in self.class_names] # Normalize

    def _create_detection_directories(self):
        """
        Creates directories for storing classified images based on class names, using detections_save_dir.
        """
        if not self.detections_save_dir:
            print("Error: detections_save_dir is not set. Cannot create directories.")
            return
        os.makedirs(self.detections_save_dir, exist_ok=True)
        for class_name in self.class_names:
            # Ensure class_name is a valid directory name (it should be from config)
            os.makedirs(os.path.join(self.detections_save_dir, class_name), exist_ok=True)
        # Also create a directory for 'unknown' or index-based classifications if class_names is empty
        if not self.class_names:
            os.makedirs(os.path.join(self.detections_save_dir, "unknown_classification"), exist_ok=True)

    def _save_detection(self, image_path, predicted_label):
        """
        Saves a copy of the classified image to the appropriate detection directory.
        The entire image is saved as AutoKeras is a classifier.

        Args:
            image_path (str): Path to the original image file.
            predicted_label (str): The predicted class label (already lowercased if from class_names).

        Returns:
            str: Path to the saved detection file, or None if save failed.
        """
        if not self.detections_save_dir:
            print("Error: detections_save_dir not configured. Cannot save detection.")
            return None
        try:
            unique_id = uuid.uuid4()
            # Save as PNG as per user request for detections
            filename = f"{unique_id}.png"
            
            # Determine category directory
            category_dir_name = predicted_label
            if predicted_label not in self.class_names and self.class_names: # If label is an index or unexpected
                print(f"Warning: Predicted label '{predicted_label}' not in known class_names. Saving to 'unknown_classification'.")
                category_dir_name = "unknown_classification"
            elif not self.class_names: # If no class_names defined, save to a default folder
                 category_dir_name = "unknown_classification"

            detection_category_dir = os.path.join(self.detections_save_dir, category_dir_name)
            os.makedirs(detection_category_dir, exist_ok=True) # Ensure it exists
            detection_path = os.path.join(detection_category_dir, filename)

            # Read the image and save it as PNG
            img_to_save = cv2.imread(image_path)
            if img_to_save is None:
                print(f"Error: Could not read image from {image_path} for saving.")
                return None
            
            cv2.imwrite(detection_path, img_to_save)
            print(f"AutoKeras: Saved classified image to {detection_path}")
            return detection_path
        except Exception as e:
            print(f"Error saving AutoKeras detection: {e}")
            return None

    def _load_model(self):
        """
        Loads the Keras model from the specified model_path.
        It uses `tensorflow.keras.utils.custom_object_scope` with `ak.CUSTOM_OBJECTS`
        to ensure any custom layers or objects used by AutoKeras are recognized.
        Sets `self.model_loaded` to True if successful, False otherwise.
        """
        if os.path.exists(self.model_path):
            try:
                print(f"Loading existing model from {self.model_path} for inferencing...")
                # When loading, ensure custom_objects from AutoKeras are passed
                self.clf = tensorflow.keras.models.load_model(self.model_path, custom_objects=ak.CUSTOM_OBJECTS)
                print("Model loaded successfully for inferencing.")
                self.model_loaded = True
            except Exception as e:
                print(f"Failed to load model for inferencing: {e}. Ensure the model was trained and exported correctly.")
                self.model_loaded = False
        else:
            print(f"Model file not found at {self.model_path}. Cannot perform inference.")
            self.model_loaded = False

    def process_image(self, image_path: str):
        """
        Processes an image from a file path using the loaded AutoKeras model.
        Does NOT save the image; saving is handled by save_processed_frame.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing the predicted 'tags' (list with one class name)
                  and 'confidence' (float), and an empty 'objects' list. 
                  Returns an error tag if processing fails.
        """
        if not self.model_loaded or not self.clf:
            print("Model not available for processing.")
            return {"tags": ["error_model_not_ready"], "confidence": 0.0, "objects": []}
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return {"tags": ["error_image_not_found"], "confidence": 0.0, "objects": []}

        try:
            img = tensorflow.keras.utils.load_img(image_path, target_size=(256, 256))
            img_array = tensorflow.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            with tensorflow.keras.utils.custom_object_scope(ak.CUSTOM_OBJECTS):
                prediction = self.clf.predict(img_array)

            # print(f"Confidence scores for {image_path}:") # Optional: for debugging
            # for i, class_name in enumerate(self.class_names):
            #     if i < len(prediction[0]):
            #         conf = float(prediction[0][i])
            #         print(f"  - {class_name}: {conf:.4f} ({conf*100:.1f}%)")

            predicted_label_index = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))

            if confidence < self.confidence_threshold:
                print(f"Processed {image_path}: Confidence {confidence:.4f} below threshold {self.confidence_threshold}. No label assigned.")
                return {"tags": ["no_confident_match"], "confidence": confidence, "objects": []}

            if self.class_names and 0 <= predicted_label_index < len(self.class_names):
                predicted_label = self.class_names[predicted_label_index]
            else:
                print(f"Warning: class_names not set, empty, or index {predicted_label_index} out of bounds. Prediction will be an index.")
                predicted_label = str(predicted_label_index)

            print(f"Processed {image_path}: Label = {predicted_label}, Confidence = {confidence:.4f}")

            # REMOVED: self._save_detection(image_path, predicted_label)
            # Saving is now handled by save_processed_frame, called by FrameExtractor

            return {"tags": [predicted_label], "confidence": confidence, "objects": []}
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return {"tags": ["error_processing_image"], "confidence": 0.0, "objects": []}

    def process_frame(self, frame_data: np.ndarray):
        """
        Processes a raw image frame (e.g., from an OpenCV video stream)
        using the loaded AutoKeras model.

        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array.
                                     It is assumed to be in BGR format if coming from OpenCV.

        Returns:
            dict: A dictionary containing the predicted 'tags' (list with one class name)
                  and 'confidence' (float). Returns an error tag if processing fails.
        """
        if not self.model_loaded or not self.clf:
            print("Model not available for processing.")
            return {"tags": ["error_model_not_ready"], "confidence": 0.0}

        try:
            # Assuming frame_data is a NumPy array (e.g., from OpenCV BGR)
            # Convert BGR to RGB, then resize
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            # Resize using TensorFlow to ensure consistency with training preprocessing.
            # The model was trained with (256, 256) images.
            img_resized = tensorflow.image.resize(rgb_frame, (256, 256))
            img_array = tensorflow.keras.utils.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            # As with process_image, custom_object_scope is used for robustness.
            with tensorflow.keras.utils.custom_object_scope(ak.CUSTOM_OBJECTS):
                prediction = self.clf.predict(img_array)

            predicted_label_index = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))

            # Only process predictions with confidence above the threshold
            if confidence < self.confidence_threshold:
                return {"tags": ["no_confident_match"], "confidence": confidence}

            if self.class_names and 0 <= predicted_label_index < len(self.class_names):
                predicted_label = self.class_names[predicted_label_index] # Already lowercased
            else:
                print(f"Warning: class_names not set, empty, or index {predicted_label_index} out of bounds. Prediction will be an index.")
                predicted_label = str(predicted_label_index)

            # For frames, we don't save here. save_processed_frame will be called by main.
            return {"tags": [predicted_label], "confidence": confidence, "objects": []} # Add empty objects for consistency
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {"tags": ["error_processing_frame"], "confidence": 0.0, "objects": []}

    def save_processed_frame(self, frame_data: np.ndarray, result: dict, frame_path: str = None):
        """
        Saves a processed frame that has been classified with high confidence by AutoKeras.
        The entire frame is saved as AutoKeras is a classifier.

        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array.
            result (dict): The classification result containing tags and confidence.
            frame_path (str, optional): Path where the frame was saved originally (not used for content, but could inspire name if needed).

        Returns:
            str: Path to the saved detection file, or None if save failed.
        """
        if not self.detections_save_dir:
            print("Error: detections_save_dir not configured. Cannot save frame.")
            return None
        if not result or not result.get("tags") or not result["tags"][0]:
            return None
        
        tag = result["tags"][0]
        if tag.startswith("error_") or tag == "no_confident_match":
            return None

        confidence = result.get("confidence", 0.0)
        if confidence < self.confidence_threshold:
            return None

        try:
            predicted_label = tag # Already lowercased if from class_names or is an index string
            
            unique_id = uuid.uuid4()
            filename = f"{unique_id}.png" # Save as PNG

            category_dir_name = predicted_label
            if predicted_label not in self.class_names and self.class_names:
                print(f"Warning: Predicted label '{predicted_label}' not in known class_names. Saving to 'unknown_classification'.")
                category_dir_name = "unknown_classification"
            elif not self.class_names:
                 category_dir_name = "unknown_classification"

            detection_category_dir = os.path.join(self.detections_save_dir, category_dir_name)
            os.makedirs(detection_category_dir, exist_ok=True)
            detection_path = os.path.join(detection_category_dir, filename)

            cv2.imwrite(detection_path, frame_data)
            print(f"AutoKeras: Saved classified frame to {detection_path}")
            return detection_path
        except Exception as e:
            print(f"Error saving AutoKeras classified frame: {e}")
            return None

