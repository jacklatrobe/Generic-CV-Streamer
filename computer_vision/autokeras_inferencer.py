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
import json  # Added import for json

class AutoKerasInferencer:
    """
    Manages loading a pre-trained AutoKeras model and performing inference
    on images or image frames.
    """
    def __init__(self, model_path, class_names):  # Removed class_names from parameters
        """
        Initializes the AutoKerasInferencer.

        Args:
            model_path (str): Path to the saved Keras model file.
            class_names (list): A list of class names corresponding to the model's output indices.
                                This should be loaded from config.json by the calling module.
        """
        self.model_path = model_path
        self.class_names = class_names
        self.clf = None
        self.model_loaded = False
        self.confidence_threshold = 0.7  # 70% confidence threshold

        if not self.class_names:
            raise ValueError("class_names list cannot be empty.")

        # Create detection directories
        self.detections_base_dir = os.path.abspath(os.path.join(os.path.dirname(self.model_path), "..", "detections"))
        self._create_detection_directories()

        self._load_model()

    def _create_detection_directories(self):
        """
        Creates directories for storing detected objects based on class names.
        """
        os.makedirs(self.detections_base_dir, exist_ok=True)
        for class_name in self.class_names:
            os.makedirs(os.path.join(self.detections_base_dir, class_name), exist_ok=True)

    def _save_detection(self, image_path, predicted_label):
        """
        Saves a copy of the detected image to the appropriate detection directory.

        Args:
            image_path (str): Path to the original image file.
            predicted_label (str): The predicted class label.

        Returns:
            str: Path to the saved detection file, or None if save failed.
        """
        try:
            # Create a unique filename using timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            filename = os.path.basename(image_path)
            base_name, ext = os.path.splitext(filename)

            # Destination path in the detections directory
            detection_dir = os.path.join(self.detections_base_dir, predicted_label)
            detection_path = os.path.join(detection_dir, f"{base_name}_{timestamp}{ext}")

            # Copy the image to the detections directory
            shutil.copy2(image_path, detection_path)
            print(f"Saved detection to {detection_path}")
            return detection_path
        except Exception as e:
            print(f"Error saving detection: {e}")
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

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing the predicted 'tags' (list with one class name)
                  and 'confidence' (float). Returns an error tag if processing fails.
        """
        if not self.model_loaded or not self.clf:
            print("Model not available for processing.")
            return {"tags": ["error_model_not_ready"], "confidence": 0.0}
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return {"tags": ["error_image_not_found"], "confidence": 0.0}

        try:
            img = tensorflow.keras.utils.load_img(image_path, target_size=(256, 256))
            img_array = tensorflow.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Wrap predict in custom_object_scope for safety, though usually more critical for loading.
            # This ensures that if the model contains custom AutoKeras layers/objects,
            # they are handled correctly during prediction.
            with tensorflow.keras.utils.custom_object_scope(ak.CUSTOM_OBJECTS):
                prediction = self.clf.predict(img_array)

            # Print confidence for all classes to verify model is considering all classes
            print(f"Confidence scores for {image_path}:")
            for i, class_name in enumerate(self.class_names):
                if i < len(prediction[0]):
                    conf = float(prediction[0][i])
                    print(f"  - {class_name}: {conf:.4f} ({conf*100:.1f}%)")

            predicted_label_index = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))

            # Only process predictions with confidence above the threshold
            if confidence < self.confidence_threshold:
                print(f"Processed {image_path}: Confidence {confidence:.4f} below threshold {self.confidence_threshold}. No label assigned.")
                return {"tags": ["no_confident_match"], "confidence": confidence}

            if self.class_names and 0 <= predicted_label_index < len(self.class_names):
                predicted_label = self.class_names[predicted_label_index]
            else:
                print(f"Warning: class_names not set or index out of bounds. Prediction will be an index: {predicted_label_index}")
                predicted_label = str(predicted_label_index)

            print(f"Processed {image_path}: Label = {predicted_label}, Confidence = {confidence:.4f}")

            # Save detection if confidence is above threshold
            self._save_detection(image_path, predicted_label)

            return {"tags": [predicted_label], "confidence": confidence}
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return {"tags": ["error_processing_image"], "confidence": 0.0}

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
                predicted_label = self.class_names[predicted_label_index]
            else:
                print(f"Warning: class_names not set or index out of bounds. Prediction will be an index: {predicted_label_index}")
                predicted_label = str(predicted_label_index)

            # For frames, we can't directly save them (no path), but frame saving would happen elsewhere
            return {"tags": [predicted_label], "confidence": confidence}
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {"tags": ["error_processing_frame"], "confidence": 0.0}

    def save_processed_frame(self, frame_data: np.ndarray, result: dict, frame_path: str = None):
        """
        Saves a processed frame that has been classified with high confidence.

        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array.
            result (dict): The classification result containing tags and confidence.
            frame_path (str, optional): Path where the frame was saved originally, if available.
                                      Used to generate a descriptive filename.

        Returns:
            str: Path to the saved detection file, or None if save failed.
        """
        if not result or "tags" not in result or not result["tags"] or result["tags"][0] == "no_confident_match":
            return None

        try:
            # Get the predicted class
            predicted_label = result["tags"][0]
            confidence = result["confidence"]

            # Skip if confidence is below threshold
            if confidence < self.confidence_threshold:
                return None

            # Create a unique filename using timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            base_name = os.path.basename(frame_path).split('.')[0] if frame_path else f"frame_{timestamp}"

            # Destination path in the detections directory
            detection_dir = os.path.join(self.detections_base_dir, predicted_label)
            detection_path = os.path.join(detection_dir, f"{base_name}_{confidence:.2f}.jpg")

            # Save the frame to the detections directory
            cv2.imwrite(detection_path, frame_data)
            print(f"Saved detection to {detection_path}")
            return detection_path
        except Exception as e:
            print(f"Error saving detection frame: {e}")
            return None

