"""
Handles inference using a pre-trained AutoKeras image classification model.
"""
import os
import numpy as np
import tensorflow
import autokeras as ak
import cv2

class AutoKerasInferencer:
    """
    Manages loading a pre-trained AutoKeras model and performing inference
    on images or image frames.
    """
    def __init__(self, model_path, class_names):
        """
        Initializes the AutoKerasInferencer.

        Args:
            model_path (str): Path to the saved Keras model file.
            class_names (list): A list of strings representing the class names that
                                the model was trained on, in the correct order.
        """
        self.model_path = model_path
        self.class_names = class_names
        self.clf = None
        self.model_loaded = False
        self._load_model()

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
            
            predicted_label_index = np.argmax(prediction[0])
            if self.class_names and 0 <= predicted_label_index < len(self.class_names):
                predicted_label = self.class_names[predicted_label_index]
            else:
                print(f"Warning: class_names not set or index out of bounds. Prediction will be an index: {predicted_label_index}")
                predicted_label = str(predicted_label_index)
            
            confidence = float(np.max(prediction[0]))
            print(f"Processed {image_path}: Label = {predicted_label}, Confidence = {confidence:.4f}")
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
            if self.class_names and 0 <= predicted_label_index < len(self.class_names):
                predicted_label = self.class_names[predicted_label_index]
            else:
                print(f"Warning: class_names not set or index out of bounds. Prediction will be an index: {predicted_label_index}")
                predicted_label = str(predicted_label_index)
            
            confidence = float(np.max(prediction[0]))
            return {"tags": [predicted_label], "confidence": confidence}
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {"tags": ["error_processing_frame"], "confidence": 0.0}

