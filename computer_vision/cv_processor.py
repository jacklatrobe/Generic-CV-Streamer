"""
Provides local computer vision processing of images/frames using AutoKeras.
"""
import os
import numpy as np
import tensorflow as tf
import autokeras as ak
import cv2 # For BGR to RGB conversion if needed for frames

MODEL_FILENAME = "autokeras_image_classifier.keras"
# Data directory relative to this file (cv_processor.py)
# computer_vision/ -> ../data/
DATA_DIR_RELATIVE = "../data" 

class LocalCVProcessor:
    """Uses AutoKeras for local image classification."""

    def __init__(self, max_trials=3, epochs=10):
        """Initializes the LocalCVProcessor with AutoKeras.

        Args:
            max_trials: Maximum number of Keras Models to try.
            epochs: Number of epochs to train the model.
        """
        self.max_trials = max_trials
        self.epochs = epochs
        
        # Determine absolute paths
        current_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(current_dir, MODEL_FILENAME)
        self.data_dir = os.path.abspath(os.path.join(current_dir, DATA_DIR_RELATIVE))

        self.clf = None
        self.class_names = []
        self.model_trained_or_loaded = False

        self._load_or_initialize_model()
        self._ensure_model_trained()

    def _load_or_initialize_model(self):
        """Loads a pre-trained model or initializes a new one."""
        if os.path.exists(self.model_path):
            try:
                print(f"Loading existing model from {self.model_path}...")
                self.clf = tf.keras.models.load_model(self.model_path, custom_objects=ak.CUSTOM_OBJECTS)
                print("Model loaded successfully.")
                self.model_trained_or_loaded = True
            except Exception as e:
                print(f"Failed to load model: {e}. Initializing a new model.")
                self.clf = ak.ImageClassifier(max_trials=self.max_trials, overwrite=True, project_name="autokeras_boat_ramp")
        else:
            print("No pre-trained model found. Initializing a new model.")
            self.clf = ak.ImageClassifier(max_trials=self.max_trials, overwrite=True, project_name="autokeras_boat_ramp")
        
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            self.class_names = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
            if not self.class_names:
                print(f"Warning: Data directory {self.data_dir} does not contain class subdirectories.")
        else:
            print(f"Warning: Data directory {self.data_dir} not found or is not a directory.")


    def _ensure_model_trained(self):
        """Ensures the model is trained if not already loaded."""
        if not self.model_trained_or_loaded:
            if not self.class_names:
                print("Cannot train model: No class data found. Please ensure data is in subdirectories (e.g., data/cars, data/boats) under {self.data_dir}")
                return
            self._train_model()
            self.model_trained_or_loaded = True

    def _train_model(self):
        """Trains the AutoKeras model using data from self.data_dir."""
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            print(f"Data directory {self.data_dir} is empty or does not exist. Skipping training.")
            return

        print(f"Starting model training using data from: {self.data_dir}")
        if not self.class_names:
            print("Error: Class names not determined. Cannot create dataset. Ensure data subdirectories exist.")
            return
        print(f"Class names for training: {self.class_names}")

        # Count total number of images
        total_images = 0
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif') # Common image extensions
        for class_name in self.class_names:
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                try:
                    total_images += len([
                        name for name in os.listdir(class_path)
                        if os.path.isfile(os.path.join(class_path, name)) and name.lower().endswith(image_extensions)
                    ])
                except OSError:
                    print(f"Warning: Could not access or list files in {class_path}")
                    continue
        
        print(f"Found {total_images} total training image files.")

        if total_images == 0:
            print("Error: No training images found. Skipping training.")
            return
        
        if total_images == 1:
            print("ERROR: Only 1 image found. AutoKeras requires the dataset to be splittable into at least 2 batches. Training cannot proceed.")
            return

        # Dynamically set batch_size
        # AutoKeras's fit method needs the input dataset to have at least 2 batches to perform its own split.
        # Number of batches from image_dataset_from_directory = ceil(total_images / batch_size).
        
        batch_size: int
        if total_images < 4: # Covers 2 or 3 images
            batch_size = 1 # This will produce total_images (2 or 3) batches from the loader.
            print(f"WARNING: Very few training images ({total_images}). Using batch_size=1. Model fit may be poor and training unstable.")
        else: # total_images >= 4
            # Set batch_size to half the images, capped at 32, ensuring at least 1.
            # This ensures image_dataset_from_directory produces at least 2 batches.
            batch_size = min(total_images // 2, 32)
            batch_size = max(1, batch_size) # Ensure batch_size is at least 1.
            if total_images < 50:
                print(f"WARNING: Low number of training images ({total_images}). Using batch_size={batch_size}. Model fit may not be optimal. Consider adding more data.")
            elif batch_size < 4 and total_images >= 20: 
                print(f"Note: Using a small batch_size={batch_size} due to dataset size relative to cap (32).")
        
        try:
            # Define image size
            image_size = (256, 256)

            print(f"Loading training data from {self.data_dir} with image_size={image_size} and batch_size={batch_size}")
            
            # Create a tf.data.Dataset from the directory structure
            # Ensure class_names are passed if known, to maintain order and for label inference
            train_dataset = tf.keras.utils.image_dataset_from_directory(
                self.data_dir,
                labels='inferred', # Infers labels from directory names
                label_mode='categorical', # For multi-class classification with AutoKeras
                class_names=self.class_names, # Ensures consistent class indexing
                image_size=image_size,
                batch_size=batch_size,
                shuffle=True,
                seed=123 # for reproducibility
            )
            
            # Verify the dataset output
            for images, labels in train_dataset.take(1):
                print(f"Dataset batch shape: images={images.shape}, labels={labels.shape}")
                break # Just inspect the first batch

            print("Fitting AutoKeras model with the loaded dataset...")
            # AutoKeras ImageClassifier can accept a tf.data.Dataset directly
            self.clf.fit(train_dataset, epochs=self.epochs)
            print("Model training complete.")
            
            print(f"Exporting model to {self.model_path}...")
            self.clf.export_model(self.model_path)
            print("Model exported successfully.")
        except Exception as e:
            print(f"An error occurred during model training or export: {e}")
            import traceback
            traceback.print_exc()

    def process_image(self, image_path: str):
        """Processes an image from a file path using the trained AutoKeras model.

        Args:
            image_path: Path to the image file.

        Returns:
            A dictionary with classification results (tag and confidence).
        """
        if not self.model_trained_or_loaded or not self.clf:
            print("Model not available for processing.")
            return {"tags": ["error_model_not_ready"], "confidence": 0.0}
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return {"tags": ["error_image_not_found"], "confidence": 0.0}

        try:
            img_array = tf.keras.utils.img_to_array(tf.keras.utils.load_img(image_path))
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0) 
            
            prediction = self.clf.predict(img_array)
            
            if not self.class_names: # Should have been set during init/train
                print("Warning: class_names not set. Prediction will be an index.")
                predicted_label = str(np.argmax(prediction[0]))
            else:
                predicted_label = self.class_names[np.argmax(prediction[0])]
            
            confidence = float(np.max(prediction[0]))
            
            print(f"Processed {image_path}: Label = {predicted_label}, Confidence = {confidence:.4f}")
            return {"tags": [predicted_label], "confidence": confidence}
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return {"tags": ["error_processing_image"], "confidence": 0.0}

    def process_frame(self, frame_data: np.ndarray):
        """Processes a raw image frame (e.g., from OpenCV) using AutoKeras.

        Args:
            frame_data: The raw image data (NumPy array, assumed BGR).

        Returns:
            A dictionary with classification results (tag and confidence).
        """
        if not self.model_trained_or_loaded or not self.clf:
            print("Model not available for processing.")
            return {"tags": ["error_model_not_ready"], "confidence": 0.0}

        try:
            # Convert BGR (OpenCV default) to RGB
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            # Add batch dimension
            img_array = np.expand_dims(rgb_frame, axis=0)

            prediction = self.clf.predict(img_array)

            if not self.class_names: # Should have been set during init/train
                print("Warning: class_names not set. Prediction will be an index.")
                predicted_label = str(np.argmax(prediction[0]))
            else:
                predicted_label = self.class_names[np.argmax(prediction[0])]

            confidence = float(np.max(prediction[0]))
            
            # print(f"Processed frame: Label = {predicted_label}, Confidence = {confidence:.4f}") # Can be too verbose for video
            return {"tags": [predicted_label], "confidence": confidence}
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {"tags": ["error_processing_frame"], "confidence": 0.0}

# Example usage (for testing purposes, not part of the class itself)
if __name__ == '__main__':
    # This assumes you run this script from the 'computer_vision' directory
    # and your data is in '../data/'
    print("Initializing CV Processor for testing...")
    # Create dummy data directories and files for testing if they don't exist
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR_RELATIVE))
    dummy_classes = ['cars', 'boats', 'trailers']
    for class_name in dummy_classes:
        class_dir = os.path.join(base_data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        # Create a dummy image file if one doesn't exist
        dummy_image_path = os.path.join(class_dir, f"dummy_{class_name}.jpg")
        if not os.path.exists(dummy_image_path):
            try:
                # Create a small black image
                dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(dummy_image_path, dummy_img)
                print(f"Created dummy image: {dummy_image_path}")
            except Exception as e:
                print(f"Could not create dummy image {dummy_image_path}: {e}")
    
    processor = LocalCVProcessor(max_trials=1, epochs=1) # Quick test

    # Test with a dummy image (if available)
    test_image_path = os.path.join(base_data_dir, 'cars', 'dummy_cars.jpg')
    if os.path.exists(test_image_path):
        print(f"\nTesting process_image with {test_image_path}:")
        result = processor.process_image(test_image_path)
        print(f"Result: {result}")
    else:
        print(f"\nSkipping process_image test, dummy image not found at {test_image_path}")

    # Test with a dummy frame
    print("\nTesting process_frame with a dummy frame:")
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) # Example frame
    result_frame = processor.process_frame(dummy_frame)
    print(f"Result for dummy frame: {result_frame}")
    
    print("\nCV Processor test finished.")
