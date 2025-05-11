"""
Orchestrates the AutoKeras-based computer vision processing pipeline.

This module provides the `AutoKerasCVProcessor` class, which handles:
- Determining if model training is necessary based on existing model files or explicit flags.
- Invoking the `AutoKerasTrainer` for model training if required.
- Initializing the `AutoKerasInferencer` with a trained model for predictions.
- Providing high-level methods to process images (from file paths) and
  image frames (raw NumPy arrays) for classification.
"""
import os
import numpy as np
import cv2 # Keep for dummy frame generation and BGR-RGB conversion in example

from .autokeras_trainer import AutoKerasTrainer
from .autokeras_inferencer import AutoKerasInferencer

class AutoKerasCVProcessor:
    """
    Manages the AutoKeras model, including training and inference.
    """

    def __init__(self, retrain_model=False, class_names=None, model_dir=None, model_filename=None, image_data_dir="image_data"):
        """
        Initializes the AutoKerasCVProcessor.

        Args:
            retrain_model (bool): Whether to retrain the model.
            class_names (list): List of class names for the model.
            model_dir (str): Directory where the model is stored or will be saved.
            model_filename (str): Filename for the Keras model.
            image_data_dir (str): Directory containing training/testing images.
        """
        if class_names is None:
            raise ValueError("class_names must be provided to AutoKerasCVProcessor.")
        if model_dir is None or model_filename is None:
            raise ValueError("model_dir and model_filename must be provided.")

        self.class_names = class_names
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.model_path = os.path.join(self.model_dir, self.model_filename)
        self.image_data_dir = image_data_dir # Used by the trainer

        self.model_ready_for_inference = False
        self.inferencer = None # Initialize later

        if retrain_model:
            print("Retraining requested for AutoKeras model.")
            self._train_model()
        else:
            print("Attempting to load existing AutoKeras model.")
            if os.path.exists(self.model_path):
                print(f"Found existing model at {self.model_path}")
                self._initialize_inferencer() # Initialize inferencer with existing model
            else:
                print(f"No existing model found at {self.model_path}. Model training is required or path is incorrect.")
                # Optionally, you could trigger training here if no model exists and retrain_model was false
                # self._train_model() 
                # For now, we just indicate it's not ready.

    def _initialize_inferencer(self):
        """Initializes the AutoKerasInferencer with the current model path and class names."""
        try:
            self.inferencer = AutoKerasInferencer(
                model_path=self.model_path, 
                class_names=self.class_names
            )
            self.model_ready_for_inference = True
            print("AutoKerasInferencer initialized successfully.")
        except Exception as e:
            print(f"Error initializing AutoKerasInferencer: {e}")
            self.model_ready_for_inference = False

    def _train_model(self):
        """Trains the AutoKeras model."""
        print(f"Starting AutoKeras model training. Image data directory: {self.image_data_dir}")
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        trainer = AutoKerasTrainer(
            image_data_dir=self.image_data_dir, 
            model_save_path=self.model_path,
            class_names=self.class_names # Pass class_names to trainer
        )
        try:
            trained_model = trainer.train()
            if trained_model:
                print(f"Model training complete. Model saved to {self.model_path}")
                self._initialize_inferencer() # Initialize inferencer with the newly trained model
            else:
                print("Model training did not complete successfully or no model was returned.")
                self.model_ready_for_inference = False # Ensure this is set
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            self.model_ready_for_inference = False # Ensure this is set

    def process_image(self, image_path: str):
        """
        Processes an image from a file path using the configured `AutoKerasInferencer`.

        Args:
            image_path (str): Absolute or relative path to the image file.

        Returns:
            dict: A dictionary with classification results (e.g., {'tags': ['boat'], 'confidence': 0.95})
                  from the inferencer, or an error dictionary if the inferencer is not ready
                  (e.g., {'tags': ['error_inferencer_not_ready'], 'confidence': 0.0}).
        """
        if self.inferencer and self.inferencer.model_loaded:
            return self.inferencer.process_image(image_path)
        else:
            print("Inferencer not available or model not loaded. Cannot process image.")
            return {"tags": ["error_inferencer_not_ready"], "confidence": 0.0}

    def process_frame(self, frame_data: np.ndarray):
        """
        Processes a raw image frame (NumPy array) using the configured `AutoKerasInferencer`.

        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array (e.g., from OpenCV).

        Returns:
            dict: A dictionary with classification results (e.g., {'tags': ['boat'], 'confidence': 0.95})
                  from the inferencer, or an error dictionary if the inferencer is not ready
                  (e.g., {'tags': ['error_inferencer_not_ready'], 'confidence': 0.0}).
        """
        if self.inferencer and self.inferencer.model_loaded:
            return self.inferencer.process_frame(frame_data)
        else:
            print("Inferencer not available or model not loaded. Cannot process frame.")
            return {"tags": ["error_inferencer_not_ready"], "confidence": 0.0}

# Example usage (for testing purposes when running this file directly)
if __name__ == '__main__':
    print("Initializing AutoKerasCVProcessor for testing...")
    # This example assumes the script is run from the 'computer_vision' directory
    
    # Define constants for the test, as they are no longer hardcoded in the class
    DATA_DIR_RELATIVE = "../../image_data" # Relative path to image_data from computer_vision directory
    MODEL_FILENAME = "test_model.keras" # Define a model filename for the test

    current_script_dir = os.path.dirname(__file__)
    # Training data directory
    base_data_dir = os.path.abspath(os.path.join(current_script_dir, DATA_DIR_RELATIVE))
    # Model storage directory (for verification if needed, though not directly used by test logic below)
    model_output_dir = os.path.abspath(os.path.join(current_script_dir, "..", "data", "models"))
    print(f"Models will be stored in: {model_output_dir}")
    print(f"Training data will be sourced from: {base_data_dir}")

    # Ensure dummy data directories and a few images exist for testing
    dummy_classes = ['cars', 'boats', 'trailers']
    for class_name_item in dummy_classes:
        class_dir = os.path.join(base_data_dir, class_name_item)
        os.makedirs(class_dir, exist_ok=True)
        # Create a couple of dummy image files in each class directory if they don't exist
        for i in range(2): # Create 2 dummy images per class
            dummy_image_path = os.path.join(class_dir, f"dummy_{class_name_item}_{i}.jpg")
            if not os.path.exists(dummy_image_path):
                try:
                    # Create a small black image
                    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8) # Small image for quick test
                    cv2.imwrite(dummy_image_path, dummy_img)
                    print(f"Created dummy image: {dummy_image_path}")
                except Exception as e:
                    print(f"Could not create dummy image {dummy_image_path}: {e}")
    
    # --- Test Case 1: Retrain model ---
    print("\n--- Testing with retrain_model=True (quick training) ---")
    # Using minimal trials/epochs for faster testing
    processor_train = AutoKerasCVProcessor(retrain_model=True, class_names=dummy_classes, model_dir=model_output_dir, model_filename=MODEL_FILENAME, image_data_dir=base_data_dir) 

    if processor_train.model_ready_for_inference:
        # Test with a dummy image from one of the classes
        test_image_path = os.path.join(base_data_dir, 'cars', 'dummy_cars_0.jpg')
        if os.path.exists(test_image_path):
            print(f"\nTesting process_image with {test_image_path}:")
            result = processor_train.process_image(test_image_path)
            print(f"Result: {result}")
        else:
            print(f"\nSkipping process_image test, dummy image not found at {test_image_path}")

        # Test with a dummy frame
        print("\nTesting process_frame with a dummy frame:")
        dummy_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) # Example frame
        result_frame = processor_train.process_frame(dummy_frame)
        print(f"Result for dummy frame: {result_frame}")
    else:
        print("Skipping inference tests as model training failed or model was not ready after retraining.")

    # --- Test Case 2: Load existing model (if training was successful) ---
    if processor_train.model_ready_for_inference: # Proceed only if the first training created a model
        print("\n--- Testing with existing model (retrain_model=False) ---")
        processor_load = AutoKerasCVProcessor(retrain_model=False, class_names=dummy_classes, model_dir=model_output_dir, model_filename=MODEL_FILENAME) # Should load the model trained above
        if processor_load.model_ready_for_inference:
            test_image_path_boats = os.path.join(base_data_dir, 'boats', 'dummy_boats_0.jpg')
            if os.path.exists(test_image_path_boats):
                print(f"\nTesting process_image with {test_image_path_boats} using loaded model:")
                result = processor_load.process_image(test_image_path_boats)
                print(f"Result: {result}")
            else:
                print(f"\nSkipping process_image test, dummy image not found at {test_image_path_boats}")
        else:
            print("Failed to initialize processor with existing model, or model was not deemed ready.")
    else:
        print("\nSkipping test for loading existing model as prior training may have failed.")
    
    print("\nAutoKerasCVProcessor test finished.")
