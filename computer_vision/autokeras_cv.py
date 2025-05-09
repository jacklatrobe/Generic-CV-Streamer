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
import numpy as np # Keep for dummy frame generation in example
import cv2 # Keep for dummy frame generation and BGR-RGB conversion in example

from .autokeras_trainer import AutoKerasTrainer
from .autokeras_inferencer import AutoKerasInferencer

MODEL_FILENAME = "autokeras_image_classifier.keras"
DATA_DIR_RELATIVE = "../data" # Relative to the 'computer_vision' directory
AUTOKERAS_PROJECT_NAME = "autokeras_boat_ramp_project"

class AutoKerasCVProcessor:
    """
    Orchestrates AutoKeras training and inference for image classification.

    This class acts as a high-level interface to the AutoKeras-based
    computer vision functionalities. It manages the model lifecycle, including
    training (if necessary or forced) and setting up an inferencer for predictions.
    It determines class names from the data directory structure.
    """

    def __init__(self, force_train=False, max_trials=3, epochs=10):
        """
        Initializes the AutoKerasCVProcessor.

        Determines if a model needs to be trained or if an existing one can be loaded.
        If training is required (either `force_train` is True or no model exists at
        `self.model_path`), it instantiates and uses `AutoKerasTrainer`.
        After ensuring a model is available (either trained or pre-existing),
        it sets up `AutoKerasInferencer` for processing images/frames.

        Args:
            force_train (bool, optional): If True, training will be forced even if a
                                          model file already exists at `self.model_path`.
                                          Defaults to False.
            max_trials (int, optional): Maximum number of different Keras Models for
                                        AutoKeras to try during the training phase.
                                        Passed to `AutoKerasTrainer`. Defaults to 3.
            epochs (int, optional): Number of epochs to train each model for during
                                    AutoKeras training. Passed to `AutoKerasTrainer`.
                                    Defaults to 10.
        """
        current_script_dir = os.path.dirname(__file__) # computer_vision directory
        
        # Define the base directory for models, relative to the project root
        # Project root is one level up from current_script_dir
        project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
        self.model_storage_dir = os.path.join(project_root, "data", "models")
        
        # Ensure the model storage directory exists
        os.makedirs(self.model_storage_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.model_storage_dir, MODEL_FILENAME)
        
        # Data directory for training images remains the same
        self.data_dir = os.path.abspath(os.path.join(current_script_dir, DATA_DIR_RELATIVE))
        
        self.class_names = []
        self._determine_class_names() # Populates self.class_names

        self.inferencer = None
        self.model_ready_for_inference = False

        # Training logic: train if forced or if the model file doesn't exist.
        if force_train or not os.path.exists(self.model_path):
            if force_train:
                print(f"Training is forced for model: {self.model_path}")
            else:
                print(f"Model not found at {self.model_path}. Training required.")
            
            trainer = AutoKerasTrainer(
                data_dir=self.data_dir, 
                model_path=self.model_path, # This path now points to data/models/
                class_names=self.class_names,
                project_name=AUTOKERAS_PROJECT_NAME, # AK project will be created in os.path.dirname(self.model_path)
                max_trials=max_trials, 
                epochs=epochs
            )
            if trainer.train_and_export():
                print("Training completed and model exported successfully.")
                self.model_ready_for_inference = True
            else:
                print("Training failed or model was not exported. Inference will not be available.")
                self.model_ready_for_inference = False # Explicitly set
        else:
            print(f"Model already exists at {self.model_path}. Skipping training.")
            self.model_ready_for_inference = True

        # Initialize inferencer if the model is ready
        if self.model_ready_for_inference:
            self.inferencer = AutoKerasInferencer(self.model_path, self.class_names)
            if not self.inferencer.model_loaded:
                 # This indicates an issue during model loading by the inferencer itself
                 print("Warning: Inferencer was created, but it reported an issue loading the model. Inference may not work as expected.")
                 self.model_ready_for_inference = False # Update status if inferencer failed to load
        else:
            print("Model is not ready for inference. Inferencer will not be initialized.")

    def _determine_class_names(self):
        """
        Determines class names by listing subdirectories in `self.data_dir`.
        The names are sorted alphabetically to ensure consistent class indexing.
        Populates `self.class_names`.
        Prints warnings if the data directory is not found or is empty/has no subdirs.
        """
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            # List directories (which are assumed to be class names)
            self.class_names = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
            if not self.class_names:
                print(f"Warning: Data directory {self.data_dir} found, but it does not contain any class subdirectories.")
        else:
            print(f"Warning: Data directory {self.data_dir} not found or is not a directory. Cannot determine class names.")

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
    
    # --- Test Case 1: Force training ---
    print("\n--- Testing with force_train=True (quick training) ---")
    # Using minimal trials/epochs for faster testing
    processor_train = AutoKerasCVProcessor(force_train=True, max_trials=1, epochs=1) 

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
        print("Skipping inference tests as model training failed or model was not ready after forced training.")

    # --- Test Case 2: Load existing model (if training was successful) ---
    if processor_train.model_ready_for_inference: # Proceed only if the first training created a model
        print("\n--- Testing with existing model (force_train=False) ---")
        processor_load = AutoKerasCVProcessor(force_train=False) # Should load the model trained above
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
