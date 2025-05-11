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

    def __init__(self, retrain_model=False, class_names=None, expansion_map=None, unexpand_detections=True, model_dir=None, model_filename=None, image_data_dir="image_data", confidence_threshold=0.5): # Added expansion_map, unexpand_detections, confidence_threshold
        """
        Initializes the AutoKerasCVProcessor.

        Args:
            retrain_model (bool): Whether to retrain the model.
            class_names (list): List of original class names for the model.
            expansion_map (dict, optional): Dictionary for expanding class names. Used for training dir structure and potentially for unexpanding.
            unexpand_detections (bool): Whether to unexpand detected names (primarily for consistency, though AutoKeras usually gives original class names).
            model_dir (str): Directory where the model is stored or will be saved.
            model_filename (str): Filename for the Keras model.
            image_data_dir (str): Base directory containing training/testing images.
            confidence_threshold (float): Confidence threshold for detections.
        """
        if class_names is None:
            raise ValueError("class_names must be provided to AutoKerasCVProcessor.")
        if model_dir is None or model_filename is None:
            raise ValueError("model_dir and model_filename must be provided.")

        self.original_class_names = [name.lower() for name in class_names if isinstance(name, str)]
        self.expansion_map = expansion_map if expansion_map else {}
        self.unexpand_detections = unexpand_detections # Stored, though direct application in AutoKeras might be limited
        self.confidence_threshold = confidence_threshold # Store confidence threshold
        
        # For AutoKeras, the class_names passed to trainer/inferencer are typically the direct subfolder names
        # If expansion_map is used, it implies the image_data_dir might be structured with original names,
        # and we need to tell AutoKeras about all *expanded* names if it were to predict them.
        # However, AutoKeras is trained on specific folder names. The expansion map is more for *interpreting* its output
        # or for standardizing how *other* CV systems might see the classes.
        # For training AutoKeras, the class_names should be the actual directory names it will find under image_data_dir.
        # Let's assume image_data_dir is structured with original_class_names as subdirectories.
        self.training_class_names = list(self.original_class_names) # These are the names AutoKeras will train on (folder names)
        
        # Prepare a reverse map for unexpansion, similar to other processors
        self._reverse_expansion_map = {}
        if self.expansion_map:
            print(f"Info: Preparing reverse expansion map for AutoKeras based on: {self.expansion_map}")
            for original_name, expanded_terms_list in self.expansion_map.items():
                original_name_lower = original_name.lower()
                for term in expanded_terms_list:
                    term_lower = term.lower()
                    # If an expanded term is already mapped, prefer the shorter original name (heuristic)
                    if term_lower not in self._reverse_expansion_map or \
                       len(original_name_lower) < len(self._reverse_expansion_map[term_lower]):
                        self._reverse_expansion_map[term_lower] = original_name_lower
            # Ensure original names also map to themselves if not an expansion term elsewhere
            for original_name_l in self.original_class_names:
                if original_name_l not in self._reverse_expansion_map:
                    self._reverse_expansion_map[original_name_l] = original_name_l
        else: # No expansion map, so original names map to themselves
            for original_name_l in self.original_class_names:
                self._reverse_expansion_map[original_name_l] = original_name_l
        print(f"Info: Reverse expansion map for AutoKeras: {self._reverse_expansion_map}")

        self.model_dir = model_dir
        self.model_filename = model_filename
        self.model_path = os.path.join(self.model_dir, self.model_filename)
        self.image_data_dir = image_data_dir

        self.model_ready_for_inference = False
        self.inferencer = None

        if retrain_model:
            print("Retraining requested for AutoKeras model.")
            self._train_model()
        else:
            print("Attempting to load existing AutoKeras model.")
            if os.path.exists(self.model_path):
                print(f"Found existing model at {self.model_path}")
                self._initialize_inferencer()
            else:
                print(f"No existing model found at {self.model_path}. Training is required or path is incorrect.")

    def _initialize_inferencer(self):
        """Initializes the AutoKerasInferencer."""
        try:
            self.inferencer = AutoKerasInferencer(
                model_path=self.model_path, 
                class_names=self.training_class_names, # Inferencer needs to know the classes it was trained on
                confidence_threshold=self.confidence_threshold # Pass confidence threshold
            )
            self.model_ready_for_inference = self.inferencer.model_loaded # Check if model loaded successfully
            if self.model_ready_for_inference:
                print("AutoKerasInferencer initialized successfully.")
            else:
                print("AutoKerasInferencer initialized, but model was not loaded successfully.")
        except Exception as e:
            print(f"Error initializing AutoKerasInferencer: {e}")
            self.model_ready_for_inference = False

    def _train_model(self):
        """Trains the AutoKeras model using self.training_class_names."""
        print(f"Starting AutoKeras model training. Image data directory: {self.image_data_dir}")
        print(f"Training for classes (directory names): {self.training_class_names}")
        os.makedirs(self.model_dir, exist_ok=True)

        trainer = AutoKerasTrainer(
            image_data_dir=self.image_data_dir, 
            model_save_path=self.model_path,
            class_names=self.training_class_names # Trainer uses these for finding subdirectories
        )
        try:
            trained_model = trainer.train()
            if trained_model:
                print(f"Model training complete. Model saved to {self.model_path}")
                self._initialize_inferencer() 
            else:
                print("Model training did not complete successfully or no model was returned.")
                self.model_ready_for_inference = False
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            self.model_ready_for_inference = False

    def _unexpand_detection_name(self, detected_name: str) -> str:
        """
        Unexpands a detected name to its original class name if unexpand_detections is True.
        AutoKeras typically predicts one of the `training_class_names`.
        This function maps that name back using the `_reverse_expansion_map` if needed.
        """
        if not self.unexpand_detections or not self._reverse_expansion_map:
            return detected_name # Return as is if not unexpanding or no map
        
        # AutoKeras inferencer should return one of the self.training_class_names.
        # These training_class_names are derived from self.original_class_names.
        # The _reverse_expansion_map is built to map expanded terms back to original names.
        # If training_class_names are original names, this will map original -> original.
        # If training_class_names were hypothetically expanded terms (not current setup), it would map expanded -> original.
        unexpanded_name = self._reverse_expansion_map.get(detected_name.lower(), detected_name.lower())
        print(f"Debug Autokeras Unexpanding: '{detected_name.lower()}' -> '{unexpanded_name}' (Unexpand active: {self.unexpand_detections})")
        return unexpanded_name

    def process_image(self, image_path: str):
        """
        Processes an image from a file path.
        Applies unexpansion to the detected class name if enabled.
        """
        if self.inferencer and self.inferencer.model_loaded:
            result = self.inferencer.process_image(image_path)
            # AutoKeras inferencer returns a dict like {"tags": ["class_name"], "confidence": 0.95, "objects": [...]}
            # The "tags" will contain the class name it predicted (one of self.training_class_names)
            if result and result.get("tags") and self.unexpand_detections:
                detected_tag = result["tags"][0] # Assuming one primary tag/class from AutoKeras classification
                unexpanded_tag = self._unexpand_detection_name(detected_tag)
                result["tags"] = [unexpanded_tag]
                # If 'objects' are present and have names, unexpand them too if necessary
                # For AutoKeras classification, 'objects' might be less common or structured differently than object detection APIs
                if "objects" in result:
                    for obj_info in result.get("objects", []):
                        if "name" in obj_info:
                            obj_info["name"] = self._unexpand_detection_name(obj_info["name"])
            return result
        else:
            print("Inferencer not available or model not loaded. Cannot process image.")
            return {"tags": ["error_inferencer_not_ready"], "confidence": 0.0, "objects": []}

    def process_frame(self, frame_data: np.ndarray):
        """
        Processes a raw image frame.
        Applies unexpansion to the detected class name if enabled.
        """
        if self.inferencer and self.inferencer.model_loaded:
            result = self.inferencer.process_frame(frame_data)
            if result and result.get("tags") and self.unexpand_detections:
                detected_tag = result["tags"][0]
                unexpanded_tag = self._unexpand_detection_name(detected_tag)
                result["tags"] = [unexpanded_tag]
                if "objects" in result:
                    for obj_info in result.get("objects", []):
                        if "name" in obj_info:
                            obj_info["name"] = self._unexpand_detection_name(obj_info["name"])
            return result
        else:
            print("Inferencer not available or model not loaded. Cannot process frame.")
            return {"tags": ["error_inferencer_not_ready"], "confidence": 0.0, "objects": []}

    # Add a save_processed_frame method for consistency, though AutoKerasInferencer might not do cropping by default.
    # This would be called by FrameExtractor.
    # AutoKeras is typically image classification, not object detection with bounding boxes unless the inferencer is adapted.
    # Assuming the inferencer might be updated to return bounding boxes in its 'objects' list.
    def save_processed_frame(self, image_np: np.ndarray, result_from_process_frame: dict):
        """
        Saves cropped images of detected objects if bounding boxes are provided by the inferencer.
        This is a placeholder and depends on AutoKerasInferencer providing bounding box info.
        Applies unexpansion to category name if enabled.

        Args:
            image_np (np.ndarray): The image data as a NumPy array (BGR format from OpenCV).
            result_from_process_frame (dict): The result dictionary from process_frame, expected to contain 'objects'.
                                            Each object in 'objects' should have 'name', 'confidence', 'bounding_box'.
        Returns:
            list: A list of paths to the saved detection files.
        """
        saved_files = []
        detections_save_dir = "detections" # Should ideally come from config via main
        # Try to get detections_save_dir from a config manager if available, or use a default
        # This part is a bit of a simplification as AutoKerasCVProcessor doesn't have direct config access here.
        # For a robust solution, detections_save_dir should be passed in or set during __init__.

        result_objects = result_from_process_frame.get("objects", [])

        if not result_objects: # If no objects with bounding boxes, nothing to save by cropping
            # If the main tag is present and confident, we could save the whole frame under that unexpanded tag.
            # This is an alternative for classification-only models.
            primary_tag_list = result_from_process_frame.get("tags", [])
            primary_confidence = result_from_process_frame.get("confidence", 0.0)
            if primary_tag_list and primary_confidence >= self.confidence_threshold:
                detected_name = primary_tag_list[0]
                save_category_name = self._unexpand_detection_name(detected_name) if self.unexpand_detections else detected_name
                
                category_save_dir = os.path.join(detections_save_dir, save_category_name)
                try:
                    os.makedirs(category_save_dir, exist_ok=True)
                    import uuid # for unique filename
                    filename = f"frame_{uuid.uuid4()}.png"
                    save_path = os.path.join(category_save_dir, filename)
                    cv2.imwrite(save_path, image_np)
                    print(f"Info (AutoKeras): Saved full frame for class '{save_category_name}' to {save_path}")
                    saved_files.append(save_path)
                except Exception as e:
                    print(f"Error (AutoKeras): Could not save full frame for {save_category_name}: {e}")
            return saved_files

        # If there ARE result_objects with bounding boxes (future AutoKeras object detection model)
        if not os.path.exists(detections_save_dir):
            try:
                os.makedirs(detections_save_dir, exist_ok=True)
            except Exception as e:
                print(f"Error (AutoKeras): Could not create base detections directory {detections_save_dir}: {e}")
                return saved_files

        img_h, img_w = image_np.shape[:2]

        for obj in result_objects: # These objects are from the inferencer's output
            obj_name_detected = obj.get("name", "unknown").lower() # Already unexpanded by process_frame/image if needed
            obj_confidence = obj.get("confidence", 0.0)
            bounding_box = obj.get("bounding_box") # Expects [x, y, width, height] or similar

            if not bounding_box or len(bounding_box) != 4:
                print(f"Debug (AutoKeras): Skipping object due to missing/invalid bbox: {obj_name_detected}")
                continue

            if obj_confidence < self.confidence_threshold:
                print(f"Debug (AutoKeras): Skipping object {obj_name_detected} due to low confidence: {obj_confidence}")
                continue

            # The obj_name_detected should already be unexpanded if self.unexpand_detections was true
            # when process_frame/process_image was called.
            # So, save_category_name is directly obj_name_detected.
            save_category_name = obj_name_detected 
            
            # Assuming bounding_box is [x_min, y_min, width, height]
            x, y, w, h = [int(v) for v in bounding_box]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            if x2 > x1 and y2 > y1:
                cropped_image = image_np[y1:y2, x1:x2]
                if cropped_image.size == 0:
                    print(f"Warning (AutoKeras): Cropped image for {obj_name_detected} is empty. Skipping.")
                    continue

                category_save_dir = os.path.join(detections_save_dir, save_category_name)
                try:
                    os.makedirs(category_save_dir, exist_ok=True)
                    import uuid # for unique filename
                    filename = f"crop_{uuid.uuid4()}.png"
                    save_path = os.path.join(category_save_dir, filename)
                    cv2.imwrite(save_path, cropped_image)
                    print(f"Info (AutoKeras): Saved detected object '{obj_name_detected}' to {save_path}")
                    saved_files.append(save_path)
                except Exception as e:
                    print(f"Error (AutoKeras): Failed to save cropped image to {save_path}: {e}")
            else:
                print(f"Warning (AutoKeras): Invalid bbox for {obj_name_detected} after clipping. Skipping.")
        return saved_files

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
    dummy_classes = ['object_a', 'object_b', 'object_c']
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
        test_image_path = os.path.join(base_data_dir, 'object_a', 'dummy_object_a_0.jpg')
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
            test_image_path_sample = os.path.join(base_data_dir, 'object_a', 'dummy_object_a_0.jpg')
            if os.path.exists(test_image_path_sample):
                print(f"\nTesting process_image with {test_image_path_sample} using loaded model:")
                result = processor_load.process_image(test_image_path_sample)
                print(f"Result: {result}")
            else:
                print(f"\nSkipping process_image test, dummy image not found at {test_image_path_sample}")
        else:
            print("Failed to initialize processor with existing model, or model was not deemed ready.")
    else:
        print("\nSkipping test for loading existing model as prior training may have failed.")
    
    print("\nAutoKerasCVProcessor test finished.")
