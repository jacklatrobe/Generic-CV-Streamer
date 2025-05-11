"""
Handles inference using Azure Computer Vision API.
"""
import os
import io
import json
import cv2 # For frame processing
import numpy as np
import logging # Added
import uuid # Added for unique filenames
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__) # Added

class AzureCVInferencer:
    """
    Manages performing inference using Azure Computer Vision API.
    """
    def __init__(self, credentials_path, confidence_threshold=0.5, class_names_override=None): # Added class_names_override
        """
        Initializes the AzureCVInferencer.

        Args:
            credentials_path (str): Path to the Azure CV credentials JSON file.
            confidence_threshold (float): Minimum confidence for a detection to be considered.
            class_names_override (list, optional): A list of class names to use instead of from config.json.
        """
        self.credentials_path = credentials_path
        self.client = None
        self.api_ready = False
        self.confidence_threshold = confidence_threshold
        self.detections_save_dir = "detections"
        self.class_names = []

        self._load_config_and_initialize_client(class_names_override) # Pass override

    def _load_config_and_initialize_client(self, class_names_override=None): # Added class_names_override
        """
        Loads configuration (class_names, detections_save_dir) and initializes the Azure CV client.
        Confidence threshold is now passed via __init__.
        Class names can be overridden.
        """
        # Corrected project_root calculation
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(project_root, "config.json")
        
        if class_names_override is not None:
            self.class_names = [name.lower() for name in class_names_override if isinstance(name, str)]
            logger.info(f"Using overridden class_names: {self.class_names}")
        else:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    loaded_class_names = config.get("class_names", [])
                    if isinstance(loaded_class_names, list):
                        self.class_names = [name.lower() for name in loaded_class_names if isinstance(name, str)]
                    else:
                        logger.warning(f"'class_names' in {config_path} is not a list. Using empty list.")
                        self.class_names = []
                    
                    if not self.class_names:
                        logger.warning(f"'class_names' not found, empty, or invalid in {config_path}. Detections might not be categorized correctly for filtering, but will save under detected name.")
                    else:
                        logger.info(f"Loaded class_names: {self.class_names} from {config_path}")
            except FileNotFoundError:
                logger.error(f"Configuration file {config_path} not found. Using empty list for class_names as no override was provided.")
                self.class_names = [] # Ensure it's an empty list
            except json.JSONDecodeError:
                logger.error(f"Could not decode {config_path}. Using empty list for class_names as no override was provided.")
                self.class_names = [] # Ensure it's an empty list
            except Exception as e:
                logger.exception(f"An unexpected error occurred while loading class_names from config: {e}. Using empty list.")
                self.class_names = [] # Ensure it's an empty list

        # Load other configs like detections_save_dir, which are not overridden by this new param
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.detections_save_dir = config.get("detections_save_dir", self.detections_save_dir)
        except FileNotFoundError:
            logger.info(f"Configuration file {config_path} not found. Using default detections_save_dir: {self.detections_save_dir}")
        except json.JSONDecodeError:
            logger.error(f"Could not decode {config_path} for other settings. Using default detections_save_dir: {self.detections_save_dir}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while loading other config settings: {e}")

        logger.info(f"Azure confidence threshold set to: {self.confidence_threshold}")
        logger.info(f"Detections will be saved to: {self.detections_save_dir}")

        self._initialize_client()

    def _initialize_client(self):
        """
        Initializes the Azure Computer Vision client using credentials from the JSON file.
        Sets `self.api_ready` to True if successful, False otherwise.
        """
        try:
            if not os.path.exists(self.credentials_path):
                logger.error(f"Azure credentials file not found at {self.credentials_path}")
                self.api_ready = False
                return

            with open(self.credentials_path, 'r') as f:
                credentials = json.load(f)
            
            endpoint = credentials.get("endpoint")
            key = credentials.get("api_key")

            if not endpoint or not key:
                logger.error("'endpoint' or 'api_key' missing in Azure credentials file.")
                self.api_ready = False
                return

            self.client = ImageAnalysisClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key)
            )
            logger.info("Azure Computer Vision client initialized successfully.")
            self.api_ready = True
        except json.JSONDecodeError:
            logger.error(f"Could not decode Azure credentials file at {self.credentials_path}. Ensure it is valid JSON.")
            self.api_ready = False
        except Exception as e:
            logger.exception(f"Failed to initialize Azure Computer Vision client: {e}. Ensure 'azure-ai-vision-imageanalysis' is installed and credentials are correct.")
            self.api_ready = False

    def process_image(self, image_path: str):
        """
        Processes an image from a file path using Azure CV API for object detection.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing 'tags' (list of relevant class names based on detected objects),
                  'confidence' (highest confidence among relevant tags),
                  and 'objects' (list of all detected objects with details: name, confidence, bounding_box).
                  Returns an error structure if processing fails.
        """
        if not self.api_ready or not self.client:
            logger.warning("Azure CV API not available for processing.")
            return {"tags": ["error_api_not_ready"], "confidence": 0.0, "objects": []}
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return {"tags": ["error_image_not_found"], "confidence": 0.0, "objects": []}

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            return self._analyze_image_data(image_data)
            
        except Exception as e:
            logger.exception(f"Error processing image {image_path} with Azure CV: {e}")
            return {"tags": ["error_processing_azure_cv"], "confidence": 0.0, "objects": []}

    def process_frame(self, frame_data: np.ndarray):
        """
        Processes a raw image frame (NumPy array) using Azure CV API for object detection.

        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array (BGR format from OpenCV).

        Returns:
            dict: A dictionary containing 'tags', 'confidence', and 'objects'.
        """
        if not self.api_ready or not self.client:
            logger.warning("Azure CV API not available for processing.")
            return {"tags": ["error_api_not_ready"], "confidence": 0.0, "objects": []}

        try:
            is_success, buffer = cv2.imencode(".jpg", frame_data)
            if not is_success:
                logger.error("Failed to encode frame to JPEG for Azure CV")
                raise ValueError("Failed to encode frame to JPEG for Azure CV")
            image_data = buffer.tobytes()

            return self._analyze_image_data(image_data)

        except Exception as e:
            logger.exception(f"Error processing frame with Azure CV: {e}")
            return {"tags": ["error_processing_azure_cv_frame"], "confidence": 0.0, "objects": []}

    def _analyze_image_data(self, image_data: bytes):
        """
        Helper function to analyze image data (bytes) using Azure CV API.
        """
        result = self.client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.OBJECTS, VisualFeatures.TAGS] # Added TAGS for broader context if needed, OBJECTS is primary
        )
        logger.info(f"Raw Azure CV response: {result}") # Added line to log raw response

        processed_objects = []
        relevant_tags = [] # Tags based on class_names and confidence
        highest_confidence_for_relevant_tags = 0.0

        if result.objects is not None:
            for detected_object in result.objects.list:
                obj_name = "unknown"
                obj_confidence = 0.0
                # Azure SDK for Image Analysis typically gives object names in detected_object.tags[0].name
                # and its confidence in detected_object.tags[0].confidence
                if detected_object.tags and len(detected_object.tags) > 0:
                    obj_name = detected_object.tags[0].name.lower() 
                    obj_confidence = detected_object.tags[0].confidence
                
                bounding_box = [
                    detected_object.bounding_box.x,
                    detected_object.bounding_box.y,
                    detected_object.bounding_box.width, 
                    detected_object.bounding_box.height
                ]
                
                current_obj_details = {
                    "name": obj_name,
                    "confidence": obj_confidence,
                    "bounding_box": bounding_box 
                }
                processed_objects.append(current_obj_details)

                # Check if this object's name matches any of our target class_names
                # and if its confidence meets the threshold for "relevant_tags" summary
                if obj_confidence >= self.confidence_threshold:
                    is_relevant_to_class_names = False
                    if not self.class_names: # If no class_names, consider any confident object relevant
                        is_relevant_to_class_names = True
                    else:
                        for target_class in self.class_names:
                            if target_class.lower() in obj_name: 
                                is_relevant_to_class_names = True
                                break
                    
                    if is_relevant_to_class_names:
                        # Use target_class for relevant_tags if matched, else obj_name
                        tag_to_add = obj_name
                        if self.class_names:
                            for target_class in self.class_names:
                                if target_class.lower() in obj_name:
                                    tag_to_add = target_class.lower()
                                    break
                        relevant_tags.append(tag_to_add)
                        if obj_confidence > highest_confidence_for_relevant_tags:
                            highest_confidence_for_relevant_tags = obj_confidence
        
        # Fallback for tags if no relevant objects found but general tags exist (from VisualFeatures.TAGS)
        # This part is more about image-level tags, not specific objects for cropping.
        # For object saving, processed_objects is the key.
        # The 'tags' in the final result dict is a summary for the image.

        if not relevant_tags and processed_objects:
            # If class_names is empty, all detected objects above threshold could be considered relevant for summary
            if not self.class_names:
                for obj in processed_objects:
                    if obj["confidence"] >= self.confidence_threshold:
                        relevant_tags.append(obj["name"])
                        if obj["confidence"] > highest_confidence_for_relevant_tags:
                             highest_confidence_for_relevant_tags = obj["confidence"]
            
            if not relevant_tags: # Still no relevant tags for summary
                 return {"tags": ["no_confident_match_azure"], "confidence": 0.0, "objects": processed_objects, "image_height": result.metadata.height if result.metadata else None, "image_width": result.metadata.width if result.metadata else None, "model_version": result.model_version}

        if not relevant_tags and not processed_objects: # No objects detected or none met criteria
            return {"tags": ["no_labels_from_azure"], "confidence": 0.0, "objects": processed_objects, "image_height": result.metadata.height if result.metadata else None, "image_width": result.metadata.width if result.metadata else None, "model_version": result.model_version}
        
        return {
            "tags": list(set(relevant_tags)), 
            "confidence": highest_confidence_for_relevant_tags,
            "objects": processed_objects,
            "image_height": result.metadata.height if result.metadata else None,
            "image_width": result.metadata.width if result.metadata else None,
            "model_version": result.model_version
        }

    def save_processed_frame(self, image_np: np.ndarray, result_objects: list):
        """
        Saves cropped images of detected objects that meet criteria.

        Args:
            image_np (np.ndarray): The image data as a NumPy array (BGR format from OpenCV).
            result_objects (list): A list of detected object dictionaries from the _analyze_image_data method.
                                   Each dict should have 'name', 'confidence', 'bounding_box'.
        Returns:
            list: A list of paths to the saved detection files.
        """
        saved_files = []
        if not self.detections_save_dir:
            logger.warning("Detections save directory ('detections_save_dir') is not configured. Cannot save detected objects.")
            return saved_files
        
        if not os.path.exists(self.detections_save_dir):
            try:
                os.makedirs(self.detections_save_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Could not create base detections directory {self.detections_save_dir}: {e}")
                return saved_files

        img_h, img_w = image_np.shape[:2]

        for obj in result_objects:
            obj_name_detected = obj.get("name", "unknown").lower()
            obj_confidence = obj.get("confidence", 0.0)
            bounding_box = obj.get("bounding_box")

            if not bounding_box or len(bounding_box) != 4:
                logger.debug(f"Skipping object due to missing or invalid bounding box: {obj_name_detected}")
                continue

            if obj_confidence < self.confidence_threshold:
                logger.debug(f"Skipping object {obj_name_detected} due to low confidence: {obj_confidence} < {self.confidence_threshold}")
                continue

            # Determine if the object matches one of the target class_names (if any are specified)
            # And determine the folder name for saving (either the matched class_name or the specific detected name)
            qualifies_for_saving = False
            save_category_name = obj_name_detected # Default to the specific name Azure detected

            if not self.class_names: # If no class_names in config, save any confident detection under its own name
                qualifies_for_saving = True
            else:
                for target_class in self.class_names:
                    if target_class.lower() in obj_name_detected:
                        qualifies_for_saving = True
                        save_category_name = target_class.lower() # Save under the broader category from config
                        break
            
            if qualifies_for_saving:
                x, y, w, h = [int(v) for v in bounding_box]

                # Ensure coordinates are within image bounds for cropping
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img_w, x + w), min(img_h, y + h)

                if x2 > x1 and y2 > y1: # Check if the crop area is valid
                    cropped_image = image_np[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        logger.warning(f"Cropped image for {obj_name_detected} is empty. Original box: {[x,y,w,h]}, clipped: {[x1,y1,x2,y2]}. Skipping save.")
                        continue

                    category_save_dir = os.path.join(self.detections_save_dir, save_category_name)
                    try:
                        os.makedirs(category_save_dir, exist_ok=True)
                    except Exception as e:
                        logger.error(f"Could not create category directory {category_save_dir}: {e}")
                        continue # Skip saving this object if its directory can't be made

                    unique_id = uuid.uuid4()
                    filename = f"{unique_id}.png"
                    save_path = os.path.join(category_save_dir, filename)

                    try:
                        cv2.imwrite(save_path, cropped_image)
                        logger.info(f"Saved detected object '{obj_name_detected}' (category: {save_category_name}) to {save_path}")
                        saved_files.append(save_path)
                    except Exception as e:
                        logger.error(f"Failed to save cropped image to {save_path}: {e}")
                else:
                    logger.warning(f"Invalid bounding box for {obj_name_detected} after clipping: {[x1,y1,x2,y2]} from original {[x,y,w,h]}. Skipping save.")
            else:
                logger.debug(f"Object {obj_name_detected} did not qualify for saving based on class_names filter.")
                
        return saved_files

