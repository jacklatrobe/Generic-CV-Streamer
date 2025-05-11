"""
Handles inference using Azure Computer Vision API.
"""
import os
import io
import json
import cv2 # For frame processing
import numpy as np
import logging # Added
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__) # Added

class AzureCVInferencer:
    """
    Manages performing inference using Azure Computer Vision API.
    """
    def __init__(self, credentials_path):
        """
        Initializes the AzureCVInferencer.

        Args:
            credentials_path (str): Path to the Azure CV credentials JSON file 
                                    (containing 'api_key' and 'endpoint').
        """
        self.credentials_path = credentials_path
        self.class_names = []
        self.client = None
        self.api_ready = False
        self.confidence_threshold = 0.7  # Default confidence threshold

        self._load_config()  # Load class_names from general config.json
        self._initialize_client()

    def _load_config(self):
        """
        Loads class_names from config.json located at the project root.
        Also loads confidence_threshold if specified.
        """
        # Corrected project_root calculation
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(project_root, "config.json")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.class_names = config.get("class_names", [])
                self.confidence_threshold = float(config.get("azure_confidence_threshold", self.confidence_threshold))
                if not self.class_names:
                    logger.warning(f"'class_names' not found or empty in {config_path}. Detections might not be categorized correctly.")
                else:
                    logger.info(f"Loaded class_names: {self.class_names} from {config_path}")
                logger.info(f"Azure confidence threshold set to: {self.confidence_threshold}")
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found. Please create it with 'class_names' list. Using default class_names and threshold.")
        except json.JSONDecodeError:
            logger.error(f"Could not decode {config_path}. Please ensure it is valid JSON. Using default class_names and threshold.")
        except ValueError:
            logger.error(f"Invalid 'azure_confidence_threshold' in {config_path}. Using default {self.confidence_threshold}.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while loading config: {e}. Using default class_names and threshold.")

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
            visual_features=[VisualFeatures.OBJECTS]
        )

        processed_objects = []
        relevant_tags = []
        highest_confidence_for_relevant_tags = 0.0

        if result.objects is not None:
            for detected_object in result.objects.list:
                obj_name = "unknown"
                obj_confidence = 0.0
                if detected_object.tags and detected_object.tags[0].name:
                    obj_name = detected_object.tags[0].name.lower() # Use first tag name as object name
                    obj_confidence = detected_object.tags[0].confidence
                
                # Bounding box: [x, y, w, h] - Corrected attributes
                bounding_box = [
                    detected_object.bounding_box.x,
                    detected_object.bounding_box.y,
                    detected_object.bounding_box.width, # Corrected: was .w
                    detected_object.bounding_box.height # Corrected: was .h
                ]
                
                processed_objects.append({
                    "name": obj_name,
                    "confidence": obj_confidence,
                    "bounding_box": bounding_box 
                })

                # Check if this object's name matches any of our target class_names
                # and if its confidence meets the threshold
                if obj_confidence >= self.confidence_threshold:
                    for target_class in self.class_names:
                        if target_class.lower() in obj_name: # Simple substring match
                            relevant_tags.append(target_class)
                            if obj_confidence > highest_confidence_for_relevant_tags:
                                highest_confidence_for_relevant_tags = obj_confidence
                            break # Found a match for this object

        if not relevant_tags and processed_objects: # No relevant tags, but objects were detected
            # Find the object with the highest confidence among all detected objects
            # if no specific class_names were matched above threshold
            # This provides a fallback if no target classes are found but other objects are.
            # However, the primary goal is to find 'relevant_tags'.
            # If class_names is empty, all detected objects above threshold could be considered relevant.
            if not self.class_names: # If no class_names defined, consider all detected objects above threshold
                for obj in processed_objects:
                    if obj["confidence"] >= self.confidence_threshold:
                        relevant_tags.append(obj["name"]) # Use object's own name
                        if obj["confidence"] > highest_confidence_for_relevant_tags:
                             highest_confidence_for_relevant_tags = obj["confidence"]
            
            if not relevant_tags: # Still no relevant tags
                 return {"tags": ["no_confident_match_azure"], "confidence": 0.0, "objects": processed_objects}


        if not relevant_tags: # No objects detected or none met criteria
            return {"tags": ["no_labels_from_azure"], "confidence": 0.0, "objects": processed_objects}

        # Deduplicate relevant_tags while preserving order (though order might not be critical here)
        # For now, simple list is fine, can be refined if multiple objects map to same class_name
        
        return {
            "tags": list(set(relevant_tags)), # Unique tags
            "confidence": highest_confidence_for_relevant_tags,
            "objects": processed_objects,
            "image_height": result.metadata.height if result.metadata else None,
            "image_width": result.metadata.width if result.metadata else None,
            "model_version": result.model_version
        }

    # Placeholder for save_processed_frame, similar to GoogleCVInferencer
    # This would typically be called by the main processing loop if a frame/image
    # is classified with high confidence for a target class.
    # For now, it's not implemented as its logic depends on how detections are saved.
    def save_processed_frame(self, frame_data: np.ndarray, result: dict, frame_path: str = None):
        """
        Saves a processed frame if it meets criteria (e.g., high confidence detection of a target class).
        This is a placeholder and needs to be adapted based on project's saving strategy.
        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array.
            result (dict): The classification result from process_frame or process_image.
            frame_path (str, optional): Original path of the frame/image if available.
        Returns:
            str: Path to the saved detection file, or None.
        """
        logger.info(f"save_processed_frame called with result: {result}. Functionality not fully implemented for Azure yet.")
        # Example logic (needs refinement based on actual requirements for saving):
        # if result and result.get("tags") and self.class_names:
        #     if any(tag in self.class_names for tag in result["tags"]) and result.get("confidence", 0) >= self.confidence_threshold:
        #         # ... implement saving logic similar to GoogleCVInferencer._save_detection ...
        #         logger.info(f"Placeholder: Would save frame for tags {result['tags']}")
        #         return "path/to/saved/azure_detection.jpg" 
        return None

