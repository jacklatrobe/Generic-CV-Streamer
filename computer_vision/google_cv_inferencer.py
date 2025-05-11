# filepath: c:\\Users\\jlatrobe\\BoatRampTagger\\computer_vision\\google_cv_inferencer.py
"""
Handles inference using Google Cloud Vision API.
"""
import os
import io
import shutil
import json  # Added import for json
from datetime import datetime
from google.cloud import vision
import numpy as np
import cv2
import uuid # Added for unique filenames

class GoogleCVInferencer:
    """
    Manages performing inference using Google Cloud Vision API.
    """
    def __init__(self, credentials_path, confidence_threshold=0.7): # Added confidence_threshold
        """
        Initializes the GoogleCVInferencer.

        Args:
            credentials_path (str): Path to the Google Cloud credentials JSON file.
            confidence_threshold (float): Minimum confidence for a detection to be considered.
        """
        self.credentials_path = credentials_path
        self.client = None
        self.model_loaded = False
        self.confidence_threshold = confidence_threshold # Set from parameter
        self.detections_save_dir = "detections" # Default save directory
        self.use_object_localization = True # Assuming this should be True by default to get bounding boxes

        self._load_config_class_names_and_save_dir() # Renamed and modified
        self._initialize_client()

    def _load_config_class_names_and_save_dir(self): # Renamed and modified
        """
        Loads class_names and detections_save_dir from config.json.
        Confidence threshold is now passed via __init__.
        """
        # Assuming the script is in BoatRampTagger/computer_vision/
        # Project root is two levels up from this file's directory.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Corrected path
        config_path = os.path.join(project_root, "config.json")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.class_names = config.get("class_names", [])
                # self.confidence_threshold = float(config.get("google_confidence_threshold", self.confidence_threshold)) # Removed
                self.detections_save_dir = config.get("detections_save_dir", self.detections_save_dir)
                
                if not self.class_names:
                    print(f"Warning: 'class_names' not found or empty in {config_path}. Detections will be saved under their detected names.")
                else:
                    print(f"Loaded class_names: {self.class_names} from {config_path}")
                print(f"Google CV confidence threshold set to: {self.confidence_threshold}")
                print(f"Google CV detections will be saved to: {self.detections_save_dir}")

        except FileNotFoundError:
            print(f"Error: {config_path} not found. Using default settings.")
            self.class_names = [] 
        except json.JSONDecodeError:
            print(f"Error: Could not decode {config_path}. Using default settings.")
            self.class_names = []
        except ValueError:
            print(f"Error: Invalid 'google_confidence_threshold' in {config_path}. Using default {self.confidence_threshold}.")
        except Exception as e:
            print(f"An unexpected error occurred while loading config: {e}. Using default settings.")
            self.class_names = []
        self.class_names = [name.lower() for name in self.class_names]  # Normalize class names to lowercase

    def _initialize_client(self):
        """
        Initializes the Google Cloud Vision client.
        Sets `self.api_ready` to True if successful, False otherwise.
        """
        try:
            if self.credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
                self.client = vision.ImageAnnotatorClient()
                print("Google Cloud Vision client initialized using provided credentials.")
            else:
                self.client = vision.ImageAnnotatorClient()
                print("Google Cloud Vision client initialized using default credentials.")
            self.api_ready = True
        except Exception as e:
            print(f"Failed to initialize Google Cloud Vision client: {e}. "
                  "Ensure 'google-cloud-vision' is installed and authentication is configured.")
            self.api_ready = False

    def process_image(self, image_path: str):
        """
        Processes an image from a file path using Google Cloud Vision API for object localization.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing 'tags' (list of relevant class names based on detected objects),
                  'confidence' (highest confidence among relevant tags),
                  and 'objects' (list of all detected objects with details: name, confidence, bounding_box).
                  Returns an error structure if processing fails.
        """
        if not self.api_ready or not self.client:
            print("Google CV API not available for processing.")
            return {"tags": ["error_api_not_ready"], "confidence": 0.0, "objects": []}
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return {"tags": ["error_image_not_found"], "confidence": 0.0, "objects": []}

        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            # Re-read image with OpenCV to get dimensions for BBox conversion later if needed by save_processed_frame
            # Or, ensure localize_objects can pass image dimensions if it reads it.
            # For now, assuming main.py will pass the frame_data to save_processed_frame.

            localization_result = self.localize_objects(image_content=content)
            
            # Process localization_result to fit the expected output structure
            # The 'objects' list from localize_objects is already good.
            # We need to derive 'tags' and 'confidence' based on class_names and detected objects.

            detected_objects = localization_result.get("objects", [])
            relevant_tags = []
            highest_confidence_for_relevant_tags = 0.0

            if detected_objects:
                for obj in detected_objects:
                    obj_name = obj.get("name", "unknown").lower()
                    obj_confidence = obj.get("confidence", 0.0)

                    if obj_confidence >= self.confidence_threshold:
                        is_relevant_to_class_names = False
                        tag_to_add = obj_name # Default tag is the detected object name

                        if not self.class_names: # If no class_names, any confident object is relevant
                            is_relevant_to_class_names = True
                        else:
                            for target_class in self.class_names:
                                if target_class.lower() in obj_name:
                                    is_relevant_to_class_names = True
                                    tag_to_add = target_class.lower() # Use the matched class_name for the tag
                                    break
                        
                        if is_relevant_to_class_names:
                            relevant_tags.append(tag_to_add)
                            if obj_confidence > highest_confidence_for_relevant_tags:
                                highest_confidence_for_relevant_tags = obj_confidence
            
            if not relevant_tags and detected_objects:
                 # Fallback if no class_names matched but objects were found
                if not self.class_names: # If class_names is empty, all confident objects are "relevant"
                    for obj in detected_objects:
                        if obj["confidence"] >= self.confidence_threshold:
                            relevant_tags.append(obj["name"].lower())
                            if obj["confidence"] > highest_confidence_for_relevant_tags:
                                highest_confidence_for_relevant_tags = obj["confidence"]
                
                if not relevant_tags: # Still no relevant tags for summary
                    return {"tags": ["no_confident_match_google"], "confidence": 0.0, "objects": detected_objects}


            if not relevant_tags:
                return {"tags": ["no_labels_from_google"], "confidence": 0.0, "objects": detected_objects}

            return {
                "tags": list(set(relevant_tags)), 
                "confidence": highest_confidence_for_relevant_tags,
                "objects": detected_objects 
            }

        except Exception as e:
            print(f"Error processing image {image_path} with Google CV: {e}")
            return {"tags": ["error_processing_google_cv"], "confidence": 0.0, "objects": []}

    def process_frame(self, frame_data: np.ndarray):
        """
        Processes a raw image frame using Google Cloud Vision API for object localization.

        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array (BGR format from OpenCV).

        Returns:
            dict: A dictionary containing 'tags', 'confidence', and 'objects'.
        """
        if not self.api_ready or not self.client:
            print("Google CV API not available for processing.")
            return {"tags": ["error_api_not_ready"], "confidence": 0.0, "objects": []}

        try:
            # Encode frame to pass to localize_objects
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            is_success, buffer = cv2.imencode(".jpg", rgb_frame)
            if not is_success:
                raise ValueError("Failed to encode frame to JPEG")
            content = buffer.tobytes()

            localization_result = self.localize_objects(image_content=content)
            
            # Process localization_result similar to process_image
            detected_objects = localization_result.get("objects", [])
            relevant_tags = []
            highest_confidence_for_relevant_tags = 0.0

            if detected_objects:
                for obj in detected_objects:
                    obj_name = obj.get("name", "unknown").lower()
                    obj_confidence = obj.get("confidence", 0.0)

                    if obj_confidence >= self.confidence_threshold:
                        is_relevant_to_class_names = False
                        tag_to_add = obj_name 

                        if not self.class_names: 
                            is_relevant_to_class_names = True
                        else:
                            for target_class in self.class_names:
                                if target_class.lower() in obj_name:
                                    is_relevant_to_class_names = True
                                    tag_to_add = target_class.lower()
                                    break
                        
                        if is_relevant_to_class_names:
                            relevant_tags.append(tag_to_add)
                            if obj_confidence > highest_confidence_for_relevant_tags:
                                highest_confidence_for_relevant_tags = obj_confidence
            
            if not relevant_tags and detected_objects:
                if not self.class_names:
                    for obj in detected_objects:
                        if obj["confidence"] >= self.confidence_threshold:
                            relevant_tags.append(obj["name"].lower())
                            if obj["confidence"] > highest_confidence_for_relevant_tags:
                                highest_confidence_for_relevant_tags = obj["confidence"]
                if not relevant_tags:
                    return {"tags": ["no_confident_match_google"], "confidence": 0.0, "objects": detected_objects}

            if not relevant_tags:
                return {"tags": ["no_labels_from_google"], "confidence": 0.0, "objects": detected_objects}

            return {
                "tags": list(set(relevant_tags)), 
                "confidence": highest_confidence_for_relevant_tags,
                "objects": detected_objects
            }

        except Exception as e:
            print(f"Error processing frame with Google CV: {e}")
            return {"tags": ["error_processing_google_cv_frame"], "confidence": 0.0, "objects": []}

    def save_processed_frame(self, image_np: np.ndarray, result_objects: list):
        """
        Saves cropped images of detected objects that meet criteria.

        Args:
            image_np (np.ndarray): The image data as a NumPy array (BGR format from OpenCV).
            result_objects (list): A list of detected object dictionaries from localize_objects.
                                   Each dict should have 'name', 'confidence', 'bounding_box' (normalized vertices).
        Returns:
            list: A list of paths to the saved detection files.
        """
        saved_files = []
        if not self.detections_save_dir:
            print("Warning: Detections save directory ('detections_save_dir') is not configured for GoogleCV. Cannot save detected objects.")
            return saved_files
        
        if not os.path.exists(self.detections_save_dir):
            try:
                os.makedirs(self.detections_save_dir, exist_ok=True)
            except Exception as e:
                print(f"Error: Could not create base detections directory {self.detections_save_dir}: {e}")
                return saved_files

        img_h, img_w = image_np.shape[:2]

        for obj in result_objects:
            obj_name_detected = obj.get("name", "unknown").lower()
            obj_confidence = obj.get("confidence", 0.0)
            # Google's bounding_box is object_.bounding_poly.normalized_vertices
            # It's a list of 4 (x,y) tuples. For a rectangle, typically:
            # vertices[0] = top-left, vertices[1] = top-right, 
            # vertices[2] = bottom-right, vertices[3] = bottom-left
            normalized_vertices = obj.get("bounding_box") 

            if not normalized_vertices or len(normalized_vertices) != 4:
                print(f"Debug: Skipping object due to missing or invalid bounding box: {obj_name_detected}")
                continue

            if obj_confidence < self.confidence_threshold:
                print(f"Debug: Skipping object {obj_name_detected} due to low confidence: {obj_confidence} < {self.confidence_threshold}")
                continue

            qualifies_for_saving = False
            save_category_name = obj_name_detected 

            if not self.class_names: 
                qualifies_for_saving = True
            else:
                for target_class in self.class_names:
                    if target_class.lower() in obj_name_detected:
                        qualifies_for_saving = True
                        save_category_name = target_class.lower() 
                        break
            
            if qualifies_for_saving:
                # Convert normalized vertices to pixel coordinates for cropping
                # Assuming a rectangular bounding box from the normalized vertices
                # x_coords = [v[0] for v in normalized_vertices]
                # y_coords = [v[1] for v in normalized_vertices]
                # x_min_norm, y_min_norm = min(x_coords), min(y_coords)
                # x_max_norm, y_max_norm = max(x_coords), max(y_coords)
                
                # More directly, for a typical response:
                x_min_norm = normalized_vertices[0][0]
                y_min_norm = normalized_vertices[0][1]
                x_max_norm = normalized_vertices[2][0]
                y_max_norm = normalized_vertices[2][1]

                x1 = int(x_min_norm * img_w)
                y1 = int(y_min_norm * img_h)
                x2 = int(x_max_norm * img_w)
                y2 = int(y_max_norm * img_h)
                
                # Ensure coordinates are within image bounds and valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)

                if x2 > x1 and y2 > y1: 
                    cropped_image = image_np[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        print(f"Warning: Cropped image for {obj_name_detected} is empty. Original box (norm): {normalized_vertices}, clipped (px): {[x1,y1,x2,y2]}. Skipping save.")
                        continue

                    category_save_dir = os.path.join(self.detections_save_dir, save_category_name)
                    try:
                        os.makedirs(category_save_dir, exist_ok=True)
                    except Exception as e:
                        print(f"Error: Could not create category directory {category_save_dir}: {e}")
                        continue 

                    unique_id = uuid.uuid4()
                    filename = f"{unique_id}.png" # Save as PNG
                    save_path = os.path.join(category_save_dir, filename)

                    try:
                        cv2.imwrite(save_path, cropped_image)
                        print(f"Info: Saved detected object '{obj_name_detected}' (category: {save_category_name}) to {save_path}")
                        saved_files.append(save_path)
                    except Exception as e:
                        print(f"Error: Failed to save cropped image to {save_path}: {e}")
                else:
                    print(f"Warning: Invalid bounding box for {obj_name_detected} after clipping (px): {[x1,y1,x2,y2]} from normalized {normalized_vertices}. Skipping save.")
            else:
                print(f"Debug: Object {obj_name_detected} did not qualify for saving based on class_names filter.")
                
        return saved_files

    def localize_objects(self, image_path=None, image_content=None, frame_data=None):
        """
        Detects and localizes objects in an image using Google Cloud Vision API.
        
        Args:
            image_path (str, optional): Path to the image file.
            image_content (bytes, optional): Raw image content as bytes.
            frame_data (np.ndarray, optional): Frame data as a NumPy array.
            
        Returns:
            dict: A dictionary containing detected objects with bounding boxes and confidences.
        """
        if not self.api_ready or not self.client:
            print("Google CV API not available for processing.")
            return {"objects": [], "error": "API not ready"}
            
        try:
            # Prepare the image for the API
            image = None
            if image_path:
                if not os.path.exists(image_path):
                    print(f"Image path does not exist: {image_path}")
                    return {"objects": [], "error": "Image not found"}
                with io.open(image_path, 'rb') as image_file:
                    content = image_file.read()
                image = vision.Image(content=content)
            elif image_content:
                image = vision.Image(content=image_content)
            elif frame_data is not None:
                rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                is_success, buffer = cv2.imencode(".jpg", rgb_frame)
                if not is_success:
                    raise ValueError("Failed to encode frame to JPEG")
                content = buffer.tobytes()
                image = vision.Image(content=content)
            else:
                return {"objects": [], "error": "No image data provided"}
                
            # Perform object localization
            # For object_localization, the bounding box is in object_.bounding_poly.normalized_vertices
            response = self.client.object_localization(image=image)
            localized_objects_annotations = response.localized_object_annotations # Renamed for clarity
            
            if response.error.message:
                raise Exception(f"Google CV API Error: {response.error.message}")
                
            # Process the results
            objects_detected = [] # This will store our processed list of objects
            # ... (removed detected_target_classes, highest_confidence, best_object_for_saving as they are re-derived in process_image/frame)
            
            # print(f"Number of objects detected by API: {len(localized_objects_annotations)}") # Debug
            for object_annotation in localized_objects_annotations: # Iterate through API response
                object_name = object_annotation.name # Keep original case for display if needed, but compare lowercase
                confidence = object_annotation.score
                
                # Extract normalized vertices for the bounding_box
                normalized_vertices = []
                for vertex in object_annotation.bounding_poly.normalized_vertices:
                    normalized_vertices.append((vertex.x, vertex.y))
                
                # print(f"  - API Object: {object_name}, Score: {confidence:.4f}, Bounds: {normalized_vertices}") # Debug
                
                # Add object to our results list
                objects_detected.append({
                    "name": object_name, # Storing the name as detected
                    "confidence": confidence,
                    "bounding_box": normalized_vertices, # This is a list of (x,y) tuples
                    # "matched_class": matched_class # This logic is now in process_image/frame
                })
            
            # The result from localize_objects is now just the list of detected objects.
            # Filtering by class_names and confidence, and determining "best" tags,
            # will be handled by the calling methods (process_image, process_frame).
            return {
                "objects": objects_detected,
                "error": None # Explicitly state no error if successful
            }
            
        except Exception as e:
            print(f"Error in object localization: {e}")
            return {"objects": [], "error": str(e)}