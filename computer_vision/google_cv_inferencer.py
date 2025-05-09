# filepath: c:\Users\jlatrobe\BoatRampTagger\computer_vision\google_cv_inferencer.py
"""
Handles inference using Google Cloud Vision API.
"""
import os
import io
import shutil
from datetime import datetime
from google.cloud import vision
import numpy as np
import cv2

class GoogleCVInferencer:
    """
    Manages using Google Cloud Vision API for image labeling and object localization.
    """
    def __init__(self, class_names, credentials_path=None):
        """
        Initializes the GoogleCVInferencer.

        Args:
            class_names (list): A list of strings representing the class names
                                that the application is interested in.
                                Google Vision API might return more labels.
            credentials_path (str, optional): Path to Google Cloud service account JSON file.
                                            If None, attempts to use default credentials.
        """
        self.class_names = class_names # These are the target classes we are interested in
        self.client = None
        self.api_ready = False
        self.confidence_threshold = 0.7  # 70% confidence threshold for a label to be considered
        self.use_object_localization = True  # Enable object localization by default

        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        self.detections_base_dir = os.path.join(project_root, "data", "detections_google_cv")
        self._create_detection_directories()

        self._initialize_client(credentials_path)

    def _create_detection_directories(self):
        """
        Creates directories for storing detected objects based on class names.
        """
        os.makedirs(self.detections_base_dir, exist_ok=True)
        for class_name in self.class_names: # Create dirs for our target classes
            os.makedirs(os.path.join(self.detections_base_dir, class_name), exist_ok=True)
        # Removed creation of other_google_detections directory

    def _save_detection(self, image_path, predicted_label, original_image_path_or_data):
        """
        Saves a copy of the detected image to the appropriate detection directory.

        Args:
            image_path (str): Path to the original image file (if processing a file)
                              or None if processing a frame.
            predicted_label (str): The predicted class label.
            original_image_path_or_data (str or np.ndarray): Path to original image or frame data.
        Returns:
            str: Path to the saved detection file, or None if save failed.
        """
        try:
            # Only proceed if the predicted label is in our target classes
            if predicted_label not in self.class_names:
                return None
                
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            
            if image_path: # Processing an image file
                filename = os.path.basename(image_path)
                base_name, ext = os.path.splitext(filename)
            else: # Processing a frame
                base_name = "frame"
                ext = ".jpg" # Assume saving as jpg

            detection_dir = os.path.join(self.detections_base_dir, predicted_label)
            detection_path = os.path.join(detection_dir, f"{base_name}_{timestamp}{ext}")

            if isinstance(original_image_path_or_data, str): # it's a path
                shutil.copy2(original_image_path_or_data, detection_path)
            elif isinstance(original_image_path_or_data, np.ndarray): # it's frame data
                cv2.imwrite(detection_path, original_image_path_or_data)
            else:
                print("Error saving detection: Invalid image data type.")
                return None

            print(f"Saved Google CV detection to {detection_path}")
            return detection_path
        except Exception as e:
            print(f"Error saving Google CV detection: {e}")
            return None

    def _initialize_client(self, credentials_path=None):
        """
        Initializes the Google Cloud Vision client.
        Sets `self.api_ready` to True if successful, False otherwise.
        """
        try:
            if credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
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
        Processes an image from a file path using Google Cloud Vision API.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing the predicted 'tags' (list of relevant class names)
                  and 'confidence' (highest confidence among relevant tags).
                  Returns an error tag if processing fails.
        """
        if not self.api_ready or not self.client:
            print("Google CV API not available for processing.")
            return {"tags": ["error_api_not_ready"], "confidence": 0.0}
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return {"tags": ["error_image_not_found"], "confidence": 0.0}

        try:
            # If object localization is enabled, use it instead of label detection
            if self.use_object_localization:
                print(f"Using object localization for {image_path}")
                localization_result = self.localize_objects(image_path=image_path)
                
                if "error" in localization_result and localization_result["error"] and localization_result["error"] != "No target classes detected":
                    print(f"Error in object localization: {localization_result['error']}")
                    return {"tags": ["error_object_localization"], "confidence": 0.0}
                
                # Convert localization results to our standard format
                detected_tags = localization_result.get("detected_classes", [])
                highest_confidence = localization_result.get("highest_confidence", 0.0)
                best_label_for_saving = localization_result.get("best_object_for_saving")
                
                # Draw bounding boxes on a copy of the image if objects were detected
                objects = localization_result.get("objects", [])
                if objects and len(objects) > 0:
                    # Load the original image
                    img = cv2.imread(image_path)
                    img_height, img_width = img.shape[:2]
                    
                    # Draw bounding boxes for each detected object
                    for obj in objects:
                        if obj["confidence"] >= self.confidence_threshold:
                            # Convert normalized coordinates to pixel coordinates
                            box = obj["bounding_box"]
                            x_min = int(box[0][0] * img_width)
                            y_min = int(box[0][1] * img_height)
                            x_max = int(box[2][0] * img_width)
                            y_max = int(box[2][1] * img_height)
                            
                            # Choose color based on whether it's a target class
                            color = (0, 255, 0) if obj["matched_class"] else (0, 0, 255)  # Green for targets, red for others
                            
                            # Draw the box
                            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                            
                            # Add text label
                            text = f"{obj['name']} ({obj['confidence']:.2f})"
                            cv2.putText(img, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # If we detected any target classes, save the annotated image
                    if best_label_for_saving in self.class_names:
                        # Create a new filename for the annotated image
                        base_name = os.path.basename(image_path)
                        name, ext = os.path.splitext(base_name)
                        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                        detection_dir = os.path.join(self.detections_base_dir, best_label_for_saving)
                        annotated_path = os.path.join(detection_dir, f"{name}_annotated_{timestamp}{ext}")
                        
                        # Save the annotated image
                        cv2.imwrite(annotated_path, img)
                        print(f"Saved annotated image with bounding boxes to {annotated_path}")
                
                if not detected_tags:
                    if objects:
                        # Find the highest confidence non-target object
                        top_object = max(objects, key=lambda o: o["confidence"]) if objects else None
                        if top_object:
                            print(f"Processed {image_path}: No target class detected. Top object: {top_object['name']} ({top_object['confidence']:.4f})")
                            return {"tags": ["no_confident_match_google"], "confidence": top_object["confidence"]}
                    
                    print(f"Processed {image_path}: No objects detected.")
                    return {"tags": ["no_labels_from_google"], "confidence": 0.0}
                
                print(f"Processed {image_path} with object localization: Detected classes = {detected_tags}, Highest Confidence = {highest_confidence:.4f}")
                return {
                    "tags": detected_tags, 
                    "confidence": highest_confidence,
                    "objects": objects,
                    "best_label_for_saving": best_label_for_saving
                }
            
            # If object localization is not enabled, use the original label detection method
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            
            response = self.client.label_detection(image=image)
            labels = response.label_annotations

            if response.error.message:
                raise Exception(f"Google CV API Error: {response.error.message}")

            # ...existing label detection code...

        except Exception as e:
            print(f"Error processing image {image_path} with Google CV: {e}")
            return {"tags": ["error_processing_google_cv"], "confidence": 0.0}

    def process_frame(self, frame_data: np.ndarray):
        """
        Processes a raw image frame using Google Cloud Vision API.

        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array (BGR format from OpenCV).

        Returns:
            dict: A dictionary containing predicted 'tags' and 'confidence'.
        """
        if not self.api_ready or not self.client:
            print("Google CV API not available for processing.")
            return {"tags": ["error_api_not_ready"], "confidence": 0.0}

        try:
            # If object localization is enabled, use it instead of label detection
            if self.use_object_localization:
                print("Using object localization for frame")
                localization_result = self.localize_objects(frame_data=frame_data)
                
                if "error" in localization_result and localization_result["error"] and localization_result["error"] != "No target classes detected":
                    print(f"Error in object localization: {localization_result['error']}")
                    return {"tags": ["error_object_localization"], "confidence": 0.0}
                
                # Convert localization results to our standard format
                detected_tags = localization_result.get("detected_classes", [])
                highest_confidence = localization_result.get("highest_confidence", 0.0)
                best_label_for_saving = localization_result.get("best_object_for_saving")
                objects = localization_result.get("objects", [])
                
                # Draw bounding boxes on a copy of the frame if objects were detected
                annotated_frame = None
                if objects and len(objects) > 0:
                    # Make a copy of the frame
                    annotated_frame = frame_data.copy()
                    img_height, img_width = annotated_frame.shape[:2]
                    
                    # Draw bounding boxes for each detected object
                    for obj in objects:
                        if obj["confidence"] >= self.confidence_threshold:
                            # Convert normalized coordinates to pixel coordinates
                            box = obj["bounding_box"]
                            x_min = int(box[0][0] * img_width)
                            y_min = int(box[0][1] * img_height)
                            x_max = int(box[2][0] * img_width)
                            y_max = int(box[2][1] * img_height)
                            
                            # Choose color based on whether it's a target class
                            color = (0, 255, 0) if obj["matched_class"] else (0, 0, 255)  # Green for targets, red for others
                            
                            # Draw the box
                            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
                            
                            # Add text label
                            text = f"{obj['name']} ({obj['confidence']:.2f})"
                            cv2.putText(annotated_frame, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if not detected_tags:
                    if objects:
                        # Find the highest confidence non-target object
                        top_object = max(objects, key=lambda o: o["confidence"]) if objects else None
                        if top_object:
                            print(f"Processed frame: No target class detected. Top object: {top_object['name']} ({top_object['confidence']:.4f})")
                            return {"tags": ["no_confident_match_google"], "confidence": top_object["confidence"]}
                    
                    print("Processed frame: No objects detected.")
                    return {"tags": ["no_labels_from_google"], "confidence": 0.0}
                
                print(f"Processed frame with object localization: Detected classes = {detected_tags}, Highest Confidence = {highest_confidence:.4f}")
                return {
                    "tags": detected_tags, 
                    "confidence": highest_confidence,
                    "objects": objects,
                    "best_label_for_saving": best_label_for_saving,
                    "annotated_frame": annotated_frame  # Include the annotated frame for saving
                }
            
            # If object localization is not enabled, fall back to the original label detection method
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            is_success, buffer = cv2.imencode(".jpg", rgb_frame)
            if not is_success:
                raise ValueError("Failed to encode frame to JPEG")
            content = buffer.tobytes()

            image = vision.Image(content=content)
            response = self.client.label_detection(image=image)
            labels = response.label_annotations

            if response.error.message:
                raise Exception(f"Google CV API Error: {response.error.message}")

            # ...existing code...

        except Exception as e:
            print(f"Error processing frame with Google CV: {e}")
            return {"tags": ["error_processing_google_cv_frame"], "confidence": 0.0}

    def save_processed_frame(self, frame_data: np.ndarray, result: dict, frame_path: str = None):
        """
        Saves a processed frame that has been classified with high confidence by Google CV.
        
        Args:
            frame_data (np.ndarray): The raw image data as a NumPy array.
            result (dict): The classification result from process_frame.
            frame_path (str, optional): Path where the frame might have been saved originally.
                                      
        Returns:
            str: Path to the saved detection file, or None if save failed or not applicable.
        """
        if not result or "tags" not in result or not result["tags"]:
            return None
        
        if result["tags"][0].startswith("error_") or result["tags"][0] in ["no_confident_match_google", "no_labels_from_google"]:
            # Skip saving any non-target classes
            return None

        predicted_label_for_saving = result.get("best_label_for_saving")
        if not predicted_label_for_saving and result["tags"]:
             if result.get("confidence", 0) < self.confidence_threshold:
                 return None
             print("Warning: save_processed_frame called without 'best_label_for_saving' in result. Frame not saved.")
             return None

        if result.get("confidence", 0) < self.confidence_threshold:
            return None 
            
        # Only save if the label is one of our target classes
        if predicted_label_for_saving in self.class_names:
            return self._save_detection(None, predicted_label_for_saving, frame_data)
        return None

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
            response = self.client.object_localization(image=image)
            localized_objects = response.localized_object_annotations
            
            if response.error.message:
                raise Exception(f"Google CV API Error: {response.error.message}")
                
            # Process the results
            objects_detected = []
            detected_target_classes = set()
            highest_confidence = 0.0
            best_object_for_saving = None
            
            print(f"Number of objects detected: {len(localized_objects)}")
            for object_ in localized_objects:
                object_name = object_.name.lower()
                confidence = object_.score
                
                # Create the bounding box coordinates
                vertices = []
                for vertex in object_.bounding_poly.normalized_vertices:
                    vertices.append((vertex.x, vertex.y))
                
                print(f"  - Object: {object_.name}, Score: {confidence:.4f}, Bounds: {vertices}")
                
                # Check if this object matches our target classes
                matched_class = None
                for target_class in self.class_names:
                    if target_class.lower() in object_name or object_name in target_class.lower():
                        matched_class = target_class
                        detected_target_classes.add(target_class)
                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            best_object_for_saving = matched_class
                        break
                
                # Add object to our results
                objects_detected.append({
                    "name": object_.name,
                    "confidence": confidence,
                    "bounding_box": vertices,
                    "matched_class": matched_class
                })
            
            # Prepare the result
            result = {
                "objects": objects_detected,
                "detected_classes": list(detected_target_classes),
                "highest_confidence": highest_confidence,
                "best_object_for_saving": best_object_for_saving
            }
            
            if not detected_target_classes:
                if objects_detected:
                    print(f"No target classes detected among {len(objects_detected)} objects.")
                else:
                    print("No objects detected in the image.")
            else:
                print(f"Detected target classes: {', '.join(detected_target_classes)} with highest confidence {highest_confidence:.4f}")
                
            return result
            
        except Exception as e:
            print(f"Error in object localization: {e}")
            return {"objects": [], "error": str(e)}