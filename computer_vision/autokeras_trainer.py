"""
Handles the training and exporting of an AutoKeras image classification model.
"""
import os
import numpy as np
import tensorflow
import autokeras as ak
import traceback

class AutoKerasTrainer:
    """
    Manages the training process for an AutoKeras ImageClassifier,
    including data loading, model fitting, and exporting the trained model.
    """
    def __init__(self, data_dir, model_path, class_names, project_name="autokeras_boat_ramp_project", max_trials=3, epochs=10):
        """
        Initializes the AutoKerasTrainer.

        Args:
            data_dir (str): Path to the directory containing training image data,
                            organized into subdirectories named by class.
            model_path (str): Path where the trained Keras model should be saved.
            class_names (list): A list of strings representing the class names,
                                derived from the subdirectory names in data_dir.
            project_name (str, optional): Name for the AutoKeras project. This is
                                          used by AutoKeras to store temporary files
                                          related to the hyperparameter tuning trials.
                                          Defaults to "autokeras_boat_ramp_project".
            max_trials (int, optional): Maximum number of different Keras models
                                        for AutoKeras to try. Defaults to 3.
            epochs (int, optional): Number of epochs to train each model for.
                                    Defaults to 10.
        """
        self.data_dir = data_dir
        self.model_path = model_path
        self.class_names = class_names
        self.project_name = project_name # Used by AutoKeras for its working files
        self.max_trials = max_trials
        self.epochs = epochs
        # Initialize the AutoKeras ImageClassifier. Overwrite ensures fresh trials if project_name dir exists.
        self.clf = ak.ImageClassifier(
            max_trials=self.max_trials, 
            overwrite=True,  # Set to False if you want to resume previous HPO search
            project_name=self.project_name,
            directory=os.path.dirname(self.model_path) # Store AK project near the model
        )

    def train_and_export(self):
        """
        Trains the AutoKeras image classification model using data from self.data_dir
        and exports the best model found to self.model_path.

        The training data is loaded using tensorflow.keras.utils.image_dataset_from_directory.
        Batch size is dynamically adjusted based on the number of images.

        Returns:
            bool: True if training and export were successful, False otherwise.
        """
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            print(f"Data directory {self.data_dir} is empty or does not exist. Skipping training.")
            return False

        if not self.class_names:
            print("Error: Class names not determined. Cannot create dataset. Ensure data subdirectories exist.")
            return False
        print(f"Class names for training: {self.class_names}")

        total_images = 0
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        for class_name_item in self.class_names: # Renamed to avoid conflict with outer scope if any
            class_path = os.path.join(self.data_dir, class_name_item)
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
            return False
        if total_images == 1: # AutoKeras needs to split data, so >1 image is essential
            print("ERROR: Only 1 image found. AutoKeras requires the dataset to be splittable. Training cannot proceed.")
            return False

        batch_size: int
        if total_images < 4: # Covers 2 or 3 images; ensures at least 2 batches if batch_size=1
            batch_size = 1
            print(f"WARNING: Very few training images ({total_images}). Using batch_size=1.")
        else: # total_images >= 4
            # Set batch_size to half the images, capped at 32, ensuring at least 1.
            # This aims to produce at least 2 batches for AutoKeras's internal validation split.
            batch_size = min(total_images // 2, 32) 
            batch_size = max(1, batch_size) # Ensure batch_size is at least 1.
            if total_images < 50:
                print(f"WARNING: Low number of training images ({total_images}). Using batch_size={batch_size}.")
        
        try:
            image_size = (256, 256)
            print(f"Loading training data from {self.data_dir} with image_size={image_size} and batch_size={batch_size}")
            
            train_dataset = tensorflow.keras.utils.image_dataset_from_directory(
                self.data_dir,
                labels='inferred', # Infers labels from directory names
                label_mode='categorical', # For multi-class classification with AutoKeras
                class_names=self.class_names, # Ensures consistent class indexing
                image_size=image_size,
                batch_size=batch_size,
                shuffle=True,
                seed=123
            )
            
            for images, labels in train_dataset.take(1):
                print(f"Dataset batch shape: images={images.shape}, labels={labels.shape}")
                break

            # AutoKeras ImageClassifier can accept a tf.data.Dataset directly
            print("Fitting AutoKeras model with the loaded dataset...")
            self.clf.fit(train_dataset, epochs=self.epochs)
            print("Model training complete.")
            
            print(f"Exporting model to {self.model_path}...")
            # Exports the best model found by AutoKeras during the 'fit' process.
            self.clf.export_model(filepath=self.model_path)
            print("Model exported successfully.")
            return True
        except Exception as e:
            print(f"An error occurred during model training or export: {e}")
            traceback.print_exc()
            return False

