"""
Handles the training and exporting of an AutoKeras image classification model.
"""
import os
import tensorflow as tf
import autokeras as ak
import traceback

class AutoKerasTrainer:
    """
    Manages the training process for an AutoKeras ImageClassifier,
    including data loading, model fitting, and exporting the trained model.
    """
    def __init__(self, image_data_dir, model_save_path, class_names, max_trials=10, epochs=10):
        """
        Initializes the AutoKerasTrainer.

        Args:
            image_data_dir (str): Path to the root directory of the image dataset.
                                  This directory should contain subdirectories for each class.
            model_save_path (str): Path where the trained Keras model will be saved (e.g., 'model.keras').
            class_names (list): A list of class names. These should correspond to the
                                subdirectory names in image_data_dir.
            max_trials (int): The maximum number of different Keras models to try.
            epochs (int): The number of epochs to train each model.
        """
        self.image_data_dir = image_data_dir
        self.model_save_path = model_save_path
        self.class_names = class_names # Expected to be provided
        self.max_trials = max_trials
        self.epochs = epochs

        if not self.class_names:
            raise ValueError("class_names must be provided and cannot be empty.")
        if not os.path.isdir(self.image_data_dir):
            # Create the directory if it doesn't exist, as per previous logic for dummy data.
            # However, for actual training, it should ideally exist and contain data.
            print(f"Warning: Image data directory '{self.image_data_dir}' not found. It will be created if dummy data generation is run.")
            # os.makedirs(self.image_data_dir, exist_ok=True) # Consider if this is the right place or should be handled by caller

    def _prepare_datasets(self):
        """
        Prepares training and testing datasets from the image_data_dir.
        This version uses autokeras.image_dataset_from_directory.
        Assumes a directory structure like:
        image_data_dir/
            class_a/
                image1.jpg
                image2.jpg
            class_b/
                image3.jpg
                image4.jpg
        """
        if not os.path.exists(self.image_data_dir) or not any(os.scandir(self.image_data_dir)):
            print(f"Error: Image data directory '{self.image_data_dir}' is empty or does not exist.")
            print("Please ensure it contains subdirectories for each class, populated with images.")
            # Example: Create dummy data if it's missing (for testing/demonstration)
            # self._create_dummy_data_if_needed() # This was part of the example, decide if it belongs here for production
            # if not os.path.exists(self.image_data_dir) or not any(os.scandir(self.image_data_dir)):
            return None, None # Return None if data is still not available

        # Ensure class_names match directory names for safety, though image_dataset_from_directory infers them.
        # This is a good check if class_names are passed externally.
        # for name in self.class_names:
        #     if not os.path.isdir(os.path.join(self.image_data_dir, name)):
        #         print(f"Warning: Directory for class '{name}' not found in '{self.image_data_dir}'.")

        try:
            # Load the dataset using AutoKeras utility, splitting into training and testing
            # The utility will infer class names from directory structure if not explicitly passed,
            # but it's good to be aware of how `class_names` (passed to __init__) relates.
            # `labels='inferred'` is default and uses dir names as labels.
            # `label_mode='categorical'` for multi-class classification (one-hot encoded).
            # `image_size` should match what the model expects (AutoKeras handles this internally to some extent).
            # `validation_split` creates a test set from the training data.
            print(f"Loading images from: {self.image_data_dir}")
            train_dataset = ak.image_dataset_from_directory(
                self.image_data_dir,
                validation_split=0.2, # Use 20% of the data for validation/testing
                subset="training",
                seed=123, # For reproducibility
                image_size=(256, 256), # Standardize image size
                batch_size=32, # Adjust batch size based on memory
                label_mode='categorical', # For multi-class classification
                class_names=self.class_names # Explicitly pass class names
            )
            test_dataset = ak.image_dataset_from_directory(
                self.image_data_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(256, 256),
                batch_size=32,
                label_mode='categorical',
                class_names=self.class_names # Explicitly pass class names
            )
            
            # Log the found class names from the dataset to confirm
            if hasattr(train_dataset, 'class_names'):
                print(f"Dataset loaded. Found class names: {train_dataset.class_names}")
                if set(train_dataset.class_names) != set(self.class_names):
                    print(f"Warning: Class names from dataset ({train_dataset.class_names}) do not exactly match provided class_names ({self.class_names}).")
            else:
                print("Class names not directly available on dataset object after loading.")

            # Prefetch for performance
            train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            
            return train_dataset, test_dataset
        except Exception as e:
            print(f"Error preparing dataset from '{self.image_data_dir}': {e}")
            print("Ensure the directory structure is correct (subdirectories for classes) and images are valid.")
            return None, None

    def train(self):
        """
        Trains an AutoKeras ImageClassifier model and saves it.

        Returns:
            tensorflow.keras.Model: The trained Keras model, or None if training failed.
        """
        print("Preparing datasets for training...")
        train_dataset, test_dataset = self._prepare_datasets()

        if train_dataset is None or test_dataset is None:
            print("Failed to prepare datasets. Aborting training.")
            return None

        print(f"Starting AutoKeras training with max_trials={self.max_trials} and epochs={self.epochs}.")
        try:
            # Initialize ImageClassifier
            # Ensure num_classes is correctly set based on the provided class_names
            clf = ak.ImageClassifier(
                overwrite=True, 
                max_trials=self.max_trials, 
                project_name="autokeras_cv_project", # Can be customized
                # num_classes=len(self.class_names) # AutoKeras usually infers this from data
            )

            # Feed the model with training data
            # Early stopping is implicitly handled by AutoKeras during the search phase
            # and can be configured further if needed via callbacks.
            clf.fit(train_dataset, epochs=self.epochs, validation_data=test_dataset)

            print("Training complete. Evaluating model...")
            loss, accuracy = clf.evaluate(test_dataset)
            print(f"Accuracy on test set: {accuracy*100:.2f}%")

            print("Exporting the best model...")
            # Export the model as a Keras model (TensorFlow SavedModel format)
            # The model is saved to a directory if model_save_path ends with '/', 
            # or as an H5 file if it ends with '.h5'. For '.keras', it's the new Keras v3 format.
            exported_model = clf.export_model()
            
            # Ensure the directory for the model_save_path exists
            model_save_dir = os.path.dirname(self.model_save_path)
            if model_save_dir and not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir, exist_ok=True)
            
            try:
                exported_model.save(self.model_save_path, save_format="tf") # Explicitly use SavedModel format
                print(f"Model saved successfully to {self.model_save_path}")
                return exported_model # Return the Keras model object
            except Exception as e:
                print(f"Error saving exported model to {self.model_save_path}: {e}")
                # Fallback or alternative save if needed, e.g., H5, though SavedModel is preferred.
                # try:
                #     h5_path = self.model_save_path.replace(".keras", "") + ".h5"
                #     exported_model.save(h5_path)
                #     print(f"Model saved in H5 format to {h5_path} as a fallback.")
                #     return exported_model
                # except Exception as e2:
                #     print(f"Error saving model in H5 format: {e2}")
                return None # Indicate failure to save

        except Exception as e:
            print(f"An error occurred during the training process: {e}")
            # Log the full traceback for debugging if necessary
            import traceback
            traceback.print_exc()
            return None

