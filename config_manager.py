\
import json
import os

class ConfigManager:
    """
    Manages loading and accessing configuration settings from a JSON file.
    """
    def __init__(self, config_path="config.json"):
        """
        Initializes the ConfigManager and loads the configuration.

        Args:
            config_path (str): Path to the JSON configuration file.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            json.JSONDecodeError: If the configuration file is not valid JSON.
            IOError: For other I/O errors during file reading.
        """
        self.config_path = config_path
        self._config_data = self._load_config()

    def _load_config(self):
        """Loads the configuration from the JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.")
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # Re-raise with more context
            raise json.JSONDecodeError(f"Could not decode JSON from '{self.config_path}': {e.msg}", e.doc, e.pos) from e
        except Exception as e:
            raise IOError(f"An error occurred while reading config file '{self.config_path}': {e}") from e

    def get_setting(self, key, default=None):
        """
        Retrieves a setting by key.

        Args:
            key (str): The configuration key.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default.
        """
        return self._config_data.get(key, default)

    def get_class_names(self, default=None):
        """
        Retrieves the 'class_names' setting.

        Args:
            default (list, optional): Default value if 'class_names' is not found.

        Returns:
            list: A list of class names.

        Raises:
            KeyError: If 'class_names' is not found and no default is provided.
            ValueError: If 'class_names' is not a list of strings.
        """
        names = self._config_data.get("class_names")
        if names is None:
            if default is not None:
                return default
            raise KeyError(f"'class_names' not found in configuration file '{self.config_path}'.")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("'class_names' in configuration must be a list of strings.")
        return names

    def get_raw_save_dir(self, default="raw_frames"):
        """Retrieves the 'raw_save_dir' setting."""
        return self._config_data.get("raw_save_dir", default)

    def get_detections_save_dir(self, default="detections"):
        """Retrieves the 'detections_save_dir' setting."""
        return self._config_data.get("detections_save_dir", default)

    def get_capture_every_sec(self, default=2.0):
        """Retrieves the 'capture_every_sec' setting."""
        val = self._config_data.get("capture_every_sec", default)
        try:
            return float(val)
        except (TypeError, ValueError):
            print(f"Warning: 'capture_every_sec' value '{val}' is not a valid number. Using default {default}.")
            return float(default)

    def get_confidence_threshold(self, default=0.7):
        """Retrieves the 'confidence_threshold' setting."""
        val = self._config_data.get("confidence_threshold", default)
        try:
            return float(val)
        except (TypeError, ValueError):
            print(f"Warning: 'confidence_threshold' value '{val}' is not a valid number. Using default {default}.")
            return float(default)

    def get_credentials_path(self, default="service_account.json"):
        """
        Retrieves the 'credentials_path' setting.
        A default value like "credentials/service_account.json" can be provided.
        """
        return self._config_data.get("credentials_path", default)

    def get_autokeras_model_dir(self, default="data/model/"):
        """Retrieves the 'autokeras_model_dir' setting (directory for the model)."""
        return self._config_data.get("autokeras_model_dir", default)

    def get_autokeras_model_filename(self, default="model.keras"):
        """Retrieves the 'autokeras_model_filename' setting."""
        return self._config_data.get("autokeras_model_filename", default)

    def get_autokeras_image_data_dir(self, default="data/images/"):
        """Retrieves the 'autokeras_image_data_dir' setting (directory for training images)."""
        return self._config_data.get("autokeras_image_data_dir", default)

    def get_expansion_map(self, default=None):
        """
        Retrieves the 'expansion_map' setting.

        Args:
            default (dict, optional): Default value if 'expansion_map' is not found.

        Returns:
            dict: The expansion map.

        Raises:
            KeyError: If 'expansion_map' is not found and no default is provided.
            ValueError: If 'expansion_map' is not a dictionary, or if its values are not lists of strings.
        """
        expansion_map = self._config_data.get("expansion_map")
        if expansion_map is None:
            if default is not None:
                return default
            # Return an empty dict by default if not found, to prevent crashes if it's optional
            # Or raise KeyError if it should be mandatory: 
            # raise KeyError(f"'expansion_map' not found in configuration file '{self.config_path}'.")
            return {}
        
        if not isinstance(expansion_map, dict):
            raise ValueError("'expansion_map' in configuration must be a dictionary.")
        
        for key, value in expansion_map.items():
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Invalid format for expansion_map key '{key}': must be a list of strings.")
        return expansion_map
