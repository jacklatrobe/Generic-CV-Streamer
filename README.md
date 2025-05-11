# Generic-CV-Streamer
Generic-CV-Streamer is a Python application designed to capture frames from video streams (YouTube, EarthCam, or direct stream URLs), process these frames using computer vision to identify objects, and save relevant frames.

## Features

*   Capture frames from YouTube live streams, pre-recorded videos, and EarthCam live cameras.
*   Extract frames at configurable intervals.
*   Process frames using different computer vision backends:
    *   **Local**: Using AutoKeras for training and inference (requires local training data).
    *   **Google Cloud Vision**: Utilizes the Google Cloud Vision API.
    *   **Microsoft Azure Vision**: Utilizes the Azure Computer Vision API.
*   Configurable via a `config.json` file.
*   Command-line interface for easy operation.

## Installation

1.  **Clone the repository (if applicable) or download the source code.**
2.  **Install Python 3.9 or higher.**
3.  **Install required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your system and whether you intend to use the local AutoKeras backend with GPU acceleration, you might need to install TensorFlow and CUDA separately according to the official TensorFlow documentation.*

4.  **Set up Credentials (for Cloud CV backends):**
    *   **Google Cloud Vision:**
        *   Create a service account key JSON file from your Google Cloud Console.
        *   You can specify the path to this file via the `--credentials-path` argument, or in `config.json` as `"credentials_path": "path/to/your/google-credentials.json"`.
    *   **Microsoft Azure Vision:**
        *   Create an Azure Computer Vision resource and obtain its API key and endpoint.
        *   Create a JSON file (e.g., `azure-cv-key.json`) with the following structure:
            ```json
            {
                "api_key": "YOUR_AZURE_CV_API_KEY",
                "endpoint": "YOUR_AZURE_CV_ENDPOINT"
            }
            ```
        *   You can specify the path to this file via the `--credentials-path` argument, or in `config.json` as `"azure_credentials_path": "path/to/your/azure-cv-key.json"`.

5.  **Configure the application:**
    *   Rename `config.example.json` to `config.json` (if an example is provided) or create `config.json`.
    *   Edit `config.json` to set your desired `class_names` (objects to detect), `save_dir`, `capture_every_sec`, and other settings. See `config_manager.py` for details on configurable options.

## Usage

Run the application from the command line:

```bash
python main.py --url <STREAM_URL> [OPTIONS]
```

**Command-line Arguments:**

*   `--url <STREAM_URL>`: (Required if not in `config.json`) URL of the video stream (YouTube, EarthCam, or direct HLS/MPEG-DASH).
*   `--config <CONFIG_FILE_PATH>`: Path to the configuration JSON file (default: `config.json`).
*   `--no-cv`: Disable Computer Vision processing. Only captures and saves frames.
*   `--retrain-cv`: (For `local` backend only) Retrain the AutoKeras model. If not set, uses an existing model.
*   `--cv-backend <BACKEND>`: Specify the computer vision backend.
    *   `local`: Use AutoKeras (requires local training).
    *   `google`: Use Google Cloud Vision API.
    *   `azure`: Use Microsoft Azure Vision API.
    *   (Default: `local`)
*   `--credentials-path <CREDENTIALS_FILE_PATH>`: Path to the cloud service credentials JSON file (for Google Cloud or Azure). Overrides paths specified in `config.json`.
*   `-h`, `--help`: Show the help message and exit.

**Examples:**

*   Capture frames from a YouTube stream using Google CV:
    ```bash
    python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --cv-backend google --credentials-path "path/to/google-credentials.json"
    ```
*   Capture frames from an EarthCam stream using Azure CV:
    ```bash
    python main.py --url "https://www.earthcam.com/CAM_URL" --cv-backend azure --credentials-path "path/to/azure-cv-key.json"
    ```
*   Retrain and use the local AutoKeras model:
    ```bash
    python main.py --url "rtsp://your.stream.url" --cv-backend local --retrain-cv
    ```
*   Capture frames only, no CV processing:
    ```bash
    python main.py --url "https://your.stream/playlist.m3u8" --no-cv
    ```

## Preparing Local Training Data (for AutoKeras `local` backend)

If you use the `local` CV backend, you need to provide training data for AutoKeras. The application expects a directory structure like this:

```
<autokeras_image_data_dir>/  (e.g., data/images/ specified in config.json)
    train/
        class1/
            image1.jpg
            image2.png
            ...
        class2/
            imageA.jpg
            imageB.jpeg
            ...
        ...
    test/  (Optional, but recommended for evaluating model performance)
        class1/
            image_test1.jpg
            ...
        class2/
            image_testA.jpg
            ...
        ...
```

1.  **Set `autokeras_image_data_dir` in your `config.json`**: This is the root directory for your training images (e.g., `"autokeras_image_data_dir": "data/images/"`).
2.  **Create `train` and (optionally) `test` subdirectories** inside `autokeras_image_data_dir`.
3.  **Inside `train` (and `test`), create subdirectories for each class name** you want to detect. The names of these subdirectories should exactly match the `class_names` defined in your `config.json`.
4.  **Place your training images** into the respective class subdirectories. AutoKeras will use these images to train the model.

The more diverse and representative your training images are, the better the model will perform. Ensure images are of reasonable quality and show the objects in various conditions they might appear in the video streams. Note that when you use a cloud API, it saves it's detections, so you can use one of the cloud APIs (both offer limited free trials) to gather training data for your local model.