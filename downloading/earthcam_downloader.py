"""
Provides functionality to extract HLS stream URLs from EarthCam pages.
"""
import requests
import re
from bs4 import BeautifulSoup

class EarthCamDownloader:
    """
    A class to encapsulate EarthCam stream URL retrieval.
    It fetches the EarthCam page, parses it to find streaming metadata,
    and constructs the HLS playlist URL.
    """
    def __init__(self, timeout=10):
        """
        Initializes the EarthCamDownloader.

        Args:
            timeout (int, optional): Timeout in seconds for network requests. Defaults to 10.
        """
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_stream_url(self, earthcam_page_url: str) -> str | None:
        """
        Retrieves the HLS stream URL from an EarthCam page.

        Args:
            earthcam_page_url (str): The URL of the EarthCam page.

        Returns:
            str: The HLS playlist URL if found, otherwise None.
        
        Raises:
            RuntimeError: If the page cannot be fetched or parsing fails critically.
        """
        print(f"Fetching EarthCam page: {earthcam_page_url}")
        try:
            response = requests.get(earthcam_page_url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch EarthCam page {earthcam_page_url}: {e}")

        print("Parsing EarthCam page content...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        scripts = soup.find_all('script')
        
        domain = None
        path = None

        # Regex to find the specific JSON-like assignments
        # Looking for something like: camPlayer.options.html5_streamingdomain = "domain";
        # Or within a JSON structure: "html5_streamingdomain":"domain"
        domain_regex = re.compile(r'["\']html5_streamingdomain["\']\s*[:=]\s*["\']([^"\']+)["\']', re.IGNORECASE)
        path_regex = re.compile(r'["\']html5_streampath["\']\s*[:=]\s*["\']([^"\']+)["\']', re.IGNORECASE)

        for script in scripts:
            if script.string: # Check for inline scripts
                script_content = script.string
                
                if not domain:
                    domain_match = domain_regex.search(script_content)
                    if domain_match:
                        domain = domain_match.group(1).replace('\\/', '/')
                        print(f"Found html5_streamingdomain: {domain}")

                if not path:
                    path_match = path_regex.search(script_content)
                    if path_match:
                        path = path_match.group(1).replace('\\/', '/')
                        print(f"Found html5_streampath: {path}")
                
                if domain and path:
                    break
        
        if domain and path:
            full_url = f"{domain.rstrip('/')}{path}"
            print(f"Constructed HLS URL: {full_url}")
            return full_url
        else:
            error_message = "Could not find html5_streamingdomain or html5_streampath in page scripts.\n"
            if not domain:
                error_message += " Domain not found."
            if not path:
                error_message += " Path not found."
            print(error_message)
            # Optionally, could log the script contents here for debugging if it's small enough
            # or a portion of it.
            # For now, just raising an error if critical info is missing.
            raise RuntimeError(error_message)

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    # Replace with a valid EarthCam URL that is known to have the required script structure
    test_url = "https://myearthcam.com/vmr441" 
    # This URL is an example, its structure might change or it might not be a live cam with HLS.
    # A more reliable test URL would be needed.
    
    print(f"Testing EarthCamDownloader with URL: {test_url}")
    downloader = EarthCamDownloader()
    try:
        hls_url = downloader.get_stream_url(test_url)
        if hls_url:
            print(f"Successfully extracted HLS URL: {hls_url}")
        else:
            print("Failed to extract HLS URL.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")

