"""
Provides functionality to download video stream URLs from YouTube.
"""
import yt_dlp

class YouTubeDownloader:
    """A class to encapsulate YouTube stream URL retrieval using yt-dlp."""
    def __init__(self, quiet: bool = True):
        self.ydl_opts = {
            # Try for 1080p, then 720p, then best MP4, then overall best
            'format': 'best[height>=1080][ext=mp4]/best[height>=720][ext=mp4]/best[ext=mp4]/best',
            'quiet': quiet,
        }

    def get_stream_url(self, video_url: str) -> str:
        """Return the direct video stream URL.

        For live streams, this URL typically points to a manifest file (e.g., HLS .m3u8)
        which allows a video player to access the most recent segments of the stream (the live edge).
        The player (e.g., OpenCV in FrameExtractor) is responsible for handling the manifest
        to play the live content.

        Args:
            video_url: The URL of the YouTube video.

        Returns:
            The URL of the best available MP4 video stream.

        Raises:
            RuntimeError: If a stream URL cannot be extracted.
        """
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            
            # Attempt to find the requested format; otherwise, fallback
            requested_format = next((f for f in info_dict.get('formats', []) if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('acodec') != 'none'), None)
            if requested_format:
                return requested_format['url']
            
            # Fallback if specific mp4 not found
            if 'url' in info_dict: # For direct links or simpler cases
                return info_dict['url']
            elif info_dict.get('formats'): 
                mp4_formats = [f for f in info_dict['formats'] if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('acodec') != 'none']
                if mp4_formats:
                    return mp4_formats[0]['url']
                
                suitable_formats = [f for f in info_dict['formats'] if f.get('url') and f.get('vcodec') != 'none' and f.get('acodec') != 'none']
                if suitable_formats:
                    return suitable_formats[0]['url']

                if info_dict['formats']: # Last resort, take the first format
                    return info_dict['formats'][0]['url']

        raise RuntimeError(f"Could not extract stream URL for {video_url} using yt-dlp.")

