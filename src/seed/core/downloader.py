from pathlib import Path

import requests
from tqdm import tqdm

from ..config import DEFAULT_CHUNK_SIZE, DEFAULT_TIMEOUT


class Downloader:
    """
    Downloader class for downloading files from a URL.
    """

    def __init__(
        self,
        url: str,
        output_path: Path | str,
    ):
        """
        Initializes the downloader.

        Args:
            url (str): URL of the file to download.
            output_path (Path | str): Output file path.
        """
        self._url: str = url
        self._output_path: Path = Path(output_path)

    def download(
        self,
        force_download: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> bool:
        """
        Downloads the file from the specified URL.

        Args:
            force_download (bool): Whether to force re-download if the file already exists.
            chunk_size (int): Size of each chunk to download in bytes.
            timeout (int): Request timeout in seconds.

        Returns:
            bool: True if the file was downloaded, False if it already existed.
        """
        if not force_download and self._output_path.exists():
            return False

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(self._url, timeout=timeout, stream=True) as response:
            response.raise_for_status()

            with (
                self._output_path.open("wb") as file,
                tqdm(
                    desc="Downloading",
                    total=int(response.headers.get("content-length", 0)),
                    unit="B",
                    unit_scale=True,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))

        return True
