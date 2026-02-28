from typing import Optional, Union, List, Any
from pathlib import Path
import hashlib
import requests
import sys
import logging
import os
import tarfile

logger = logging.getLogger(__name__)


def _calculate_md5(fpath: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def _check_md5(fpath: Union[str, Path], md5: str, **kwargs: Any) -> bool:
    return md5 == _calculate_md5(fpath, **kwargs)

def _check_integrity(fpath: Union[str, Path], md5: Optional[str] = None) -> bool:
    if isinstance(fpath, str):
        fpath = Path(fpath)
    if not fpath.is_file():
        return False
    if md5 is None:
        return True
    return _check_md5(fpath, md5)

def _download_from_url(
    url: str,
    root: Union[str, Path],
    filename: Optional[Union[str, Path]] = None,
    md5: Optional[str] = None,
    size: Optional[int] = None,
    chunk_size: Optional[int] = 256 * 64,
    extract_tars: bool = True,
) -> None:
    if isinstance(root, str):
        root = Path(root)

    root = root.expanduser()
    if not filename:
        # grab file ext from basename
        filename = url.split("/")[-1]
    fpath = root / filename

    root.mkdir(parents=True, exist_ok=True)

    if _check_integrity(fpath, md5):
        logger.info("Using downloaded and verified file: " + str(fpath))
        return

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fpath, "wb") as f:
            curr_size = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                    curr_size += chunk_size
                    prog = curr_size / size
                    print(f"Download in progress: {prog:.2%}", end="\r")

            assert (
                fpath.stat().st_size == size
            ), (
                f"Error: mismatch between expected and true size of downloaded file {fpath}. "
                "Delete the file and try again."
            )

    if not _check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
    else:
        logger.info("saved to {fpath} successfully")
        if extract_tars:
            logger.info("Extracting {fpath}...")
            archive = tarfile.open(fpath)
            archive.extractall(path=root)
            name_str = ", ".join(archive.getnames())
            logger.info(f"Extracted {name_str}")


def download_from_zenodo_record(
    record_id: str,
    root: Union[str, Path],
    files_to_download: Optional[List[str]] = None,
):
    if files_to_download is not None and len(files_to_download) == 0:
        return
    zenodo_api_url = "https://zenodo.org/api/records/"
    url = f"{zenodo_api_url}{record_id}"
    resp = requests.get(url)
    assert (
        resp.status_code == 200
    ), f"Error: request failed with status code {resp.status_code}"
    response_json = resp.json()
    for file_record in response_json["files"]:
        fname = file_record["key"]
        if files_to_download is None or fname in files_to_download:
            _download_from_url(
                url=file_record["links"]["self"],
                md5=file_record["checksum"][4:],  # md5 stored as 'md5:xxxxx'
                size=file_record["size"],
                root=root,
                filename=fname,
                extract_tars=True,
            )
