import hashlib
import logging
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_URL = "https://ndownloader.figshare.com/files/38548058"
MODEL_MD5 = "ef1e07ff46ee1b0b2cf7432d438c9560"
MODEL_NAME = "master"
DEFAULT_DESTINATION = Path.home() / ".cenfind" / "models"


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "download-model",
        help="Download and cache the pretrained SpotNet model weights from Figshare",
    )
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=DEFAULT_DESTINATION,
        help="Directory to download and extract the model into",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model already exists at the destination",
    )

    return parser


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, timeout: float = 30) -> None:
    with urlopen(url, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(destination, "wb") as out_file, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading model"
        ) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))


def _extract_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extracts archive into destination, rejecting members that would land
    outside it (zip-slip: entries using `../` or an absolute path)."""
    destination = destination.resolve()
    for member in archive.namelist():
        resolved = (destination / member).resolve()
        if destination not in resolved.parents and resolved != destination:
            raise ValueError(f"Refusing to extract unsafe path from archive: {member}")
    archive.extractall(destination)


def run(args):
    destination: Path = args.destination
    model_dir = destination / MODEL_NAME

    if model_dir.exists() and not args.force:
        logger.info("Model already present at %s (use --force to re-download)" % model_dir)
        print(model_dir)
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "master.zip"

    logger.info("Downloading model weights from %s" % MODEL_URL)
    _download(MODEL_URL, archive_path)

    checksum = _md5(archive_path)
    if checksum != MODEL_MD5:
        archive_path.unlink()
        raise ValueError(
            "Downloaded file checksum mismatch (expected %s, got %s). Aborting." % (MODEL_MD5, checksum)
        )

    with zipfile.ZipFile(archive_path) as zf:
        _extract_safely(zf, destination)
    archive_path.unlink()

    macosx_dir = destination / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    logger.info("Model downloaded and extracted to %s" % model_dir)
    print(model_dir)
    return 0
