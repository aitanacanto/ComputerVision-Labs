"""Download the official checkpoints required by lab 02; no model code is executed."""
from pathlib import Path
from urllib.request import urlretrieve

BASE = Path(__file__).resolve().parents[1] / "labs" / "02-resnets"
URLS = {
    "resnet18.pth": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet50.pth": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
}

if __name__ == "__main__":
    for name, url in URLS.items():
        path = BASE / name
        if path.exists():
            print(f"Already present: {name}")
        else:
            print(f"Downloading {name}")
            urlretrieve(url, path)
