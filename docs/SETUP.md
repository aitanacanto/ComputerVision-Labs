# Setup

Use Python 3.10 or 3.11 as a starting environment. Dependencies are not locked; this is not a claim of a fully verified compatibility matrix.

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`  
macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
```

From the repository root, download lab 02 checkpoints with `python scripts/download_resnet_weights.py`. The URLs are the ones specified in the original assignment. Checkpoints and downloaded CIFAR-10 data are excluded from Git.

Run each command from its directory:

| Directory | Command |
| --- | --- |
| `labs/01-convolutions` | `python assignment1.py` |
| `labs/02-resnets` | `python assignment2.py` |
| `labs/03-transfer-learning` | `python assignment3.py` |
| `labs/04-vit-attention` | `python assignment4.py` |

Lab 03 automatically downloads CIFAR-10 and pretrained weights. Lab 04 downloads DINO weights and opens Matplotlib windows. Network access is needed on the first run of these labs. Pretrained checkpoints should only be obtained from the documented upstream sources.
