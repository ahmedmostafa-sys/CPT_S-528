"""
prepare_dsct_data.py
--------------------
Downloads CIFAR-10 (ID) and TinyImageNet (OOD) and converts them into
the folder layout expected by DSCTTrainer.
"""

import os, shutil, random, tarfile, urllib.request
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from PIL import Image

ID_ROOT = "data/ID"
OOD_ROOT = "data/OOD_flat"
os.makedirs(ID_ROOT, exist_ok=True)
os.makedirs(OOD_ROOT, exist_ok=True)

# -------------------------------
# 1️⃣ Download CIFAR-10 (ID data)
# -------------------------------
print("[Download] CIFAR-10 (ID)")
id_ds = CIFAR10(root="data/raw_cifar10", train=True, download=True)
class_names = id_ds.classes
print(f"[Info] CIFAR-10 classes: {class_names}")

# Save CIFAR-10 images into class folders
transform = T.ToPILImage()
# Save CIFAR-10 images into class folders
for idx, (img, label) in enumerate(id_ds):
    cls_name = class_names[label]
    outdir = os.path.join(ID_ROOT, cls_name)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{idx:05d}.jpg")
    if not os.path.exists(outpath):
        img.save(outpath)  # <-- direct save, img is already a PIL.Image
print(f"[Done] Saved {len(id_ds)} ID images to {ID_ROOT}")


# ---------------------------------
# 2️ Download TinyImageNet (OOD)
# ---------------------------------
TIN_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TIN_ZIP = "data/tiny-imagenet-200.zip"
TIN_DIR = "data/tiny-imagenet-200"

if not os.path.exists(TIN_DIR):
    print("[Download] TinyImageNet (OOD)")
    urllib.request.urlretrieve(TIN_URL, TIN_ZIP)
    print("[Extract] Unzipping TinyImageNet...")
    shutil.unpack_archive(TIN_ZIP, "data")
else:
    print("[Found] TinyImageNet already downloaded.")

# Copy a small random subset to OOD_flat (e.g., 5k samples)
print("[Prepare] Copying TinyImageNet OOD subset...")
ood_count = 0
for wnid in os.listdir(os.path.join(TIN_DIR, "train")):
    img_dir = os.path.join(TIN_DIR, "train", wnid, "images")
    if not os.path.isdir(img_dir): continue
    imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".JPEG")]
    random.shuffle(imgs)
    for f in imgs[:25]:  # adjust number per class if needed
        shutil.copy(f, os.path.join(OOD_ROOT, os.path.basename(f)))
        ood_count += 1
print(f"[Done] Copied {ood_count} OOD images to {OOD_ROOT}")

# ---------------------------------
# 3️ Summary
# ---------------------------------
print("\n✅ Dataset prepared successfully!")
print(f"ID root:  {os.path.abspath(ID_ROOT)}")
print(f"OOD root: {os.path.abspath(OOD_ROOT)}")
print(f"Classes:  {class_names}")
print("\nExample run:")
print(f"python dsct_experiments.py --mode train --class_names {' '.join(class_names)}")
