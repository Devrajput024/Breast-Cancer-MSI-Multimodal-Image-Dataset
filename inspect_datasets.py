from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from PIL import Image

root = Path(r"c:\Users\asus\OneDrive\Desktop\Breast Cncer Project")
zip_names = [
    "Chest_XRay_MSI.zip",
    "Histopathological_MSI.zip",
    "Ultrasound Images_MSI.zip",
]
missing = [name for name in zip_names if not (root / name).exists()]
print("MISSING_ZIPS:", missing if missing else "None")

extract_root = root / "datasets"
extract_root.mkdir(exist_ok=True)

for zip_name in zip_names:
    archive = root / zip_name
    print(f"\n=== {zip_name} ===")
    if not archive.exists():
        continue

    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        top_level = sorted({name.split('/')[0] for name in names if name.strip()})
        print("Top-level entries:", top_level)
        print("Sample entries:")
        for name in names[:16]:
            print("  " + name)
        print("...")

        out_dir = extract_root / zip_name.replace(" ", "_").rsplit(".", 1)[0]
        tmp_dir = extract_root / (zip_name.replace(" ", "_").rsplit(".", 1)[0] + "_tmp")

        if out_dir.exists():
            shutil.rmtree(out_dir)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tmp_dir.mkdir(parents=True, exist_ok=True)
        z.extractall(tmp_dir)

        nested = list(tmp_dir.iterdir())
        if len(nested) == 1 and nested[0].is_dir() and nested[0].name == out_dir.name:
            src = nested[0]
            for child in src.iterdir():
                shutil.move(str(child), str(out_dir))
            shutil.rmtree(src)
        else:
            for child in nested:
                shutil.move(str(child), str(out_dir))

        shutil.rmtree(tmp_dir)
        print("Extracted to:", out_dir)

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
        class_counts = {}
        for child in sorted(out_dir.iterdir()):
            if not child.is_dir():
                continue
            count = 0
            for file in child.rglob("*"):
                if file.is_file() and file.suffix.lower() in image_exts:
                    try:
                        with Image.open(file) as img:
                            img.verify()
                        count += 1
                    except Exception as exc:
                        raise RuntimeError(f"Failed to open image {file}: {exc}") from exc
            class_counts[child.name] = count
            print(f"Class {child.name}: {count} images")

        print("CLASS_COUNTS:", class_counts)
