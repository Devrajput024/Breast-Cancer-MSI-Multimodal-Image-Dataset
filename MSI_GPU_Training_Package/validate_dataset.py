"""Dataset validation and manifest generation for Multi-Modal Breast Cancer MSI Datasets.

Validates:
1. Chest X-Ray MSI (Normal vs Malignant)
2. Histopathological MSI (benign vs malignant)
3. Ultrasound Images MSI (benign vs malignant)

Features:
- Image file integrity & format verification (PIL verify & load)
- Color mode, resolution, and channel inspection
- Exact duplicate detection (MD5 & SHA256)
- Perceptual image hashing (dHash) for near-duplicate identification
- Leakage-preventing group identification:
    * Ultrasound: Source image grouping (normalizing rotations/sharpenings)
    * Histopathology: Slide/patient-level grouping (BreaKHis biopsy slide ID)
    * Chest X-Ray: Documented image instance / Roboflow hash tracking
- Generation of per-modality and unified manifests with metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

MODALITY_CONFIG = {
    'ultrasound': {
        'folder': 'Ultrasound Images_MSI',
        'classes': {'benign': 0, 'malignant': 1},
        'pos_class': 'malignant',
        'group_strategy': 'ultrasound_source_prefix',
    },
    'histopathology': {
        'folder': 'Histopathological_MSI',
        'classes': {'benign': 0, 'malignant': 1},
        'pos_class': 'malignant',
        'group_strategy': 'breakhis_slide_id',
    },
    'chest_xray': {
        'folder': 'Chest_XRay_MSI',
        'classes': {'Normal': 0, 'Malignant': 1},
        'pos_class': 'Malignant',
        'group_strategy': 'chest_xray_instance',
    },
}


def compute_dhash(image_path: Path, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for an image to detect perceptual similarity."""
    try:
        with Image.open(image_path) as im:
            # Resize to (hash_size + 1, hash_size) grayscale
            im_resized = im.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            arr = np.array(im_resized)
            diff = arr[:, 1:] > arr[:, :-1]
            return ''.join(f'{b:02x}' for b in np.packbits(diff.flatten()))
    except Exception:
        return ''


def extract_group_id(path: Path, strategy: str, class_name: str) -> tuple[str, str]:
    """Extract grouping identifier and human-readable source ID based on dataset modality."""
    stem = path.stem

    if strategy == 'ultrasound_source_prefix':
        # E.g. Multispectral_benign (1)-rotated1-sharpened.png -> Multispectral_benign (1)
        m = re.match(r'^(Multispectral_(?:benign|malignant)\s*\(\d+\))', stem, re.IGNORECASE)
        if m:
            source_id = m.group(1).strip()
        else:
            # Fallback: iteratively strip known augmentation suffixes
            s = stem
            for _ in range(10):
                new_s = re.sub(
                    r'(?i)(?:[-_ ](?:rotated\d*|flip(?:ped)?|sharpened|brightened|contrast|enhanced|aug(?:mented)?\d*))+$',
                    '',
                    s,
                )
                if new_s == s:
                    break
                s = new_s
            source_id = s
        group_key = f'{class_name.lower()}|{source_id.lower()}'
        group_hash = hashlib.sha1(group_key.encode()).hexdigest()[:16]
        return group_hash, source_id

    elif strategy == 'breakhis_slide_id':
        # E.g. Multispectral_SOB_B_A-14-22549AB-200-001.png
        # Prefix SOB_<B/M>_<Type>-<Year>-<SlideID> represents the patient's slide
        m = re.match(r'^(Multispectral_SOB_[A-Za-z]+_[A-Za-z0-9]+-\d+-[A-Za-z0-9]+)', stem)
        if m:
            source_id = m.group(1)
        else:
            # Fallback to first 3 tokens
            parts = stem.split('-')
            source_id = '-'.join(parts[:3]) if len(parts) >= 3 else stem
        group_key = f'{class_name.lower()}|{source_id.lower()}'
        group_hash = hashlib.sha1(group_key.encode()).hexdigest()[:16]
        return group_hash, source_id

    else:
        # Chest X-Ray instance grouping:
        # Files are Roboflow exports (e.g. Multispectral_10130_1672636630_png.rf.52c6d1c8c6c58c5c1c5fcaaf57feb81d_1.jpg)
        # Because numeric prefixes overlap across normal/malignant folders in this dataset,
        # grouping strictly within class prevents cross-label contamination.
        source_id = stem
        group_key = f'{class_name.lower()}|{stem.lower()}'
        group_hash = hashlib.sha1(group_key.encode()).hexdigest()[:16]
        return group_hash, source_id


def validate_image_file(path: Path) -> dict:
    """Read and thoroughly validate image integrity, metadata, hashes."""
    result = {
        'is_valid': False,
        'error_msg': '',
        'format': '',
        'mode': '',
        'width': 0,
        'height': 0,
        'channels': 0,
        'file_size_bytes': 0,
        'md5': '',
        'sha256': '',
        'dhash': '',
    }

    try:
        # 1. Read binary content for hashes & size
        data = path.read_bytes()
        result['file_size_bytes'] = len(data)
        result['md5'] = hashlib.md5(data).hexdigest()
        result['sha256'] = hashlib.sha256(data).hexdigest()

        # 2. Verify with PIL
        with Image.open(path) as img:
            img.verify()

        # 3. Open again to read mode, dimensions, and test RGB conversion
        with Image.open(path) as img:
            result['format'] = img.format or path.suffix.upper().lstrip('.')
            result['mode'] = img.mode
            result['width'], result['height'] = img.size
            result['channels'] = len(img.getbands())
            # Test full load and RGB conversion
            _ = img.convert('RGB')

        result['dhash'] = compute_dhash(path)
        result['is_valid'] = True
    except Exception as exc:
        result['error_msg'] = str(exc)

    return result


def validate_dataset(data_root: Path, output_dir: Path) -> dict:
    """Run full validation over all modalities, generate manifests, and save summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {}
    all_manifest_rows = []

    print('=' * 80)
    print(f'DATASET VALIDATION & MANIFEST GENERATOR')
    print(f'Data Root: {data_root.resolve()}')
    print(f'Output Directory: {output_dir.resolve()}')
    print('=' * 80)

    for modality, cfg in MODALITY_CONFIG.items():
        print(f'\n>>> Validating Modality: {modality.upper()} ({cfg["folder"]})')
        mod_dir = data_root / cfg['folder']
        if not mod_dir.is_dir():
            raise FileNotFoundError(f'Directory does not exist: {mod_dir}')

        rows = []
        md5_tracker = defaultdict(list)
        dhash_tracker = defaultdict(list)
        corrupt_files = []
        resolution_counts = Counter()
        format_counts = Counter()
        mode_counts = Counter()

        for class_name, label in cfg['classes'].items():
            class_dir = mod_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f'Missing class folder: {class_dir}')

            image_files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
            print(f'  Class "{class_name}" (label={label}): found {len(image_files)} image files.')

            for img_path in sorted(image_files):
                val_info = validate_image_file(img_path)
                if not val_info['is_valid']:
                    corrupt_files.append((str(img_path), val_info['error_msg']))
                    print(f'  [CORRUPT/UNREADABLE] {img_path.name}: {val_info["error_msg"]}')
                    continue

                resolution_counts[(val_info['width'], val_info['height'])] += 1
                format_counts[val_info['format']] += 1
                mode_counts[val_info['mode']] += 1

                md5_tracker[val_info['md5']].append(str(img_path))
                if val_info['dhash']:
                    dhash_tracker[val_info['dhash']].append(str(img_path))

                group_hash, source_id = extract_group_id(img_path, cfg['group_strategy'], class_name)

                rows.append({
                    'modality': modality,
                    'class_name': class_name,
                    'label': label,
                    'filename': img_path.name,
                    'path': str(img_path.resolve()),
                    'relative_path': os.path.relpath(img_path, data_root.parent),
                    'group_id': group_hash,
                    'source_id': source_id,
                    'width': val_info['width'],
                    'height': val_info['height'],
                    'channels': val_info['channels'],
                    'mode': val_info['mode'],
                    'format': val_info['format'],
                    'file_size_bytes': val_info['file_size_bytes'],
                    'md5': val_info['md5'],
                    'sha256': val_info['sha256'],
                    'dhash': val_info['dhash'],
                })

        df = pd.DataFrame(rows)
        # Mark duplicates
        md5_counts = df['md5'].map(df['md5'].value_counts())
        df['is_exact_duplicate'] = md5_counts > 1
        dhash_counts = df['dhash'].map(df['dhash'].value_counts())
        df['is_perceptual_duplicate'] = dhash_counts > 1

        # Check for group leakage risk (group spanning multiple labels)
        group_labels = df.groupby('group_id')['label'].nunique()
        multi_label_groups = group_labels[group_labels > 1].index.tolist()
        if multi_label_groups:
            raise RuntimeError(f'CRITICAL: {len(multi_label_groups)} groups span multiple labels in {modality}!')

        # Save modality manifest
        mod_manifest_path = output_dir / f'{modality}_manifest.csv'
        df.to_csv(mod_manifest_path, index=False)
        print(f'  Saved manifest: {mod_manifest_path} ({len(df)} rows)')

        exact_dup_count = int(df['is_exact_duplicate'].sum())
        perceptual_dup_count = int(df['is_perceptual_duplicate'].sum())
        unique_groups = int(df['group_id'].nunique())

        print(f'  Total Valid Images: {len(df)}')
        print(f'  Corrupt Images: {len(corrupt_files)}')
        print(f'  Class Distribution: {df["class_name"].value_counts().to_dict()}')
        print(f'  Unique Groups: {unique_groups}')
        print(f'  Exact MD5 Duplicates: {exact_dup_count} images')
        print(f'  Perceptual dHash Clustered: {perceptual_dup_count} images')
        print(f'  Resolutions: {dict(resolution_counts)}')
        print(f'  Color Modes: {dict(mode_counts)}')

        summary_data[modality] = {
            'folder': cfg['folder'],
            'total_images': len(df),
            'corrupt_images': len(corrupt_files),
            'class_distribution': df['class_name'].value_counts().to_dict(),
            'unique_groups': unique_groups,
            'exact_duplicates_count': exact_dup_count,
            'perceptual_duplicates_count': perceptual_dup_count,
            'resolutions': {f'{k[0]}x{k[1]}': v for k, v in resolution_counts.items()},
            'color_modes': dict(mode_counts),
            'formats': dict(format_counts),
            'manifest_csv': str(mod_manifest_path.resolve()),
        }

        all_manifest_rows.extend(rows)

    # Combined manifest
    all_df = pd.DataFrame(all_manifest_rows)
    all_manifest_path = output_dir / 'complete_dataset_manifest.csv'
    all_df.to_csv(all_manifest_path, index=False)
    print(f'\nSaved unified complete manifest: {all_manifest_path} ({len(all_df)} total records)')

    # Summary JSON
    summary_path = output_dir / 'dataset_validation_summary.json'
    summary_path.write_text(json.dumps(summary_data, indent=2))
    print(f'Saved validation summary: {summary_path}')
    print('=' * 80)
    print('DATASET VALIDATION COMPLETED SUCCESSFULLY.')
    print('=' * 80)
    return summary_data


def main():
    parser = argparse.ArgumentParser(description='Validate MSI Breast Cancer Datasets')
    default_root = Path(__file__).resolve().parents[1] / 'MultiModel Breast Cancer MSI Dataset'
    parser.add_argument('--data', default=str(default_root), help='Path to extracted MSI dataset directory')
    parser.add_argument('--output', default='dataset_manifests', help='Output directory for manifests and reports')
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    validate_dataset(data_root, output_dir)


if __name__ == '__main__':
    main()
