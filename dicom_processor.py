"""
DICOM utilities for processing medical imaging studies.
"""
import io
import os
import zipfile
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import pydicom


def has_pixel_data(ds: pydicom.Dataset) -> bool:
    """Check if DICOM dataset has pixel data."""
    return (
        'PixelData' in ds or 
        'FloatPixelData' in ds or 
        'DoubleFloatPixelData' in ds
    )


def extract_dicom_from_zip(zip_bytes: bytes) -> List[Tuple[str, pydicom.Dataset]]:
    """Extract DICOM files from a ZIP archive, filtering out non-image files.
    
    Bug Fix #1: Don't filter by .dcm extension only. Many PACS exports use
    filenames like 'IM000001' or SOP UIDs with no extension. Instead, try to
    parse every file as DICOM, skipping known non-DICOM extensions and directories.
    """
    # Known non-DICOM extensions to skip without attempting parse
    SKIP_EXTENSIONS = {'.txt', '.xml', '.pdf', '.jpg', '.jpeg', '.png', '.gif',
                       '.html', '.htm', '.css', '.js', '.json', '.csv', '.log'}
    dicom_files = []
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
        for filename in zip_ref.namelist():
            # Skip directories
            if filename.endswith('/'):
                continue
            # Skip macOS resource fork junk
            if '__MACOSX/' in filename:
                continue
            # Skip known non-DICOM extensions
            _, ext = os.path.splitext(filename.lower())
            if ext in SKIP_EXTENSIONS:
                continue
            
            try:
                file_bytes = zip_ref.read(filename)
                ds = pydicom.dcmread(io.BytesIO(file_bytes))
                
                # Skip files without pixel data (SR, reports, dose records, etc.)
                if has_pixel_data(ds):
                    dicom_files.append((filename, ds))
                else:
                    print(f"Skipping {filename}: No pixel data (likely SR or report)")
                    
            except Exception as e:
                # Not a valid DICOM file - silently skip
                print(f"Skipping {filename}: not a valid DICOM file ({e})")
    
    return dicom_files


def get_modality(ds: pydicom.Dataset) -> str:
    return getattr(ds, 'Modality', 'Unknown')


def get_study_info(ds: pydicom.Dataset, total_slices: int) -> Dict:
    return {
        'StudyInstanceUID': getattr(ds, 'StudyInstanceUID', 'Unknown'),
        'StudyDescription': getattr(ds, 'StudyDescription', 'Unknown'),
        'Modality': get_modality(ds),
        'TotalSlices': total_slices,
        'StudyDate': getattr(ds, 'StudyDate', 'Unknown'),
        'PatientID': getattr(ds, 'PatientID', 'Unknown'),
    }


def get_default_window(ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
    """Get default window center and width from DICOM metadata."""
    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)

    # Handle multi-valued windows (take first)
    if wc is not None:
        wc = float(wc[0]) if hasattr(wc, '__iter__') and not isinstance(wc, str) else float(wc)
    if ww is not None:
        ww = float(ww[0]) if hasattr(ww, '__iter__') and not isinstance(ww, str) else float(ww)

    return wc, ww


def apply_windowing(
    pixel_array: np.ndarray,
    ds: pydicom.Dataset,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None
) -> np.ndarray:
    """Apply rescale slope/intercept and windowing to pixel array.
    
    Bug Fix #4: RGB/color images skip windowing - just scale to uint8.
    Bug Fix #2: MONOCHROME1 images are inverted after normalization.
    """
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    
    # Bug Fix #4: Color images (RGB, YBR_FULL, YBR_FULL_422) should not
    # have windowing/rescale applied - just ensure uint8 output
    color_interpretations = {'RGB', 'YBR_FULL', 'YBR_FULL_422', 'YBR_RCT', 'YBR_ICT',
                             'PALETTE COLOR'}
    if photometric in color_interpretations:
        if pixel_array.dtype != np.uint8:
            pmin, pmax = pixel_array.min(), pixel_array.max()
            if pmax > pmin:
                pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        return pixel_array
    
    # Apply rescale slope and intercept (converts to HU for CT)
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    pixel_array = pixel_array.astype(np.float32) * slope + intercept

    # Get window values
    if window_center is None or window_width is None:
        default_wc, default_ww = get_default_window(ds)
        if window_center is None:
            window_center = default_wc
        if window_width is None:
            window_width = default_ww

    # Apply windowing if we have valid values
    if window_center is not None and window_width is not None and window_width > 0:
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, min_val, max_val)
        normalized = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        # Fallback: normalize to full range
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(pixel_array, dtype=np.uint8)

    # Bug Fix #2: MONOCHROME1 means high pixel values = dark (inverted)
    # After normalization, invert so display is correct
    if photometric == 'MONOCHROME1':
        normalized = 255 - normalized

    return normalized


def dicom_to_pil(
    ds: pydicom.Dataset,
    size: Tuple[int, int] = (896, 896),
    window_center: Optional[float] = None,
    window_width: Optional[float] = None
) -> Image.Image:
    """Convert DICOM dataset to PIL Image with optional windowing and resizing.
    
    Bug Fix #3: Preserve aspect ratio instead of forcing square resize.
    Resize so the longest dimension fits `size`, then pad with black to
    make the output square (for model compatibility).
    """
    pixel_array = ds.pixel_array
    normalized = apply_windowing(pixel_array, ds, window_center, window_width)

    if len(normalized.shape) == 2:
        pil_image = Image.fromarray(normalized, mode='L')
    elif len(normalized.shape) == 3 and normalized.shape[2] <= 4:
        if normalized.shape[2] == 1:
            pil_image = Image.fromarray(normalized[:, :, 0], mode='L')
        elif normalized.shape[2] == 3:
            pil_image = Image.fromarray(normalized, mode='RGB')
        elif normalized.shape[2] == 4:
            pil_image = Image.fromarray(normalized[:, :, :3], mode='RGB')
        else:
            pil_image = Image.fromarray(normalized[:, :, 0], mode='L')
    else:
        pil_image = Image.fromarray(normalized[0], mode='L')

    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Bug Fix #3: Preserve aspect ratio, then pad to square
    target_w, target_h = size
    orig_w, orig_h = pil_image.size
    
    # Calculate scale to fit longest dimension
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    
    # Pad with black to make square
    if new_w != target_w or new_h != target_h:
        padded = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(pil_image, (paste_x, paste_y))
        pil_image = padded

    return pil_image


def organize_by_series(dicom_files: List[Tuple[str, pydicom.Dataset]]) -> Dict[str, List[Tuple[str, pydicom.Dataset]]]:
    series_dict = {}
    for filename, ds in dicom_files:
        series_uid = getattr(ds, 'SeriesInstanceUID', 'Unknown')
        if series_uid not in series_dict:
            series_dict[series_uid] = []
        series_dict[series_uid].append((filename, ds))
    return series_dict


def sort_slices_by_position(series_files: List[Tuple[str, pydicom.Dataset]]) -> List[Tuple[str, pydicom.Dataset]]:
    def get_sort_key(item):
        filename, ds = item
        instance_num = getattr(ds, 'InstanceNumber', None)
        if instance_num is not None:
            return (0, int(instance_num))
        
        slice_loc = getattr(ds, 'SliceLocation', None)
        if slice_loc is not None:
            return (1, float(slice_loc))
        
        return (2, filename)
    
    return sorted(series_files, key=get_sort_key)


def sample_slices_evenly(all_slices: List[Tuple[str, pydicom.Dataset]], max_slices: int = 500) -> List[Tuple[str, pydicom.Dataset]]:
    if len(all_slices) <= max_slices:
        return all_slices
    
    indices = [int(i * (len(all_slices) - 1) / (max_slices - 1)) for i in range(max_slices)]
    return [all_slices[i] for i in indices]


def process_dicom_study(
    zip_bytes: bytes,
    max_slices: int = 500,
    max_slices_per_series: Optional[int] = None,
    max_total_images: Optional[int] = None,
    image_size: int = 896,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None
) -> Tuple[str, List[Image.Image], Dict]:
    """
    Process a DICOM study from a ZIP file.

    Args:
        zip_bytes: ZIP file contents
        max_slices: Maximum total slices across all series (used if max_slices_per_series is None)
        max_slices_per_series: If set, sample this many slices evenly from each series
        max_total_images: Hard cap on total images sent to model (applied AFTER per-series sampling)
        image_size: Output image size (square, e.g., 896 for 896x896)
        window_center: Window center for display (None = use DICOM default or auto)
        window_width: Window width for display (None = use DICOM default or auto)
    """
    dicom_files = extract_dicom_from_zip(zip_bytes)

    if not dicom_files:
        raise ValueError("No valid DICOM files found in the ZIP archive")

    first_ds = dicom_files[0][1]
    modality = get_modality(first_ds)

    # Get default window from first image
    default_wc, default_ww = get_default_window(first_ds)

    series_dict = organize_by_series(dicom_files)

    # Count total original slices
    total_original_slices = sum(len(files) for files in series_dict.values())

    # Sample slices per series or globally
    sampled_slices = []
    if max_slices_per_series is not None:
        # Sample evenly from each series
        for series_uid, series_files in series_dict.items():
            sorted_slices = sort_slices_by_position(series_files)
            series_sampled = sample_slices_evenly(sorted_slices, max_slices_per_series)
            sampled_slices.extend(series_sampled)
    else:
        # Original behavior: sample globally
        all_sorted_slices = []
        for series_uid, series_files in series_dict.items():
            sorted_slices = sort_slices_by_position(series_files)
            all_sorted_slices.extend(sorted_slices)
        sampled_slices = sample_slices_evenly(all_sorted_slices, max_slices)

    # Apply hard total cap (important for multi-series studies on low VRAM)
    if max_total_images is not None and len(sampled_slices) > max_total_images:
        print(f"⚠️  Capping total images from {len(sampled_slices)} to {max_total_images}")
        sampled_slices = sample_slices_evenly(sampled_slices, max_total_images)

    sampled_count = len(sampled_slices)

    study_info = get_study_info(first_ds, sampled_count)
    study_info['SeriesCount'] = len(series_dict)
    study_info['TotalOriginalSlices'] = total_original_slices
    study_info['SampledSlices'] = sampled_count
    study_info['ImageSize'] = image_size
    study_info['DefaultWindowCenter'] = default_wc
    study_info['DefaultWindowWidth'] = default_ww
    if max_slices_per_series is not None:
        study_info['MaxSlicesPerSeries'] = max_slices_per_series

    images = []
    for filename, ds in sampled_slices:
        try:
            pil_image = dicom_to_pil(
                ds,
                size=(image_size, image_size),
                window_center=window_center,
                window_width=window_width
            )
            images.append(pil_image)
        except Exception as e:
            print(f"Error converting {filename}: {e}")

    study_info['ProcessedImages'] = len(images)

    return modality, images, study_info
