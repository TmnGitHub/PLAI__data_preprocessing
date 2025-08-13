import os
import zipfile
import tempfile
import shutil

from tabulate import tabulate
import SimpleITK as sitk

import csv
import pydicom
from pydicom.tag import Tag

from collections import defaultdict
import numpy as np
import sys

import matplotlib.pyplot as plt
import re
from IPython.display import clear_output

from dicomanonymizer import anonymize_dicom_file

import json
import uuid

import vtkplotlib as vpl
from stl.mesh import Mesh

import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
       
import nibabel as nib

import gc

import itk
import vtk
from pydicom.uid import UncompressedPixelTransferSyntaxes

from collections import defaultdict

from pydicom import dcmread

import pyvista as pv
from pyvista import examples

import logging
_logger_registry = {}

def keep_log(log_file_path, message, level=logging.INFO, exc_info=False):
    """Writes a log entry, creating directories if needed."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # ✅ ensure log folder exists

    logger_name = os.path.abspath(log_file_path)

    if logger_name not in _logger_registry:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(log_file_path)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

        _logger_registry[logger_name] = logger
    else:
        logger = _logger_registry[logger_name]

    logger.log(level, message, exc_info=exc_info)

def unzip_file(zip_file_path, dir_path):
    """
    Unzips a file to a folder named after the zip file inside the specified directory.
    
    :param zip_file_path: Path to the zip file
    :param dir_path: Directory where the extracted folder should be created
    """
    if not zip_file_path.lower().endswith('.zip'):
        raise ValueError("Provided file is not a zip archive")
    
    zip_filename = os.path.splitext(os.path.basename(zip_file_path))[0]
    extract_dir = os.path.join(dir_path, zip_filename)
    
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"Extracted to: {extract_dir}")

def unzip_all_in_directory(root_dir):
    """
    Recursively finds and extracts all zip files inside the root directory.
    
    :param root_dir: The root directory to search for zip files
    """
    for folder_path, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.zip'):
                try:
                    zip_file_path = os.path.join(folder_path, file)
                    unzip_file(zip_file_path, folder_path)  # Extract in the same branch where found
                except Exception as e:
                    keep_log("./logs/error_unzipping.log", f"Error unzipping {zip_file_path}: {e}", logging.ERROR, exc_info=True)
                    continue

def unzip_recursively(input_path, final_output_dir):
    """
    Unzip given path (zip file or directory) and recursively extract nested zips.
    Final output is stored in save_path/(base_name of input_path).
    """
    os.makedirs(final_output_dir, exist_ok=True)
    temp_dir = os.path.join(os.path.dirname(final_output_dir), "temp")
    if input_path.lower().endswith(".zip"):
        os.makedirs(temp_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except Exception as e:
            print(f"Error: {e}")
            return
    else:
        shutil.copytree(input_path, temp_dir)
        
    unzip_all_in_directory(temp_dir)
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
    if input_path.lower().endswith(".zip"):
        shutil.copytree(temp_dir, os.path.dirname(final_output_dir))
    else:
        shutil.copytree(temp_dir, final_output_dir)
    shutil.rmtree(temp_dir)
  
def get_stl_files(unzipped_directory_path):
    for dirpath, dirname, filename in os.walk(unzipped_directory_path, topdown=True):
        stl_files = []
        for file in filename:
            if file.lower().endswith(".stl"):
                if "hanche" or "tibia" or "femur" or "cheville" or "fibula" in file:
                    file_path = os.path.join(dirpath, file)
                    stl_files.append(file_path)
        if stl_files != []:
            break
    return stl_files

def get_dicom_files(series_path):
    dicom_files = []
    for f in sorted(os.listdir(series_path)):
        file_path = os.path.join(series_path, f)
        if os.path.isfile(file_path) and not f.startswith('.'):
            try:
                # Try reading as DICOM file (header only)
                pydicom.dcmread(file_path, stop_before_pixels=True)
                dicom_files.append(file_path)
            except Exception:
                # Skip non-DICOM or corrupted files
                pass
    return dicom_files

def append_middle_dicom_tags_to_csv(csv_file_path, series_path):
    # List DICOM files in the series directory
    dicom_files = get_dicom_files(series_path)

    if not dicom_files:
        print(f"No DICOM files found in series: {series_path}")
        return

    # Get middle file
    middle_index = len(dicom_files) // 2
    middle_file = dicom_files[middle_index]

    # Load DICOM file
    ds = pydicom.dcmread(middle_file, stop_before_pixels=True)

    # Initialize row with patient and series info
    row = {
        "Patient_dir": os.path.basename(os.path.dirname(series_path)),
        "series_id": os.path.basename(series_path)
    }

    # Extract DICOM tags safely
    for elem in ds.iterall():
        if elem.VR == 'OB' or isinstance(elem.value, bytes):
            # Skip binary values
            continue

        # Generate unique, readable column name
        tag_key = elem.keyword if elem.keyword else f"{elem.tag}"
        if tag_key in row:
            tag_key = f"{elem.tag}"  # fallback to tag address if duplicate

        row[tag_key] = str(elem.value)

    # Write or append to CSV
    write_header = not os.path.exists(csv_file_path)
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def is_dicom_file(dcm_file_path):
    try:
        pydicom.dcmread(dcm_file_path, stop_before_pixels=True)
        return True
    except Exception:
        return False


def organize_dicom_series(root_directory_path, save_root_path):
    # Ensure the save root path exists
    os.makedirs(save_root_path, exist_ok=True)

    # Walk through the root directory
    for dirpath, _, filenames in os.walk(root_directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                # Load DICOM file
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                # Extract necessary fields
                series_description = getattr(ds, 'SeriesDescription', 'UnknownSeries')
                acquisition_number = getattr(ds, 'AcquisitionNumber', 'UnknownAcq')

                # Clean series description to be safe for directory names
                safe_series_description = "_".join(series_description.split())

                # Create directory name
                dir_name = f"{safe_series_description}_{acquisition_number}"

                # Find the relative path from the root_directory_path
                relative_path = os.path.relpath(dirpath, root_directory_path)

                # Create full destination path, maintaining hierarchy
                destination_dir = os.path.join(save_root_path, relative_path)

                # Ensure the destination directory exists
                os.makedirs(destination_dir, exist_ok=True)

                # Create a subdirectory inside the destination directory for the series
                series_dir = os.path.join(destination_dir, dir_name)
                os.makedirs(series_dir, exist_ok=True)

                # Handle potential file name collisions
                destination_file_path = os.path.join(series_dir, filename)
                if os.path.exists(destination_file_path):
                    # If file exists, add a unique suffix
                    name, ext = os.path.splitext(filename)
                    unique_name = f"{name}_{uuid.uuid4().hex}{ext}"
                    destination_file_path = os.path.join(series_dir, unique_name)

                # Copy the file to the new directory
                shutil.copy2(file_path, destination_file_path)

            except Exception as e:
                keep_log("./logs/organize_dicom_series.log", f"Error organizing dicom series: {e}", logging.ERROR, exc_info=True)
                continue

def get_shape(series_path):
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        size = image.GetSize()
        dimension = size[:2]
        count = size[2]
        return dimension, count
    except:
        return None, 1

def anonymize_dicom_series(series_path):
    dicom_files = [
        os.path.join(series_path, f)
        for f in os.listdir(series_path)
        if is_dicom_file(os.path.join(series_path, f))
    ]

    for dicom_file in dicom_files:
        try:
            anonymize_dicom_file(dicom_file, dicom_file)
        except Exception as e:
            print(f"Failed anonymization: {dicom_file}; {e}")

    print(f"Anonymized {len(dicom_files)} DICOM files in {series_path}")


def visualize_nifti_middle_slices_sitk(image_path):
    image = sitk.ReadImage(image_path)
    print(f"Image Size: {image.GetSize()}")
    image_array = sitk.GetArrayFromImage(image)  # [z, y, x]

    z, y, x = image_array.shape

    # Extract sagittal slices (X-axis)
    start_middle_sagittal = image_array[:, :, 2 * (x // 2) // 3]
    middle_sagittal = image_array[:, :, x // 2]
    end_middle_sagittal = image_array[:, :, 3 * (x // 2) // 2]

    # Extract coronal slices (Y-axis)
    start_middle_coronal = image_array[:, 2 * (y // 2) // 3, :]
    middle_coronal = image_array[:, y // 2, :]
    end_middle_coronal = image_array[:, 3 * (y // 2) // 2, :]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    titles = [
        'Start-Middle Sagittal', 'Middle Sagittal', 'End-Middle Sagittal',
        'Start-Middle Coronal', 'Middle Coronal', 'End-Middle Coronal'
    ]
    
    sagittal_slices = [start_middle_sagittal, middle_sagittal, end_middle_sagittal]
    coronal_slices = [start_middle_coronal, middle_coronal, end_middle_coronal]

    # Axis ticks
    z_ticks = np.arange(0, z, 50)
    x_ticks = np.arange(0, x, 50)
    y_ticks = np.arange(0, y, 50)

    # Plot sagittal slices
    for i, slice_img in enumerate(sagittal_slices):
        axes[0, i].imshow(slice_img, cmap='gray', origin='upper', aspect='auto')
        axes[0, i].set_title(titles[i])
        axes[0, i].set_ylabel('Z-axis (Height)')
        axes[0, i].set_xlabel('Y-axis')
        axes[0, i].set_yticks(z_ticks)
        axes[0, i].set_xticks(y_ticks)

    # Plot coronal slices
    for i, slice_img in enumerate(coronal_slices):
        axes[1, i].imshow(slice_img, cmap='gray', origin='upper', aspect='auto')
        axes[1, i].set_title(titles[i + 3])
        axes[1, i].set_ylabel('Z-axis (Height)')
        axes[1, i].set_xlabel('X-axis')
        axes[1, i].set_yticks(z_ticks)
        axes[1, i].set_xticks(x_ticks)

    plt.tight_layout()
    plt.show()

def show_three_coronal_slices(image_path, mask_path):
    # Read image and mask
    img = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # Convert to numpy arrays
    img_np = sitk.GetArrayFromImage(img)   # (z, y, x)
    mask_np = sitk.GetArrayFromImage(mask) > 0

    # Get coronal view (swap axes: coronal slices are along axis=1 in (z, y, x))
    img_coronal = np.transpose(img_np, (1, 0, 2))   # shape: (y, z, x)
    mask_coronal = np.transpose(mask_np, (1, 0, 2))

    # Find first and last slice with mask
    nonzero_slices = np.where(mask_coronal.any(axis=(1, 2)))[0]
    if len(nonzero_slices) == 0:
        print("Mask is empty — nothing to display.")
        return

    start_slice = nonzero_slices[0]
    end_slice = nonzero_slices[-1]
    mid_slice = (start_slice + end_slice) // 2
    offset = int(0.2 * (end_slice - start_slice))

    slices_to_show = [mid_slice - offset, mid_slice, mid_slice + offset]

    # Plot three slices side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, slice_idx in zip(axes, slices_to_show):
        ax.imshow(img_coronal[slice_idx], cmap='gray')
        ax.imshow(mask_coronal[slice_idx], cmap='Reds', alpha=0.5)
        ax.set_title(f"Slice {slice_idx}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def inclined_split_axial_nifti(image, output_dir, region, upper_percent=50, lower_percent=50):
    """
    Splits a NIfTI image along the axial x-axis with an inclined line defined by upper_percent (top row) and lower_percent (bottom row).
    
    Parameters:
    - image: loaded nifti image
    - output_dir: Directory to save outputs
    - region: Region name, to manage hierarchy on save path
    - upper_percent: Split percentage at top of image (0–100)
    - lower_percent: Split percentage at bottom of image (0–100)
    """
    assert 0 <= upper_percent <= 100 and 0 <= lower_percent <= 100, "Percent values must be between 0 and 100"

    # Load image and convert to numpy
    # image = sitk.ReadImage(nifti_path)
    array = sitk.GetArrayFromImage(image)  # (z, y, x)
    z, y, x = array.shape

    # Compute split column per row (y)
    x_coords = np.linspace(upper_percent / 100 * x, lower_percent / 100 * x, y).astype(int)  # shape: (y,)

    # Create masks
    left_array = np.zeros_like(array)
    right_array = np.zeros_like(array)

    for z_idx in range(z):
        for y_idx in range(y):
            split_col = x_coords[y_idx]
            left_array[z_idx, y_idx, :split_col] = array[z_idx, y_idx, :split_col]
            right_array[z_idx, y_idx, split_col:] = array[z_idx, y_idx, split_col:]

    # Convert to images and preserve metadata
    left_image = sitk.GetImageFromArray(left_array)
    right_image = sitk.GetImageFromArray(right_array)
    left_image.CopyInformation(image)
    right_image.CopyInformation(image)

    left_region = os.path.join(output_dir, f"left_{region}")
    right_region = os.path.join(output_dir, f"right_{region}")
    os.makedirs(left_region, exist_ok=True)
    os.makedirs(right_region, exist_ok=True)

    left_path = os.path.join(left_region, f"{region}_image.nii.gz")
    right_path = os.path.join(right_region, f"{region}_image.nii.gz")

    # Save
    sitk.WriteImage(left_image, left_path)
    sitk.WriteImage(right_image, right_path)

    print(f"Saved inclined split images:\n  {left_path}\n  {right_path}")

def extract_axial_slice_range(nifti_path, output_dir, region, start_slice, end_slice):
    """
    Extracts and saves axial slices from start_slice to end_slice (inclusive) into a new NIfTI file.

    Parameters:
    - nifti_path: Path to the input NIfTI image.
    - output_dir: Directory to save the extracted image.
    - region: Region name, to manage hierarchy on save path
    - start_slice: Starting Z-index (inclusive).
    - end_slice: Ending Z-index (inclusive).
    """
    image = sitk.ReadImage(nifti_path)
    series_path = os.path.dirname(nifti_path)
    array = sitk.GetArrayFromImage(image)  # Shape: (z, y, x)
    z = array.shape[0]

    # Validate slice range
    assert 0 <= start_slice <= end_slice < z, f"Slice indices must be within [0, {z-1}]"

    # Extract subvolume
    extracted_array = array[start_slice:end_slice + 1]

    # Convert back to SimpleITK image
    extracted_image = sitk.GetImageFromArray(extracted_array)
    
    # Adjust origin to reflect new slice position
    original_origin = list(image.GetOrigin())
    spacing = image.GetSpacing()
    new_origin = original_origin
    new_origin[2] += start_slice * spacing[2]
    extracted_image.SetOrigin(tuple(new_origin))
    
    extracted_image.SetSpacing(image.GetSpacing())
    extracted_image.SetDirection(image.GetDirection())

    re_input = True
    while re_input:
        input_str = input("Enter upper and lower percentage from the approximation for the visualization given below: ")
        upper_percent, lower_percent = map(int, input_str.split(','))
        if upper_percent >=100 or lower_percent>=100:
            re_input = True
        else:
            re_input = False

    if upper_percent ==0 and lower_percent == 0:
        images = "unilateral"
        output_save_dir = os.path.join(output_dir, f"{images}_region")
        os.makedirs(output_save_dir, exist_ok=True)
        output_path = os.path.join(output_save_dir, f"{region}_image.nii.gz")
        sitk.WriteImage(extracted_image, output_path)
        return "unilateral"
    else:
        inclined_split_axial_nifti(extracted_image, output_dir, region, upper_percent, lower_percent)
        return "bilateral"

    # print(f"Extracted axial slices {start_slice} to {end_slice} and saved to:\n  {output_path}")
    
def extract_axial_slice_range_in_mask(nifti_path, output_dir, region, start_slice, end_slice):
    """

    Starting from 0 to (N)th z slice to take the slices for particular region

    eg: if an image contains both knee and ankle then consider slice 0-340 is knee and slice 340-510 is ankle

    if split valie is 0, 0 then it considers image as unilateral image and processes the image accordingly

    Extracts and saves axial slices from start_slice to end_slice (inclusive) into a new NIfTI file.

    Parameters:
    - nifti_path: Path to the input NIfTI image.
    - output_dir: Directory to save the extracted image.
    - region: Region name, to manage hierarchy on save path
    - start_slice: Starting Z-index (inclusive).
    - end_slice: Ending Z-index (inclusive).
    """
    image = sitk.ReadImage(nifti_path)
    series_path = os.path.dirname(nifti_path)
    array = sitk.GetArrayFromImage(image)  # Shape: (z, y, x)
    z = array.shape[0]

    # Validate slice range
    assert 0 <= start_slice <= end_slice < z, f"Slice indices must be within [0, {z-1}]"

    # Extract subvolume
    extracted_array = array[start_slice:end_slice + 1]

    # Convert back to SimpleITK image
    extracted_image = sitk.GetImageFromArray(extracted_array)
    
    # Adjust origin to reflect new slice position
    original_origin = list(image.GetOrigin())
    spacing = image.GetSpacing()
    new_origin = original_origin
    new_origin[2] += start_slice * spacing[2]
    extracted_image.SetOrigin(tuple(new_origin))
    
    extracted_image.SetSpacing(image.GetSpacing())
    extracted_image.SetDirection(image.GetDirection())

    return extracted_image


def load_dicom_series(dicom_folder):
    """Load a DICOM series from a folder using SimpleITK."""
    reader = sitk.ImageSeriesReader()
    dicom_filenames = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_filenames)
    image = reader.Execute()
    return image

def process_dicom_folder(dicom_folder, output_folder):
    """Load DICOM, process to zero left/right side, and save as .nii.gz."""
    # Load the DICOM series from the folder
    acc_image = load_dicom_series(dicom_folder)
    
    # Save the results as .nii.gz
    acc_image_path = os.path.join(output_folder, f"{os.path.basename(output_folder)}_image.nii.gz")
    
    sitk.WriteImage(acc_image, acc_image_path)

    print(f"Saved: {acc_image_path}")


def compare_mask_alignment(mask, img1_path, img2_path, tol=1e-3):
    img1 = sitk.ReadImage(img1_path)
    img2 = sitk.ReadImage(img2_path)

    def resample_to_image(src, ref):
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(ref)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetTransform(sitk.Transform())
        return resample.Execute(src)

    def overlap_fraction(mask_img, ref_img):
        mask_np = sitk.GetArrayFromImage(mask_img) > 0
        ref_np = sitk.GetArrayFromImage(ref_img) > 0  # binary presence map
        overlap = np.logical_and(mask_np, ref_np)
        return np.sum(overlap) / np.sum(mask_np)

    # Resample mask into each image's space
    mask_resampled1 = resample_to_image(mask, img1)
    mask_resampled2 = resample_to_image(mask, img2)

    # Compute overlap fraction
    score1 = overlap_fraction(mask_resampled1, img1)
    score2 = overlap_fraction(mask_resampled2, img2)

    print(f"Score1: {score1:.4f}, Score2: {score2:.4f}")

    if score1 > score2:
        return "left"
    elif score2 > score1:
        return "right"

def manage_hierarchy_for_mask(series_save_path, extracted_mask, region, input_region, better):
    if better == "left":
        mask_save_path = os.path.join(series_save_path, f"left_{region}", f"{input_region}_label.nii.gz")
        index = 1
        while os.path.exists(mask_save_path):
            mask_save_path = os.path.join(series_save_path, f"left_{region}", f"{input_region}_label_{index}.nii.gz")
            index += 1
        sitk.WriteImage(extracted_mask, mask_save_path)
    if better == "right":
        mask_save_path = os.path.join(series_save_path, f"right_{region}", f"{input_region}_label.nii.gz")
        index = 1
        while os.path.exists(mask_save_path):
            mask_save_path = os.path.join(series_save_path, f"right_{region}", f"{input_region}_label_{index}.nii.gz")
            index += 1
        sitk.WriteImage(extracted_mask, mask_save_path)
    clear_output(wait=True)

def check_mha_nonzero(mask_data, file_name):
    try: 
        if np.any(mask_data > 0):
            # print(f"The {file_name} mask contains non-zero pixels (something is present).")
            contains = True
        else:
            # print(f"The {file_name} mask is completely black (all zeros).")
            contains = False
        return contains
    except:
        print(f"Failed checking for non-zero pixels for {file_name}")

def stl_to_dicom_mask(dicom_dir, stl_path):
    """
    Maps an STL file onto a DICOM series and returns a binary mask as a SimpleITK image.

    Parameters:
    dicom_dir (str): Path to folder containing DICOM series.
    stl_path (str): Path to the STL file.

    Returns:
    SimpleITK.Image: Binary mask image aligned with the DICOM series.
    """

    # Step 1: Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    size = image.GetSize()

    # Step 2: Convert SimpleITK image to VTK image
    image_vtk = vtk.vtkImageData()
    image_vtk.SetSpacing(spacing)
    image_vtk.SetOrigin(origin)
    extent = (0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    image_vtk.SetExtent(extent)

    # Step 3: Read STL file
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_path)
    stl_reader.Update()
    polydata = stl_reader.GetOutput()

    # Step 4: Create stencil from STL
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(image_vtk.GetExtent())
    pol2stenc.Update()

    # Step 5: Convert stencil to binary image
    imgstenc = vtk.vtkImageStencilToImage()
    imgstenc.SetInputConnection(pol2stenc.GetOutputPort())
    imgstenc.SetOutsideValue(0)
    imgstenc.SetInsideValue(1)
    imgstenc.SetOutputScalarTypeToUnsignedChar()
    imgstenc.Update()

    # Step 6: Convert VTK image to numpy
    vtk_mask = imgstenc.GetOutput()
    dims = vtk_mask.GetDimensions()
    scalars = vtk_mask.GetPointData().GetScalars()
    np_mask = np.frombuffer(scalars, dtype=np.uint8)
    np_mask = np_mask.reshape((dims[2], dims[1], dims[0]))

    # Step 7: Convert numpy mask to SimpleITK
    mask_image = sitk.GetImageFromArray(np_mask)
    mask_image.SetSpacing(spacing)
    mask_image.SetOrigin(origin)
    mask_image.SetDirection(direction)

    return mask_image

def transform_mask(mask_image):
    mask_array = sitk.GetArrayFromImage(mask_image)
    rotated_mask = np.rot90(mask_array, k=2)
    flipped_mask = np.fliplr(rotated_mask)
    transformed_mask = sitk.GetImageFromArray(flipped_mask)
    transformed_mask.CopyInformation(mask_image)
    return transformed_mask

def get_mask_from_stl(dicom_dir, nifti_save_path, stl_files_list):
    for stl_file_path in stl_files_list:
        mask_image = stl_to_dicom_mask(dicom_dir, stl_file_path)
        
        transformed_mask_image = transform_mask(mask_image)
        # transformed_mask_image = mask_image

        mask_image_array = sitk.GetArrayFromImage(transformed_mask_image)


        stl_file = os.path.basename(stl_file_path)
        has_mask = check_mha_nonzero(mask_image_array, stl_file)
        if has_mask:
            sitk.WriteImage(transformed_mask_image, f"{nifti_save_path}/{stl_file.replace(".stl", ".nii.gz")}")
            print(f"{stl_file} mask image saved successfully.")
        else:
            print(f"The {stl_file} mask image is zero.")


def get_nifti_images_and_mask_from_managed_patient_dir(root_path, processed_save_path):
    stl_files = get_stl_files(root_path)

    for dirpath, dirnames, filenames in os.walk(root_path):
        dicom_files = []
        # dicom_files_path = []
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=False)
                dicom_files.append(ds)
                # dicom_files_path.append(file_path)
            except Exception as e:
                # Not a valid DICOM file
                continue
        Series_description = os.path.basename(dirpath)

        # try:
            # print(f"Processing DICOM files in {dirpath}")
        if len(dicom_files) > 25:
            print(f"{len(dicom_files)} files from {dirpath}")

            valid_dicom_files = []
            for ds in dicom_files:
                if ds.file_meta.TransferSyntaxUID in UncompressedPixelTransferSyntaxes:
                    valid_dicom_files.append(ds)
                else:
                    print(f"Skipping compressed DICOM: {ds.SOPInstanceUID}")

            visualize_middle_slice(dicom_files, Series_description)
        else:
            if len(dicom_files) > 0:
                keep_log("./logs/nifti_conversion.log", f"Not enough DICOM files in {dirpath} to visualize.", logging.WARNING, exc_info=True)
                continue
        # except:
        #     continue

        if dicom_files:
            leg_count = "unilateral" if (inp := user_input("Enter u/b/i for unilateral and bilateral leg from the visualization below: ").lower()) == "u" else "bilateral" if inp == "b" else "invalid" if inp == "i" else None
            part = "ankle" if (inp := user_input("Enter a/p/k/i for ankle, pelvis and knee region from the visualization below:").lower()) == "a" else "pelvis" if inp == "p" else "knee" if inp == "k" else "invalid" if inp == "i" else None
            
            if leg_count == "invalid" and part == "invalid":
                invalid_image_json_path = "/media/interns/storage/PLAI/preprocessing_code/json_files/invalid_image_information_on_preprocessing.json"
                message_to_write = {
                    "patient_id" : os.path.basename(root_path),
                    "series_path" : dirpath
                }
                write_to_json(invalid_image_json_path, message_to_write)
                continue

            parts = dirpath.split(os.sep)

            # Get the desired folder name (3rd last element)
            target_folder = parts[-2]

            nifti_save_path =os.path.join(processed_save_path, os.path.basename(root_path), target_folder, f"{leg_count}_{part}")

            i = 0
            while os.path.exists(nifti_save_path):
                i +=1
                nifti_save_path = os.path.join(processed_save_path, os.path.basename(root_path), target_folder, f"{leg_count}_{part}_{i}")
            os.makedirs(nifti_save_path, exist_ok=True)
            
            if leg_count== "unilateral":
                process_dicom_folder(dirpath, nifti_save_path)
            elif leg_count == "bilateral":
                process_double_leg_dicom_folder(dirpath, nifti_save_path)
                
            # get_mask_from_stl(dirpath, nifti_save_path, stl_files)

