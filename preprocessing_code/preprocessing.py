import os
import shutil
       
import json
import numpy as np

import vtkplotlib as vpl
from stl.mesh import Mesh

import tempfile
import zipfile

import logging
import re
from IPython.display import clear_output

import pydicom
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
       
import json

import nibabel as nib

import SimpleITK as sitk

import gc
import tempfile

import itk
import vtk

from collections import defaultdict

from pydicom import dcmread
from pydicom.tag import Tag

import pyvista as pv
from pyvista import examples


# Set up logging
import logging
logging.basicConfig(filename="preprocessing_2021.log", level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def write_to_json(json_file_path, message_to_write):
    # If file exists, load existing data
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new message
    data.append(message_to_write)

    # Write back to file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

def user_input(prompt):
    value = input(prompt)
    match = re.search(r'Enter (.*?) (?:for|from)', prompt, flags=re.IGNORECASE)
    if match:
        keys_string = match.group(1)
        prompt_keys = keys_string.split('/')
    else:
        prompt_keys = []
    if value.lower() in [key.lower() for key in prompt_keys]:
        clear_output(wait=True)
        return value
    else:
        print("Invalid input. Try again.\n")
        clear_output(wait=True)
        return user_input(prompt)
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#For handling all the zip files in patirnt directory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                    logging.error(f"Error unzipping {zip_file_path}: {e}", exc_info=True)
                    continue

def zip_directory(root_dir, save_dir):
    """
    Zips the contents of the root_dir and saves the zip file to save_dir.

    Parameters:
    root_dir (str): Path to the directory to zip.
    save_dir (str): Path where the zipped file should be saved (including .zip filename).
    """
    # Ensure the save_dir ends with '.zip'
    if not save_dir.endswith('.zip'):
        raise ValueError("save_dir must end with '.zip'")

    with zipfile.ZipFile(save_dir, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                # Arcname ensures the zip has relative paths, not absolute ones
                arcname = os.path.relpath(file_path, start=root_dir)
                zipf.write(file_path, arcname)

    print(f"Directory '{root_dir}' zipped successfully into '{save_dir}'.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Series division based on acquisition number and series descripttion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def is_dicom_file(filepath):
    """Check if a file is a valid DICOM file."""
    try:
        # pydicom.dcmread(filepath, stop_before_pixels=True)
        # return True
        with open(filepath, "rb") as f:
            # DICOM files start with "DICM" at byte 128
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
    except Exception as e:
        # print("is_dicom_file")
        logging.error(f"Error in is_dicom_file {filepath}: {e}", exc_info=True)
        return False

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
        
def reorient(save_path, nii_path = None, nii = None):
    if nii_path is not None:
        nii = sitk.ReadImage(nii_path)
    if nii == None and nii_path == None:
        print(f"Need either nii path or loaded nii image")
    # Flip the image vertically (along the Y-axis, which corresponds to the 1st dimension)
    flipped_nii = sitk.Flip(nii, [False, False, True])  # [False, False, True] flips along the Z-axis
    # Save the flipped image to the specified save path
    sitk.WriteImage(flipped_nii, save_path)

       
def load_dicom_series(dicom_folder):
    """Load a DICOM series from a folder using SimpleITK."""
    reader = sitk.ImageSeriesReader()
    dicom_filenames = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_filenames)
    image = reader.Execute()
    return image

# def process_image_set_sides_zero(sitk_image):
#     """Set the right and left side of the image to zero and return two separate images."""
#     # Convert the SimpleITK image to a NumPy array
#     image_np = sitk.GetArrayFromImage(sitk_image)  # Shape: [slices, height, width]
#     acc_image = sitk.GetImageFromArray(image_np)
#     acc_image.CopyInformation(sitk_image)

#     return acc_image

def process_double_leg_image_set_sides_zero(sitk_image):
    """Set the right and left side of the image to zero and return two separate images."""
    # Convert the SimpleITK image to a NumPy array
    image_np = sitk.GetArrayFromImage(sitk_image)  # Shape: [slices, height, width]
    
    # Get image dimensions
    num_slices, height, width = image_np.shape
    
    # # Split the image into left and right halves
    middle = width // 2
    
    # # Create a copy of the original image and set the right side to zero
    right_zero_image_np = np.copy(image_np)
    right_zero_image_np[:, :, middle:] = -1000  # Set the right half to zero
    
    # # Create a copy of the original image and set the left side to zero
    left_zero_image_np = np.copy(image_np)
    left_zero_image_np[:, :, :middle] = -1000  # Set the left half to zero
    
    # Convert the NumPy arrays back to SimpleITK images
    right_zero_image = sitk.GetImageFromArray(right_zero_image_np)
    left_zero_image = sitk.GetImageFromArray(left_zero_image_np)
    
    # Preserve the original metadata (spacing, origin, direction)
    left_zero_image.CopyInformation(sitk_image)
    right_zero_image.CopyInformation(sitk_image)
    
    # return right_zero_image, left_zero_image
    return right_zero_image, left_zero_image

def save_as_nii_gz(sitk_image, output_path):
    """Save the given SimpleITK image as a NIfTI .nii.gz file."""
    reorient(save_path = output_path, nii = sitk_image)
    # sitk.WriteImage(sitk_image, output_path)

def process_dicom_folder(dicom_folder, output_folder):
    """Load DICOM, process to zero left/right side, and save as .nii.gz."""
    # Load the DICOM series from the folder
    acc_image = load_dicom_series(dicom_folder)
    
    # Process the image to set left and right side to zero
    # right_zero_image, left_zero_image = process_image_set_sides_zero(dicom_image)
    # acc_image = process_image_set_sides_zero(dicom_image)
    # acc_image = dicom_image.CopyInformation(dicom_image)
    
    # Save the results as .nii.gz
    acc_image_path = os.path.join(output_folder, f"{os.path.basename(output_folder)}_image.nii.gz")
    # left_zero_output = os.path.join(output_folder, "left_side.nii.gz")
    
    save_as_nii_gz(acc_image, acc_image_path)
    # save_as_nii_gz(left_zero_image, left_zero_output)

    print(f"Saved: {acc_image_path}")
    # print(f"Saved: {right_zero_output} and {left_zero_output}")


def process_double_leg_dicom_folder(dicom_folder, output_folder):
    """Load DICOM, process to zero left/right side, and save as .nii.gz."""
    # Load the DICOM series from the folder
    dicom_image = load_dicom_series(dicom_folder)
    
    # Process the image to set left and right side to zero
    right_zero_image, left_zero_image = process_double_leg_image_set_sides_zero(dicom_image)
    
    # Save the results as .nii.gz
    right_acc_image_path = os.path.join(output_folder, f"right_{os.path.basename(output_folder)}_image.nii.gz")
    left_acc_image_path = os.path.join(output_folder, f"left_{os.path.basename(output_folder)}_image.nii.gz")

    save_as_nii_gz(right_zero_image, right_acc_image_path)
    save_as_nii_gz(left_zero_image, left_acc_image_path)

    print(f"Saved: {left_acc_image_path} and {right_acc_image_path}")

def visualize_stl(stl):
    mesh = Mesh.from_file(stl)
    
    # Plot the mesh, change th opacity to make the femur head point visible
    vpl.mesh_plot(mesh, opacity=1,color = 'white')
    
    # Plot the points
    # Change the background color
    vpl.gcf().background_color = (0,0,0)  # White background
    
    # vpl.plot(physical_point, color="blue", line_width=3)  # Blue line with thickness 3
    
    # Display the plot
    vpl.show()

# hc , tc, fc, cc, pc = 0, 0, 0, 0, 0
# root_path = "/media/interns/storage/PLAI/2020"
# json_file_path = "./json_files/stl_files_report.json"

# for patient_dir in os.listdir(root_path):
#     if ".DS" in patient_dir:
#         continue
#     patient_dir_path = os.path.join(root_path, patient_dir)
#     for dirpath, dirnames, filenames in os.walk(patient_dir_path, topdown=True):
#         files = []
#         cf = []
#         for file in filenames:
#             if file.lower().endswith(".stl"):
#                 if "hanche" in file: 
#                     hc +=1
#                     cf.append(file)
#                 elif "tibia" in file:
#                     tc +=1
#                     cf.append(file)
#                 elif "femur" in file:
#                     fc +=1
#                     cf.append(file)
#                 elif "cheville" in file:
#                     cc +=1
#                     cf.append(file)
#                 else:
#                     files.append(file)
#                 # print(os.path.join(dirpath, file))
#         # if files != [] and len(cf)<4:
#             message_to_write = {
#                 "patient_dir": patient_dir,
#                 "stl_files_present": cf,
#                 "other_stl_files": files
#             }
#             write_to_json(json_file_path, message_to_write)
#     pc +=1
#     print(f"Patient directory {pc} visited.")
    
#     # break
# print(cc, fc, tc, hc)


def get_stl_files(patient_dir_path):
    for dirpath, dirname, filename in os.walk(patient_dir_path, topdown=True):
        stl_files = []
        for file in filename:
            if file.lower().endswith(".stl"):
                if "hanche" or "tibia" or "femur" or "cheville" or "fibula" in file:
                    file_path = os.path.join(dirpath, file)
                    stl_files.append(file_path)
        if stl_files != []:
            break
    return stl_files

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
                # print(f"Skipping file {file_path}: {e}")
                pass

# Example usage:
# organize_dicom_series("/path/to/root_directory", "/path/to/save_root")

def process_dicom_series(root_path, temp_root, new_root, min_files=20):
    try:
        temp_root_path = os.path.join(temp_root, os.path.basename(root_path))
        new_root_path = os.path.join(new_root, os.path.basename(root_path))

        stl_files = get_stl_files(root_path)
        stl_save_path = os.path.join(new_root_path, "stl_files")
        os.makedirs(stl_save_path, exist_ok=True)
        for stl_file in stl_files:
            shutil.copy(stl_file, stl_save_path)

        organize_dicom_series(root_path, temp_root_path)
        # Ensure the new root path exists
        os.makedirs(new_root_path, exist_ok=True)

        # Walk through the root directory
        for dirpath, dirnames, filenames in os.walk(temp_root_path):
            if len(filenames) > min_files:
                # Find the relative path from the root_path
                relative_path = os.path.relpath(dirpath, temp_root_path)

                # Create the corresponding destination path
                destination_dir = os.path.join(new_root_path, relative_path)

                # Ensure the destination directory exists
                os.makedirs(destination_dir, exist_ok=True)

                # Copy files
                for filename in filenames:
                    src_file = os.path.join(dirpath, filename)
                    dst_file = os.path.join(destination_dir, filename)
                    shutil.copy2(src_file, dst_file)
                    
        shutil.rmtree(temp_root_path)
    except Exception as e:
        logging.error(f"{root_path}", exc_info=True)



def create_masks_from_stl(dicom_directory, mask_save_directory, stl_files):
    """
    Creates segmentation masks for DICOM images using multiple STL files.
    
    Parameters:
        dicom_directory (str): Path to the directory containing DICOM files.
        mask_save_directory (str): Path where the segmentation masks will be saved.
        stl_files (list): List of STL file paths to be tested for mask generation.
    """

    # Ensure the output directory exists
    os.makedirs(mask_save_directory, exist_ok=True)

    # Read DICOM Images
    PixelType = itk.ctype("signed short")
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(dicom_directory)

    seriesUIDs = namesGenerator.GetSeriesUIDs()
    if not seriesUIDs:
        print(f"No DICOM images found in {dicom_directory}")
        return

    # Process only the first DICOM series
    seriesIdentifier = seriesUIDs[0]
    fileNames = namesGenerator.GetFileNames(seriesIdentifier)
    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()
    reader.Update()

    dicom_image = reader.GetOutput()

    # Iterate over STL files and generate masks
    for stl_filename in stl_files:
        try:
            print(f"Processing STL: {stl_filename}")

            # Convert STL to VTK
            stl_reader = vtk.vtkSTLReader()
            stl_reader.SetFileName(stl_filename)
            stl_reader.Update()
            vtk_mesh = stl_reader.GetOutput()

            # # Save as .vtk (temporary conversion)
            vtk_filename = stl_filename.replace(".stl", ".vtk")
            # vtk_writer = vtk.vtkPolyDataWriter()
            # vtk_writer.SetFileName(vtk_filename)
            # vtk_writer.SetInputData(vtk_mesh)
            # vtk_writer.Write()
            
            # Save as ASCII VTK
            vtk_writer = vtk.vtkPolyDataWriter()
            vtk_writer.SetFileName(vtk_filename)
            vtk_writer.SetInputData(vtk_mesh)
            vtk_writer.SetFileTypeToASCII()  # Ensure ASCII format
            vtk_writer.Write()


            if not vtk_mesh or vtk_mesh.GetNumberOfPoints() == 0:
                        print(f"Error: STL file {stl_filename} is empty or unreadable.")
                        continue

            # Convert VTK mesh to ITK mesh
            # MeshType = itk.Mesh[itk.F, Dimension]  # Floating point mesh
            # Read VTK as ITK mesh
            MeshType = itk.Mesh[itk.SS, Dimension]
            mesh_reader = itk.MeshFileReader[MeshType].New()
            mesh_reader.SetFileName(vtk_filename)
            mesh_reader.Update()

            # Convert Mesh to Binary Image
            OutputImageType = itk.Image[itk.SS, Dimension]
            cast_filter = itk.CastImageFilter[ImageType, OutputImageType].New()
            cast_filter.SetInput(dicom_image)

            tri_mesh_2_binary = itk.TriangleMeshToBinaryImageFilter[MeshType, OutputImageType].New()
            tri_mesh_2_binary.SetInput(mesh_reader.GetOutput())
            tri_mesh_2_binary.SetInfoImage(cast_filter.GetOutput())
            tri_mesh_2_binary.SetInsideValue(255)  # Inside value set to 255
            tri_mesh_2_binary.Update()

            segmentation_mask = tri_mesh_2_binary.GetOutput()
            mask_array = itk.GetArrayFromImage(segmentation_mask)
            
            # mask_array = np.flip(mask_array)
            mask_array = np.rot90(mask_array, k=2)
            mask_array = np.fliplr(mask_array)

            nii_path = os.path.basename(stl_filename).replace('.stl', '.nii.gz')

            contains_mask = check_mha_nonzero(mask_array, nii_path)

            if contains_mask:
                mask_filename = os.path.join(mask_save_directory, nii_path)

                # Apply transformation
                # transformed_mask = transform_seg(segmentation_mask, mask_filename)
                transformed_mask = segmentation_mask

                # Save transformed mask using ITK
                writer = itk.ImageFileWriter[OutputImageType].New()
                writer.SetFileName(mask_filename)
                writer.UseCompressionOn()
                writer.SetInput(transformed_mask)

                writer.Update()
                # transform_seg(mask_filename, mask_filename)
                print(f"Saved mask: {mask_filename}")
                
        except:
            print(f"Failed to convert {stl_filename} to nifti image")

    print("Mask generation complete!")

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


def visualize_middle_slice(dicom_files, Series_description=""):
    if dicom_files:
        # Sort slices based on InstanceNumber or SliceLocation if available
        dicom_files.sort(key=lambda x: getattr(x, 'InstanceNumber', 0))

        # Stack pixel arrays
        volume = np.stack([ds.pixel_array for ds in dicom_files])

        middle_idx = volume.shape[0] // 2
        middle_slice = volume[middle_idx]

        # Visualize all three planes
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(middle_slice, cmap='gray')
        axs[0].set_title('Axial View')
        axs[0].axis('off')

        axs[1].imshow(volume[:, middle_slice.shape[0]//2, :], cmap='gray')
        axs[1].set_title('Coronal View')
        axs[1].axis('off')

        axs[2].imshow(volume[:, :, middle_slice.shape[1]//2], cmap='gray')
        axs[2].set_title('Sagittal View')
        axs[2].axis('off')

        plt.suptitle(f"Middle Slices for Directory: {Series_description}")
        plt.show()

def get_mask_from_stl(dicom_dir, nifti_save_path, stl_files_list):
    for stl_file_path in stl_files_list:
        mask_image = stl_to_dicom_mask(dicom_dir, stl_file_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        stl_file = os.path.basename(stl_file_path)
        has_mask = check_mha_nonzero(mask_image_array, stl_file)
        if has_mask:
            sitk.WriteImage(mask_image, f"{nifti_save_path}/{stl_file.replace(".stl", ".nii.gz")}")
            print(f"{stl_file} mask image saved successfully.")
        else:
            print(f"The {stl_file} mask image is zero.")




# def get_dicom_files(root_path):
#     stl_files = [
#     os.path.join(dirpath, f)
#     for dirpath, _, filenames in os.walk(root_path)
#     for f in filenames if f.endswith(".stl")
#     ]

#     for dirpath, dirnames, filenames in os.walk(root_path):
#         dicom_files = []
#         # dicom_files_path = []
#         for filename in filenames:
#             file_path = os.path.join(dirpath, filename)
#             try:
#                 ds = pydicom.dcmread(file_path, stop_before_pixels=False)
#                 dicom_files.append(ds)
#                 # dicom_files_path.append(file_path)
#             except Exception as e:
#                 # Not a valid DICOM file
#                 continue
#         Series_description = os.path.basename(dirpath)
#         visualize_middle_slice(dicom_files, Series_description)
#         if dicom_files:
#             leg_count = "unilateral" if (inp := user_input("Enter u/b for unilateral and bilateral leg from the visualization below: ").lower()) == "u" else "bilateral" if inp == "b" else None
#             part = "ankle" if (inp := user_input("Enter a/p/k for ankle, pelvis and knee region from the visualization below:").lower()) == "a" else "pelvis" if inp == "p" else "knee" if inp == "k" else None
#             nifti_save_path =os.path.join("/Users/kamleshranabhat/Documents/nii_images/Processed_2020", os.path.basename(root_path), f"{leg_count}_{part}")
#             i = 0
            
#             while os.path.exists(nifti_save_path):
#                 i +=1
#                 nifti_save_path = os.path.join("/Users/kamleshranabhat/Documents/nii_images/Processed_2020", os.path.basename(root_path), f"{leg_count}_{part}_{i}")
#             os.makedirs(nifti_save_path, exist_ok=True)
            
#             if leg_count== "unilateral":
#                 process_dicom_folder(dirpath, nifti_save_path)
#             elif leg_count == "bilateral":
#                 process_double_leg_dicom_folder(dirpath, nifti_save_path)
                
#             create_masks_from_stl(dirpath, nifti_save_path, stl_files)
            
#             # print(leg_count, part)
            
#             # break

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
        visualize_middle_slice(dicom_files, Series_description)
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
            target_folder = parts[-3]

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
                
            get_mask_from_stl(dirpath, nifti_save_path, stl_files)

