import pydicom

# Specify the path to your DICOM file
dicom_file_path = "Altea_t1_sag_tse_DR_77233508.dcm"

# Read the DICOM file
ds = pydicom.dcmread(dicom_file_path)

# Now you can access DICOM metadata and pixel data
print(ds)


# Check if PixelSpacing is directly available
if hasattr(ds, 'PixelSpacing'):
    pixel_spacing = ds.PixelSpacing
else:
    # Look for PixelSpacing in shared functional groups sequence
    shared_functional_groups_sequence = ds.get('SharedFunctionalGroupsSequence')
    if shared_functional_groups_sequence:
        pixel_spacing = shared_functional_groups_sequence[0].PixelMeasuresSequence[0].PixelSpacing
    else:
        raise AttributeError("PixelSpacing not found in DICOM file.")

# Now you can proceed to calculate the conversion factor and other operations as before
row_spacing, column_spacing = pixel_spacing
conversion_factor = float(row_spacing)

# Now you can use this conversion factor to convert pixel distances to real-world distances
# For example, if you have a distance of 10 pixels, you can calculate the equivalent distance in millimeters:
pixel_distance = 10
real_world_distance_mm = pixel_distance * conversion_factor
print("Distance in millimeters:", real_world_distance_mm)


# Access pixel data (assuming the DICOM file contains image data)
pixel_data = ds.pixel_array

# Now you can work with the image data, for example, display it
import matplotlib.pyplot as plt
plt.imshow(pixel_data, cmap=plt.cm.gray)
plt.show()