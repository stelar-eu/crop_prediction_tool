import numpy as np
from rasterio.transform import from_origin

# GeoTIFF metadata
ulx = 704855.0
uly = 4995145.0
pixel_size = 10.0
crs_code = 'EPSG:32630'
transform = from_origin(ulx, uly, pixel_size, pixel_size)

# Crop ID color mapping
color_map = {
    10: [0, 0, 0], 11: [0, 255, 0], 12: [0, 0, 255], 13: [255, 255, 0],
    14: [255, 165, 0], 15: [255, 0, 255], 16: [0, 255, 255], 17: [128, 0, 128],
    18: [128, 128, 0], 19: [0, 128, 0], 20: [128, 0, 0], 21: [0, 0, 128],
    22: [128, 128, 128], 23: [0, 128, 128], 24: [255, 0, 0], 25: [255, 255, 255],
    26: [192, 192, 192], 27: [139, 0, 0], 28: [0, 100, 0], 29: [0, 0, 139],
    30: [255, 215, 0], 31: [255, 140, 0], 32: [139, 0, 139], 33: [0, 206, 209],
    34: [75, 0, 130], 35: [85, 107, 47], 36: [34, 139, 34], 37: [165, 42, 42],
    38: [70, 130, 180], 39: [169, 169, 169], 40: [32, 178, 170], 41: [47, 79, 79],
    42: [245, 245, 245], 43: [105, 105, 105], 44: [205, 92, 92], 45: [50, 205, 50],
    46: [65, 105, 225], 47: [255, 223, 0], 48: [255, 99, 71], 49: [186, 85, 211],
    50: [0, 191, 255], 51: [192, 192, 192]
}

# For all seasons
crop_types_all_list = [
    10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 28, 29, 30, 31, 33, 37, 38, 40, 42, 43, 44, 45,
    46, 47, 50, 51
]

# Season configuration
season_config = {
    "Feb_Aug": {
        "crop_triplets": [[43, 46, 51], [44, 47, 50]],
        "time_strip_ind": (64 * 1),
        "season_crop_ids": [10, 43, 44, 46, 47, 50, 51]
    },
    "May_Aug": {
        "crop_triplets": [[12, 25, 30], [31, 33, 38]],
        "time_strip_ind": (64 * 3),
        "season_crop_ids": [10, 12, 25, 30, 31, 33, 38]
    },
    "Jun_Oct": {
        "crop_triplets": [[18, 19, 40], [17, 28, 29]],
        "time_strip_ind": (64 * 4),
        "season_crop_ids": [10, 18, 19, 40, 17, 28, 29]
    },
    "Jan_Aug": {
        "crop_triplets": [[14, 17, 19]],
        "time_strip_ind": (64 * 0),
        "season_crop_ids": [10, 14, 17, 19]
    }
}

def get_labels_in_color(label_image):
    """
    Maps label values to RGB colors using color_map.
    """
    color_img = np.zeros(label_image.shape + (3,), dtype=np.uint8)
    for i in range(label_image.shape[0]):
        for j in range(label_image.shape[1]):
            color_img[i, j] = color_map.get(label_image[i, j], [0, 0, 0])
    return color_img
