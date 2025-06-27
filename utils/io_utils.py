import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import numpy as np
from mclient import MinioClient
import os

def save_geotiff(s3_path, array, transform, crs, mc: MinioClient):
    """
    Saves an array as a GeoTIFF to MinIO.
    """
    local_path = "/tmp/temp_output.tif"
    with rasterio.open(
        local_path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(array, 1)

    mc.put_object(s3_path=s3_path, file_path=local_path)
    os.remove(local_path)


def save_png(s3_path, array_rgb, mc: MinioClient):
    """
    Saves an RGB array as a PNG image to MinIO.
    """
    local_path = "/tmp/temp_output.png"
    plt.figure(figsize=(10, 10))
    plt.imshow(array_rgb)
    plt.axis('off')
    plt.savefig(local_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    mc.put_object(s3_path=s3_path, file_path=local_path)
    os.remove(local_path)