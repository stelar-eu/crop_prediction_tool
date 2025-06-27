import numpy as np
import tensorflow as tf
import gc
from utils.mclient import MinioClient


def pad_array(arr, pad_height=46, pad_width=46):
    return np.pad(arr, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)


def prepare_data(mc: MinioClient, lai_filepaths, label_file):
    """
    Downloads and prepares LAI time series and labels.
    Uses explicitly provided .npy files, preserving order.
    """
    from utils.config import season_config

    lai_stack = []
    for filepath in lai_filepaths:
        local_path = "/tmp/" + filepath.split("/")[-1]
        mc.get_object(s3_path=filepath, local_path=local_path)
        arr = np.load(local_path).astype(np.float32)
        arr = pad_array(arr)
        lai_stack.append(arr)

    all_processed_LAI = np.stack(lai_stack, axis=0)  # (64, H, W)

    # Download label file
    local_label_path = "/tmp/label.npy"
    mc.get_object(s3_path=label_file, local_path=local_label_path)
    labels = np.load(local_label_path).astype(np.uint8)
    labels = pad_array(labels)

    return tf.convert_to_tensor(all_processed_LAI), tf.convert_to_tensor(labels), lai_filepaths

