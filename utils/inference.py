import numpy as np
import gc
import segmentation_models_3D as sm
from scipy.stats import mode
from utils.config import crop_types_all_list, season_config
from utils.mclient import MinioClient
import tensorflow as tf


def slice_into_cubes(data_tensor, label_tensor, cube_size=64):
    """Slices large spatio-temporal arrays into 3D cubes."""
    slices_X, slices_Y = data_tensor.shape[1] // cube_size, data_tensor.shape[2] // cube_size

    X_cubes, y_cubes = [], []
    for i in range(slices_X):
        for j in range(slices_Y):
            x = data_tensor[:, i*cube_size:(i+1)*cube_size, j*cube_size:(j+1)*cube_size]
            y = label_tensor[i*cube_size:(i+1)*cube_size, j*cube_size:(j+1)*cube_size]
            X_cubes.append(x)
            y_cubes.append(y)

    X = np.stack(X_cubes, axis=0)  # (N, 64, 64, 64)
    y = np.stack(y_cubes, axis=0)
    return X, y


def run_inference(data_tensor, label_tensor, season, model_folder, mc: MinioClient):
    """
    Loads models for the given season, performs inference,
    ensembles predictions, and returns reconstructed maps.
    """
    config = season_config[season]
    chosen_crop_sets = config['crop_triplets']
    chosen_season_crops = config['season_crop_ids']

    X_cubes, y_cubes = slice_into_cubes(data_tensor, label_tensor)
    X_cubes = np.stack((X_cubes,) * 3, axis=-1)  # (N, 64, 64, 64, 3)

    y_cubes = np.repeat(y_cubes[:, np.newaxis, :, :], repeats=64, axis=1)  # shape: (N, 64, 64)

    all_preds, all_gts = [], []

    for i in range(len(X_cubes)):
        pred_ensemble = np.zeros((64, 64, 64))
        stack_preds = []
        test_img = X_cubes[i]
        gt_patch = y_cubes[i, 0, :, :] + 10

        for k in crop_types_all_list:
            if k not in chosen_season_crops:
                gt_patch[gt_patch == k] = 0

        test_input = np.expand_dims(test_img, 0)

        for crop_triplet in chosen_crop_sets:
            model_name = f"3D_unet_g4_h4_crop_{crop_triplet[0]-10}_{crop_triplet[1]-10}_{crop_triplet[2]-10}_epoch_f.h5"
            model_path = f"{model_folder}/{model_name}"
            local_model_path = f"/tmp/{model_name}"

            mc.get_object(s3_path=model_path, local_path=local_model_path)

            model = sm.Unet('vgg16', input_shape=(64, 64, 64, 3), classes=4, encoder_weights=None, activation='softmax')
            model.load_weights(local_model_path, by_name=True, skip_mismatch=True)

            pred = model.predict(test_input)
            pred_class = np.argmax(pred, axis=4)[0, :, :, :]

            for idx in range(3):
                pred_class[pred_class == idx + 1] = crop_triplet[idx]

            stack_preds.append(pred_class)
            del model
            gc.collect()

        stack_preds = np.stack(stack_preds)

        # Mode ensemble
        mode_pred = np.zeros((64, 64))
        for x in range(64):
            for y in range(64):
                valid = stack_preds[:, x, y]
                if np.sum(valid) == 0:
                    mode_pred[x, y] = 0
                else:
                    mode_pred[x, y] = mode(valid[valid > 0])[0]

        all_preds.append(mode_pred.astype(np.uint8))
        all_gts.append(gt_patch.astype(np.uint8))

    # Reconstruct full map
    grid_size = int(np.sqrt(len(all_preds)))
    H = grid_size * 64
    final_pred = np.zeros((H, H), dtype=np.uint8)
    final_gt = np.zeros((H, H), dtype=np.uint8)

    k = 0
    for i in range(0, H, 64):
        for j in range(0, H, 64):
            final_pred[i:i+64, j:j+64] = all_preds[k]
            final_gt[i:i+64, j:j+64] = all_gts[k]
            k += 1

    return final_pred[:10002, :10002], final_gt[:10002, :10002]
