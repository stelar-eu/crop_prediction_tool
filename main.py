import json
import sys
import traceback
from utils.mclient import MinioClient
from utils.processing import prepare_data
from utils.inference import run_inference
from utils.io_utils import save_geotiff, save_png
from utils.config import transform, crs_code, get_labels_in_color
import os
import numpy as np


def run(json):
    try:
        ######################## MINIO INIT ########################
        minio_id = json['minio']['id']
        minio_key = json['minio']['key']
        minio_skey = json['minio']['skey']
        minio_endpoint = json['minio']['endpoint_url']

        mc = MinioClient(minio_endpoint, minio_id, minio_key, secure=True, session_token=minio_skey)
        ############################################################

        ####################### PARAMETER READ #####################
        params = json['parameters']
        inputs = json['inputs']
        outputs = json['outputs']

        season = params['season']

        # Input paths (assuming standard keys for this pipeline)
        lai_files = inputs['lai_files']          # path to LAI folder in MinIO
        label_file = inputs['label_file'][0]         # path to aligned label file
        model_folder = inputs['model_folder'][0]     # folder with pretrained models

        # Output paths
        predicted_tif_path = outputs['predicted_tif']
        ground_truth_tif_path = outputs['ground_truth_tif']
        pred_png_path = outputs['predicted_png']
        gt_png_path = outputs['ground_truth_png']

        ################### DATA PREPARATION #######################
        data_tensor, label_tensor, file_list = prepare_data(mc, lai_files, label_file)

        ####################### INFERENCE ##########################
        prediction, ground_truth = run_inference(
            data_tensor,
            label_tensor,
            season,
            model_folder,
            mc
        )

        ####################### OUTPUT SAVE ########################
        save_geotiff(predicted_tif_path, prediction, transform, crs_code, mc)
        save_geotiff(ground_truth_tif_path, ground_truth, transform, crs_code, mc)

        # Save PNGs
        prediction_color = get_labels_in_color(prediction)
        ground_truth_color = get_labels_in_color(ground_truth)

        save_png(pred_png_path, prediction_color, mc)
        save_png(gt_png_path, ground_truth_color, mc)

        ####################### METRICS #############################
        crop_ids, crop_counts = np.unique(prediction, return_counts=True)
        crop_distribution = {int(c): int(n) for c, n in zip(crop_ids, crop_counts)}

        return {
            'message': f'Segmentation completed for season {season}',
            'outputs': {
                'predicted_tif': predicted_tif_path,
                'ground_truth_tif': ground_truth_tif_path,
                'predicted_png': pred_png_path,
                'ground_truth_png': gt_png_path
            },
            'metrics': {
                'prediction_shape': prediction.shape,
                'crop_distribution': crop_distribution
            },
            'status': 'success'
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            'message': 'An error occurred during data processing.',
            'error': traceback.format_exc(),
            'status': 500
        }


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    with open(sys.argv[1]) as o:
        j = json.load(o)
    response = run(j)
    with open(sys.argv[2], 'w') as o:
        o.write(json.dumps(response, indent=4))
