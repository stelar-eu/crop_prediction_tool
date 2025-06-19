import glob
from skimage import io
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

vista_crop_dict = {0:'NA', 10:'NA' , 11: 'ALFALFA', 12: 'BEET', 13: 'CLOVER', 14: 'FLAX', 15: 'FLOWERING_LEGUMES', 16: 'FLOWERS', 17: 'FOREST', 18: 'GRAIN_MAIZE', 19: 'GRASSLAND', 20: 'HOPS', 21: 'LEGUMES', 22: 'VISTA_NA', 23: 'PERMANENT_PLANTATIONS', 24: 'PLASTIC', 25: 'POTATO', 26: 'PUMPKIN', 27: 'RICE', 28: 'SILAGE_MAIZE', 29: 'SOY', 30: 'SPRING_BARLEY', 31: 'SPRING_OAT', 32: 'SPRING_OTHER_CEREALS', 33: 'SPRING_RAPESEED', 34: 'SPRING_RYE', 35: 'SPRING_SORGHUM', 36: 'SPRING_SPELT', 37: 'SPRING_TRITICALE', 38: 'SPRING_WHEAT', 39: 'SUGARBEET', 40: 'SUNFLOWER', 41: 'SWEET_POTATOES', 42: 'TEMPORARY_GRASSLAND', 43: 'WINTER_BARLEY', 44: 'WINTER_OAT', 45: 'WINTER_OTHER_CEREALS', 46: 'WINTER_RAPESEED', 47: 'WINTER_RYE', 48: 'WINTER_SORGHUM', 49: 'WINTER_SPELT', 50: 'WINTER_TRITICALE', 51: 'WINTER_WHEAT'}


ground_truth_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights/no_cloud_interpol/g*.tif')
ground_truth_filepaths.sort()

predictions_filepaths = glob.glob('./ensamble_results/iou_f1_class_weights/no_cloud_interpol/p*.tif')
predictions_filepaths.sort()


all_test_ground_truth = []
all_test_prediction = []
for ground_truth_address, prediction_address in zip(ground_truth_filepaths, predictions_filepaths):
    ground_truth = io.imread(ground_truth_address)
    prediction = io.imread(prediction_address)
    all_test_ground_truth.append(ground_truth)
    all_test_prediction.append(prediction)  
all_test_ground_truth = np.array(all_test_ground_truth)
all_test_prediction = np.array(all_test_prediction)

def calculate_iou(ground_truth, prediction):
    ground_truth = ground_truth.astype(np.bool_)
    prediction = prediction.astype(np.bool_)
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    if np.sum(union) == 0:
        iou = 0
    else:
        iou = np.sum(intersection) / np.sum(union)    
    return iou

def calculate_f1_score(ground_truth, prediction):
    ground_truth = ground_truth.astype(np.bool_)
    prediction = prediction.astype(np.bool_)
    tp = np.sum(np.logical_and(prediction, ground_truth))
    fp = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)))
    fn = np.sum(np.logical_and(np.logical_not(prediction), ground_truth))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def calculate_accuracy(ground_truth, prediction):
    ground_truth = ground_truth.astype(np.bool_)
    prediction = prediction.astype(np.bool_)
    
    tp = np.sum(np.logical_and(prediction, ground_truth))
    tn = np.sum(np.logical_and(np.logical_not(prediction), np.logical_not(ground_truth)))
    fp = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)))
    fn = np.sum(np.logical_and(np.logical_not(prediction), ground_truth))
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    return accuracy



all_unique_crop_types = np.array([12, 25, 30, 31, 33, 38, 43, 44, 46, 47, 50, 51, 18, 19, 40, 17, 28, 29, 14], dtype=np.uint8)


# IoU

all_crops_iou_distributions = []
for unique in all_unique_crop_types:
    each_iou_distribution = []
    for i in range(all_test_ground_truth.shape[0]):
        ground_truth, prediction = all_test_ground_truth[i].copy(), all_test_prediction[i].copy()
        ground_truth[ground_truth!=unique]=0
        prediction[prediction!=unique]=0

        ground_truth_b = ground_truth.astype(np.bool_)
        prediction_b = prediction.astype(np.bool_)
        if(not np.sum(ground_truth)==0 and not np.sum(prediction)==0):
            iou = calculate_iou(ground_truth, prediction)
            each_iou_distribution.append(iou)
    all_crops_iou_distributions.append(each_iou_distribution)


# Accuracy
all_crops_accuracy_distributions = []
for unique in all_unique_crop_types:
    each_acc_distribution = []
    for i in range(all_test_ground_truth.shape[0]):
        ground_truth, prediction = all_test_ground_truth[i].copy(), all_test_prediction[i].copy()
        ground_truth[ground_truth!=unique]=0
        prediction[prediction!=unique]=0

        ground_truth_b = ground_truth.astype(np.bool_)
        prediction_b = prediction.astype(np.bool_)
        if(not np.sum(ground_truth)==0):
            acc = calculate_accuracy(ground_truth, prediction)
            each_acc_distribution.append(acc)
    all_crops_accuracy_distributions.append(each_acc_distribution)


## F1 Score


all_crops_f1_distributions = []
for unique in all_unique_crop_types:
    each_f1_distribution = []
    for i in range(all_test_ground_truth.shape[0]):
        ground_truth, prediction = all_test_ground_truth[i].copy(), all_test_prediction[i].copy()        
        ground_truth[ground_truth!=unique]=0
        prediction[prediction!=unique]=0
        ground_truth_b = ground_truth.astype(np.bool_)
        prediction_b = prediction.astype(np.bool_)
        if(not np.sum(ground_truth)==0 and not np.sum(prediction)==0):
            f1 = calculate_f1_score(ground_truth, prediction)
            each_f1_distribution.append(f1)
    all_crops_f1_distributions.append(each_f1_distribution)



############# confusion matrix computation #############
all_unique_crop_types = np.array([12, 25, 30, 31, 33, 38, 43, 44, 46, 47, 50, 51, 18, 19, 40, 17, 28, 29, 14], dtype=np.uint8)
all_gt_labels = []
all_pred_labels = []
for i in range(all_test_ground_truth.shape[0]):
    gt = all_test_ground_truth[i].copy().flatten()
    pred = all_test_prediction[i].copy().flatten()
    mask = np.isin(gt, all_unique_crop_types)  # You could also apply this to pred if needed
    gt = gt[mask]
    pred = pred[mask]
    all_gt_labels.append(gt)
    all_pred_labels.append(pred)

all_gt_labels = np.concatenate(all_gt_labels)
all_pred_labels = np.concatenate(all_pred_labels)

crop_labels = [vista_crop_dict[crop_id] for crop_id in all_unique_crop_types]
cm = confusion_matrix(all_gt_labels, all_pred_labels, labels=all_unique_crop_types)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=crop_labels)
fig, ax = plt.subplots(figsize=(15, 15))
disp.plot(ax=ax, cmap='viridis', xticks_rotation=45)
plt.title("Confusion Matrix of Crop Type Segmentation")
plt.savefig("./evaluation_results/crop_type_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
############# confusion matrix computation #############


# Plotting IOU
data = all_crops_iou_distributions
crop_types = [vista_crop_dict[element] for element in all_unique_crop_types]
plt.figure(figsize=(14, 8)) 
box = plt.boxplot(data, patch_artist=True, widths=0.6)  
cmap = plt.cm.get_cmap('tab20', len(data)) 
colors = [cmap(i) for i in range(len(data))]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.title('IOU Distributions', fontsize=16)
plt.xlabel('Crop Types', fontsize=14)
plt.ylabel('Intersection over Union', fontsize=14)
positions = range(1, len(data) + 1)
plt.xticks(positions, crop_types, rotation=90, fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig('./evaluation_results/exp2_iou_no_cloud_interpol.png', bbox_inches='tight')
plt.show()



# Plotting F1 score
data = all_crops_f1_distributions
crop_types = [vista_crop_dict[element] for element in all_unique_crop_types]
plt.figure(figsize=(14, 8))  
box = plt.boxplot(data, patch_artist=True, widths=0.6)
cmap = plt.cm.get_cmap('tab20', len(data)) 
colors = [cmap(i) for i in range(len(data))]

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.title('F1 Scores', fontsize=16)
plt.xlabel('Crop Types', fontsize=14)
plt.ylabel('F1 Scores', fontsize=14)
positions = range(1, len(data) + 1)
plt.xticks(positions, crop_types, rotation=90, fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig('./evaluation_results/exp2_f1_no_cloud_interpol.png', bbox_inches='tight')
plt.show()
plt.close()


# Plotting Accuracy 
data = all_crops_accuracy_distributions
crop_types = [vista_crop_dict[element] for element in all_unique_crop_types]
plt.figure(figsize=(14, 8)) 
box = plt.boxplot(data, patch_artist=True, widths=0.6)  
cmap = plt.cm.get_cmap('tab20', len(data))  
colors = [cmap(i) for i in range(len(data))]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.title('Accuarcy', fontsize=16)
plt.xlabel('Crop Types', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
positions = range(1, len(data) + 1)
plt.xticks(positions, crop_types, rotation=90, fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig('./evaluation_results/exp2_acc_no_cloud_interpol.png', bbox_inches='tight')
plt.show()
plt.close()


# Writing evaluation report
with open('./evaluation_results/evaluation_report.txt', mode='w') as file:
    header = f"{'Crop Type':<20} {'Avg Accuracy':<15} {'Avg IOU':<15} {'Avg F1 Score':<15}\n"
    file.write(header)
    file.write('-' * len(header) + '\n')

    for kk in range(len(all_crops_accuracy_distributions)):
        crop_name = vista_crop_dict[all_unique_crop_types[kk]]
        average_accuracy = sum(all_crops_accuracy_distributions[kk]) / len(all_crops_accuracy_distributions[kk])
        average_iou = sum(all_crops_iou_distributions[kk]) / len(all_crops_iou_distributions[kk])
        average_f1_score = sum(all_crops_f1_distributions[kk]) / len(all_crops_f1_distributions[kk])

        line = f"{crop_name:<20} {average_accuracy:<15.6f} {average_iou:<15.6f} {average_f1_score:<15.6f}\n"
        file.write(line)