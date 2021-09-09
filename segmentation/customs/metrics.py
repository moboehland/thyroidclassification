import torch
import numpy as np
from multiprocessing.pool import Pool
from collections import Counter
import shutil
import subprocess
import tifffile as tiff
import platform
import re
import os


def iou_pytorch(predictions, labels):
    """ Online IoU-metric (batch-wise over all classes).

    :param predictions: Batch of predictions.
        :type predictions:
    :param labels: Batch of ground truths / label images.
        :type labels:
    :param device: cuda (gpu) or cpu.
        :type device:
    :return: Intersection over union.
    """

    a = predictions.clone().detach()

    # Apply sigmoid activation function in one-class problems
    a = torch.sigmoid(a)

    # Flatten predictions and apply threshold
    a = a.view(-1) > torch.tensor([0.5], requires_grad=False, device=a.device)

    # Flatten labels
    b = labels.clone().detach().view(-1).bool()

    # Calculate intersection over union
    intersection = torch.sum((a * b).float())
    union = torch.sum(torch.max(a, b).float())
    iou = intersection / (union + 1e-6)

    return iou


def metric_collection(prediction, ground_truth, num_threads=8, path_ctc_software=None):
    """ Calculation of Rand-Index, Jaccard-Index, mean average precision at different intersection over union
    thresholds (P_IoU), precision, recall, F-score and split/merged/missing/spurious objects.

    :param prediction: Prediction with intensity coded nuclei.
        :type prediction:
    :param ground_truth: Ground truth image with intensity coded nuclei.
        :type ground_truth:
    :param num_threads: Number of threads to speeden up the computation.
        :type num_threads: int
    :return: Dictionary containing the metric scores.
    """

    # Create copy of the prediction and ground truth to avoid changing them
    pred, gt = np.copy(prediction), np.copy(ground_truth)

    # Find intensity coded nuclei in the ground truth image and the prediction (simply looking for the maximum is not
    # possible because in the post-processing numbered seeds can vanish, additionally for tracking data some nuclei
    # may not appear at that time point)
    nucleus_ids_ground_truth, nucleus_ids_prediction = get_nucleus_ids(gt), get_nucleus_ids(pred)

    # Number of cell nuclei in the ground truth image and in the prediction
    num_nuclei_ground_truth, num_nuclei_prediction = len(nucleus_ids_ground_truth), len(nucleus_ids_prediction)

    # Check for empty predictions
    if num_nuclei_prediction == 0:
        return {'Rand_index': 0, 'Jaccard_index': 0, 'Aggregated_Jaccard_index': 0, 'P_IoU': 0, 'Precision': 0,
                'Recall': 0, 'F-Score': 0, 'Split': 0, 'Merged': 0, 'Missing': num_nuclei_ground_truth, 'Spurious': 0
                }, 0

    # Check for missing nuclei ids in the prediction. To build the intersection histogram the nuclei_ids should range
    # from 1 to the number of nuclei.
    if num_nuclei_prediction != pred.max():

        hist = np.histogram(pred, bins=range(1, pred.max() + 2), range=(1, pred.max() + 1))

        # Find missing values
        missing_values = np.where(hist[0] == 0)[0]

        # Decrease the ids of the nucleus with higher id than the missing. Reverse the list to avoid problems in case
        # of multiple missing objects
        for th in reversed(missing_values):
            pred[pred > th] = pred[pred > th] - 1

    # Check for missing nuclei ids in the ground truth. To build the intersection histogram the nuclei_ids should range
    # from 1 to the number of nuclei.
    if num_nuclei_ground_truth != gt.max():

        hist = np.histogram(gt, bins=range(1, gt.max() + 2), range=(1, gt.max() + 1))

        # Find missing values
        missing_values = np.where(hist[0] == 0)[0]

        # Decrease the ids of the nucleus with higher id than the missing. Reverse the list to avoid problems in case
        # of multiple missing objects
        for th in reversed(missing_values):
            gt[gt > th] = gt[gt > th] - 1

    # Change the background label from 0 to num_nuclei + 1. This enables to calculate the intersection with the
    # background efficiently.
    bg_gt, bg_pred = num_nuclei_ground_truth + 1, num_nuclei_prediction + 1
    pred[pred == 0] = bg_pred
    gt[gt == 0] = bg_gt
    nucleus_ids_ground_truth, nucleus_ids_prediction = get_nucleus_ids(gt), get_nucleus_ids(pred)

    # Preallocate arrays for the intersection histogram
    intersections = np.zeros(shape=(num_nuclei_ground_truth+1, num_nuclei_prediction+1), dtype=np.uint64)

    # Create list to calculate the histogram entries in parallel
    result_list = []

    if (num_nuclei_prediction + 1) > num_threads:

        fraction = (num_nuclei_prediction + 1) / num_threads  # + 1 because the background is added

        for i in range(num_threads):

            result_list.append([pred,
                                gt,
                                nucleus_ids_prediction[int(i * fraction):int((i+1) * fraction)],
                                nucleus_ids_ground_truth])
    else:

        result_list.append([pred, gt, nucleus_ids_prediction, nucleus_ids_ground_truth])

    # Calculate the intersection histogram entries in parallel
    pool = Pool(num_threads)
    intersection_hist_entries = pool.map(intersection_hist_col, result_list)
    pool.close()

    # Pack the intersection histogram column lists into a single list
    for i in range(len(intersection_hist_entries)):
        for j in range(len(intersection_hist_entries[i])):
            col = intersection_hist_entries[i][j][0]
            if col == bg_pred:  # Move background column to the first
                col = 0
            intersections[:, col] = intersection_hist_entries[i][j][1]

    # Calculate Rand index and Jaccard index
    a, b, c, n = 0, 0, 0, len(prediction.flatten())

    for i in range(intersections.shape[0]):
        row_sum = np.sum(intersections[i, :], dtype=np.uint64)
        b += row_sum * (row_sum - 1) / 2
        for j in range(intersections.shape[1]):
            if i == 0:
                col_sum = np.sum(intersections[:, j], dtype=np.uint64)
                c += col_sum * (col_sum - 1) / 2
            a += intersections[i, j].astype(np.float64) * (intersections[i, j].astype(np.float64) - 1) / 2
    b -= a
    c -= a
    d = n * (n - 1) / 2 - a - b - c
    rand, jaccard = (a + d) / (a + b + c + d), (a + d) / (b + c + d)

    # Match objects with maximum intersections to detect split, merged, missing and spurious objects
    gt_matches, pred_matches, merged, missing, split, spurious = [], [], 0, 0, 0, 0
    for i in range(intersections.shape[0]):
        gt_matches.append(np.argmax(intersections[i, :]))
    for j in range(intersections.shape[1]):
        pred_matches.append(np.argmax(intersections[:, j]))
    gt_matches_counts, pred_matches_counts = Counter(gt_matches), Counter(pred_matches)
    for nucleus in gt_matches_counts:
        if nucleus == 0 and gt_matches_counts[nucleus] > 1:
            missing = gt_matches_counts[nucleus] - 1
        elif nucleus != 0 and gt_matches_counts[nucleus] > 1:
            merged += gt_matches_counts[nucleus] - 1
    for nucleus in pred_matches_counts:
        if nucleus == 0 and pred_matches_counts[nucleus] > 1:
            spurious = pred_matches_counts[nucleus] - 1
        elif nucleus != 0 and pred_matches_counts[nucleus] > 1:
            split += pred_matches_counts[nucleus] - 1

    # Aggregated Jaccard index and P_IoU (for the best IoU it does not matter if the predictions are matched to ground
    # truth nuclei or the other way around since the lowest threshold used later is 0.5, for the Jaccard-index it does).
    result_list = []  # Create list to find the best intersections and the corresponding unions in parallel

    if len(gt_matches) > num_threads:

        fraction = len(gt_matches) / num_threads

        for i in range(num_threads):

            result_list.append([pred, gt, intersections, list(range(int(i * fraction), int((i+1) * fraction)))])

    else:
        result_list.append([pred, gt, intersections, list(range(1, len(gt_matches)))])

    pool = Pool(num_threads)
    best_intersections_unions = pool.map(aggregated_iou_score, result_list)
    pool.close()

    aggregated_intersection, aggregated_union, used_nuclei_pred, iou = 0, 0, [], []
    for i in range(len(best_intersections_unions)):
        aggregated_intersection += best_intersections_unions[i][0]
        aggregated_union += best_intersections_unions[i][1]
        used_nuclei_pred = used_nuclei_pred + best_intersections_unions[i][2]
        iou = iou + best_intersections_unions[i][3]

    for nucleus in nucleus_ids_prediction[:-1]:  # Exclude background
        if nucleus not in used_nuclei_pred:
            aggregated_union += np.sum(pred == nucleus)

    aggregated_jaccard_index = aggregated_intersection / aggregated_union

    # Preallocate arrays for true positives, false negatives and true positives for each IoU threshold
    tp = np.zeros(shape=(10,), dtype=np.uint16)
    fp = np.zeros(shape=(10,), dtype=np.uint16)
    fn = np.zeros(shape=(10,), dtype=np.uint16)

    # Count true positives, false positives and false negatives for different IoU-thresholds th
    for i, th in enumerate(np.arange(0.5, 1.0, 0.05)):
        matches = iou > th

        # True positive: IoU > threshold
        tp[i] = np.count_nonzero(matches)

        # False negative: ground truth object has no associated predicted object
        fn[i] = num_nuclei_ground_truth - tp[i]

        # False positive: predicted object has no associated ground truth object
        fp[i] = num_nuclei_prediction - tp[i]

    # Precision for each threshold
    prec = np.divide(tp, (tp + fp + fn))

    # Precision for IoU-threshold = 0.5
    precision = np.divide(tp[0], tp[0] + fp[0]) if (tp[0] + fp[0]) > 0 else 0

    # Recall for IoU-threshold = 0.5
    recall = np.divide(tp[0], tp[0] + fn[0]) if (tp[0] + fn[0]) > 0 else 0

    # F-Score for IoU-threshold = 0.5
    f_score = 2 * np.divide(recall * precision, recall + precision) if (precision + recall) > 0 else 0

    # Mean precision (10 thresholds)
    piou = np.mean(prec)

    # Result dictionary
    results = {'Q_rand': rand,
               'Q_jaccard': jaccard,
               'Q_aggregated_jaccard': aggregated_jaccard_index,
               'Q_piou': piou,
               'Q_P': precision,
               'Q_R': recall,
               'Q_F': f_score,
               'N_gt': num_nuclei_ground_truth,
               'N_pred': num_nuclei_prediction,
               'N_split': split,
               'N_merged': merged,
               'N_miss': missing,
               'N_add': spurious,
               'tp': tp[0],
               'fp': fp[0],
               'fn': fn[0]
               }

    # Cell tracking challenge metric
    if path_ctc_software is not None:
        results['Q_ctc'] = ctc_seg_metric(prediction=prediction,
                                          ground_truth=ground_truth,
                                          path_evaluation_software=path_ctc_software)

    return results


def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values


def intersection_hist_col(result_list):
    """ Calculation of all intersections of a predicted nucleus with the ground truth nuclei and background. This
    results in a column of the intersection histogram needed to calculate some metrics.

    :param result_list: List containing the prediction, the ground_truth, a list of the ids of the predicted nuclei and
        a list of the ids of the ground truth nuclei (no explicit color channels for prediction and ground truth). The
        maximum label in the prediction and the ground truth is the maximum in these images.
        :type result_list: list
    :return: List of intersection histogram columns and the corresponding prediction nucleus ids.
    """

    # Unpack result list
    prediction, ground_truth, nucleus_ids_prediction, nucleus_ids_ground_truth = result_list

    intersection_hist_cols = []

    # Calculate for each predicted nucleus and the background the intersections with the ground truth nuclei and
    # background
    for nucleus_id_prediction in nucleus_ids_prediction:

        # Select predicted nucleus
        nucleus_prediction = (prediction == nucleus_id_prediction)

        # Intensity-coded intersections
        intersections = nucleus_prediction * ground_truth

        # Sum intersection with every ground truth nucleus
        hist = np.histogram(intersections,
                            bins=range(1, nucleus_ids_ground_truth[-1] + 2),
                            range=(1, nucleus_ids_ground_truth[-1] + 1))
        hist = hist[0]
        # Move background to front
        hist = hist[[len(hist) - 1] + list(range(len(hist) - 1))]
        intersection_hist_cols.append([nucleus_id_prediction, hist.astype(np.uint64)])

    return intersection_hist_cols


def aggregated_iou_score(result_list):
    """ Best intersection, the corresponding unions and the best intersection over union score for each ground truth
    nucleus. It is assumed that the biggest intersection corresponds to the best intersection over union score. There
    may be cases in which this does not hold. However, in that cases, the IoU is below 0.5 and it is insignificant for
    the piou metric.

    :param result_list:
    :return:
    """

    pred, gt, intersections, hist = result_list

    aggregated_intersection, aggregated_union, used_nuclei_pred, iou = 0, 0, [], []

    for i in hist:  # start from 1 to exclude the background matches

        if i != 0:
            best_intersection_nucleus = np.argmax(intersections[i, 1:]) + 1
            best_intersection = intersections[i, best_intersection_nucleus]
            aggregated_intersection += best_intersection
            union = np.sum((gt == i) | (pred == best_intersection_nucleus))
            aggregated_union += np.sum((gt == i) | (pred == best_intersection_nucleus))
            used_nuclei_pred.append(best_intersection_nucleus)
            iou.append(best_intersection / union)

    return [aggregated_intersection, aggregated_union, used_nuclei_pred, iou]


def calc_metrics(metric_scores_list):
    """ Calculate micro and macro metrics out of the metric scores of all test images.

    :param metric_scores_list: Metric scores for all test images
        :type metric_scores_list: list
    :return: metrics
    """

    N_split, N_miss, N_add, Q_P, Q_R, Q_F, N_gt, N_pred = [], [], [], [], [], [], [], []
    Q_rand, Q_jaccard, Q_aggregated_jaccard, Q_ctc, Q_piou = [], [], [], [], []
    tp, fp, fn = [], [], []

    for score in metric_scores_list:
        N_split.append(score['N_split']), N_miss.append(score['N_miss']), N_add.append(score['N_add'])
        Q_P.append(score['Q_P']), Q_R.append(score['Q_R']), Q_F.append(score['Q_F'])
        Q_rand.append(score['Q_rand']), Q_jaccard.append(score['Q_jaccard'])
        Q_aggregated_jaccard.append(score['Q_aggregated_jaccard'])
        if "Q_ctc" in score:
            Q_ctc.append(score['Q_ctc']), 
        Q_piou.append(score['Q_piou'])
        N_gt.append(score['N_gt']), N_pred.append(score['N_pred'])
        tp.append(score['tp']), fp.append(score['fp']), fn.append(score['fn'])

    N_split, N_miss, N_add = np.array(N_split), np.array(N_miss), np.array(N_add)
    N_gt, N_pred = np.array(N_gt), np.array(N_pred)
    tp, fp, fn = np.array(tp), np.array(fp), np.array(fn)
    Q_P_macro, Q_R_macro, Q_F_macro = np.mean(np.array(Q_P)), np.mean(np.array(Q_R)), np.mean(np.array(Q_F))
    Q_P_micro = np.sum(tp) / (np.sum(tp) + np.sum(fp)) if (np.sum(tp) + np.sum(fp)) > 0 else 0
    Q_R_micro = np.sum(tp) / (np.sum(tp) + np.sum(fn)) if (np.sum(tp) + np.sum(fn)) > 0 else 0
    Q_rand_macro, Q_jaccard_macro = np.mean(np.array(Q_rand)), np.mean(np.array(Q_jaccard))
    Q_aggregated_jaccard_macro = np.mean(np.array(Q_aggregated_jaccard))
    Q_ctc_macro, Q_piou_macro = np.mean(np.array(Q_ctc)), np.mean(np.array(Q_piou))

    metrics = {
        'Q_split_micro': float(np.sum(N_split) / np.sum(N_gt)),
        'Q_split_macro': float(np.mean(N_split / N_gt)),
        'Q_miss_micro': float(np.sum(N_miss) / np.sum(N_gt)),
        'Q_miss_macro': float(np.mean(N_miss / N_gt)),
        'Q_add_micro': float(np.sum(N_add) / np.sum(N_gt)),
        'Q_add_macro': float(np.mean(N_add / N_gt)),
        'N_gt': int(np.sum(N_gt)),
        'N_pred': int(np.sum(N_pred)),
        'Q_rand_macro': float(Q_rand_macro),
        'Q_jaccard_macro': float(Q_jaccard_macro),
        'Q_aggregated_jaccard_macro': float(Q_aggregated_jaccard_macro),
        'Q_ctc_macro': float(Q_ctc_macro),
        'Q_piou_macro': float(Q_piou_macro),
        'Q_P_micro': float(Q_P_micro),
        'Q_P_macro': float(Q_P_macro),
        'Q_R_micro': float(Q_R_micro),
        'Q_R_macro': float(Q_R_macro),
        'Q_F_macro': float(Q_F_macro),
        'Q_F_micro': float(2 * Q_P_micro * Q_R_micro / (Q_P_micro + Q_R_micro)) if (Q_P_micro + Q_R_micro) > 0 else 0
    }
    return metrics


def ctc_seg_metric(prediction, ground_truth, path_evaluation_software):
    """ Apply the cell tracking challenge SEG metric to a prediction - ground truth pair. For more information and the
    evaluation software see: http://celltrackingchallenge.net/evaluation-methodology/

    :param prediction: Prediction with intensity coded nuclei.
        :type prediction:
    :param ground_truth: Ground truth image with intensity coded nuclei.
        :type ground_truth:
    :param path_evaluation_software: Path to the evaluation software root directory.
        :type path_evaluation_software: str
    :return:
    """

    # Check for empty predictions
    num_nuclei_prediction = len(get_nucleus_ids(prediction))
    if num_nuclei_prediction == 0:
        return 0

    # Clear temporary result directory if exists
    if os.path.exists(path_evaluation_software + '/tmp'):
        shutil.rmtree(path_evaluation_software + '/tmp')

    # Create new clean result directory
    for directory in ['/tmp', '/tmp/01_GT', '/tmp/01_GT/SEG', '/tmp/01_RES']:
        os.mkdir(path_evaluation_software + directory)

    # Chose the executable in dependency of the operating system
    if platform.system() == 'Linux':
        path_seg_executable = path_evaluation_software + '/Linux/SEGMeasure'
    elif platform.system() == 'Windows':
        path_seg_executable = path_evaluation_software + '/Win/SEGMeasure.exe'
    elif platform.system() == 'Darwin':
        path_seg_executable = path_evaluation_software + '/Mac/SEGMeasure'
    else:
        raise ValueError('Platform not supported')

    # Check for missing nuclei ids in the prediction. To build the intersection histogram the nuclei_ids should range
    # from 1 to the number of nuclei. Copy the prediction to avoid changing it.
    pred = np.copy(prediction)

    if num_nuclei_prediction != pred.max():

        hist = np.histogram(prediction, bins=range(1, pred.max() + 2), range=(1, pred.max() + 1))

        # Find missing values
        missing_values = np.where(hist[0] == 0)[0]

        # Decrease the ids of the nucleus with higher id than the missing. Reverse the list to avoid problems in case
        # of multiple missing objects
        for th in reversed(missing_values):
            pred[pred > th] = pred[pred > th] - 1

    # Temporarily save the prediction and the ground truth with the naming convention needed for the evaluation software
    tiff.imsave(path_evaluation_software + '/tmp/01_GT/SEG/man_seg000.tif', ground_truth.astype(np.uint16))
    tiff.imsave(path_evaluation_software + '/tmp/01_RES/mask000.tif', pred.astype(np.uint16))

    # Apply the evaluation software to calculate the cell tracking challenge SEG measure
    output = subprocess.Popen([path_seg_executable, path_evaluation_software + '/tmp', '01', '3'],
                              stdout=subprocess.PIPE)
    result, _ = output.communicate()
    seg_measure = re.findall(r'\d\.\d*', result.decode('utf-8'))
    seg_measure = float(seg_measure[0])

    # Remove the temporary folder
    shutil.rmtree(path_evaluation_software + '/tmp')

    return seg_measure
