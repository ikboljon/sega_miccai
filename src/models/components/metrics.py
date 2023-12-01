from monai.metrics.hausdorff_distance import compute_hausdorff_distance
import numpy as np
import SimpleITK as sitk

def compute_agg_dice(intermediate_results):
    """
    Compute the aggregate dice score from the intermediate results
    """
    aggregate_results = {}
    TP1s = [v["TP1"] for v in intermediate_results]
    TP2s = [v["TP2"] for v in intermediate_results]
    vol_sum1s = [v["vol_sum1"] for v in intermediate_results]
    vol_sum2s = [v["vol_sum2"] for v in intermediate_results]
    DSCagg1 = 2 * np.sum(TP1s) / np.sum(vol_sum1s)
    DSCagg2 = 2 * np.sum(TP2s) / np.sum(vol_sum2s)
    aggregate_results['AggregatedDsc'] = {
        'GTVp': DSCagg1,
        'GTVn': DSCagg2,
        'mean': np.mean((DSCagg1, DSCagg2)),
    }

    return aggregate_results

def compute_volumes(im):
    """
    Compute the volumes of the GTVp and the GTVn
    """
    spacing = im.GetSpacing()
    voxvol = spacing[0] * spacing[1] * spacing[2]
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(im, im)
    nvoxels1 = stats.GetCount(1)
    nvoxels2 = stats.GetCount(2)
    return nvoxels1 * voxvol, nvoxels2 * voxvol


def get_intermediate_metrics(groundtruth, prediction):
    """
    Compute intermediate metrics for a given groundtruth and prediction.
    These metrics are used to compute the aggregate dice.
    """
    groundtruth = groundtruth.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()
    
    groundtruth = sitk.GetImageFromArray(groundtruth)
    prediction = sitk.GetImageFromArray(prediction)
    
    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(groundtruth, prediction)

    DSC1 = overlap_measures.GetDiceCoefficient(1)
    DSC2 = overlap_measures.GetDiceCoefficient(2)

    vol_gt1, vol_gt2 = compute_volumes(groundtruth)
    vol_pred1, vol_pred2 = compute_volumes(prediction)

    vol_sum1 = vol_gt1 + vol_pred1
    vol_sum2 = vol_gt2 + vol_pred2
    TP1 = DSC1 * (vol_sum1) / 2
    TP2 = DSC2 * (vol_sum2) / 2
    return {
        "TP1": TP1,
        "TP2": TP2,
        "vol_sum1": vol_sum1,
        "vol_sum2": vol_sum2,
    }



def dice_hecktor(pred_list, label_list):
    agg_dsc = []
    

    for i in range(1, 3):
        for pred, label in zip(pred_list, label_list):
            pred = pred.squeeze().cpu().numpy()
            label = label.squeeze().cpu().numpy()
            
            numerator, denominator = None, None

            organ_arr_result = np.where(pred == i, 1, 0)
            organ_arr_refer = np.where(label == i, 1, 0)

            curr_numerator = np.sum(np.multiply(organ_arr_result, organ_arr_refer))
            curr_denominator = np.sum(organ_arr_result + organ_arr_refer)

            if numerator is None:
                numerator = curr_numerator
                denominator = curr_denominator
            else:
                numerator += curr_numerator
                denominator += curr_denominator

        agg_dsc.append(2 * numerator / denominator)

    agg_dsc_GTVp = agg_dsc[0]
    agg_dsc_GTVn = agg_dsc[1]
    avg_dsc = (agg_dsc_GTVp + agg_dsc_GTVn) / 2

    return agg_dsc_GTVp, agg_dsc_GTVn, avg_dsc


def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-5)

    return score.mean()


def recall(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)
    recall = true_positives / (all_positives + 1e-5)

    return recall.mean()


def precision(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)
    precision = true_positives / (all_positive_calls + 1e-5)

    return precision.mean()


def hausdorff(input, target, percentile=95): # [batch_size, x, y, z]
    target = target.unsqueeze(1)
    input = input.unsqueeze(1)
    
    return compute_hausdorff_distance(input, target, include_background=True, percentile=percentile)