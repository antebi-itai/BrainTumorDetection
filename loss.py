import torch


def calc_accuracy(pred_tumors_scores, gt_tumor_types, tumor_type=None):
	batch_size = gt_tumor_types.size(0)
	_, pred_tumors_types = pred_tumors_scores.max(dim=1)
	if tumor_type is None:
		correct_predictions = (pred_tumors_types == gt_tumor_types).count_nonzero()
		acc = correct_predictions / batch_size
	else:
		correct_tumor_predictions = ((pred_tumors_types == gt_tumor_types) & (gt_tumor_types == tumor_type)).count_nonzero()
		tumor_batch_size = (gt_tumor_types == tumor_type).count_nonzero()
		if tumor_batch_size != 0:
			acc = correct_tumor_predictions / tumor_batch_size
		else:
			acc = torch.tensor(1, dtype=pred_tumors_scores.dtype, device=pred_tumors_scores.device)
	return round(acc.item(), 2)
