

def accuracy(pred_tumors_scores, gt_tumor_types):
	batch_size = gt_tumor_types.size(0)
	_, pred_tumors_types = pred_tumors_scores.max(dim=1)
	correct_predictions = (pred_tumors_types == gt_tumor_types).count_nonzero()
	acc = correct_predictions / batch_size
	return acc
