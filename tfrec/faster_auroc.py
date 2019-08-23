import model
import train
from load_file import load_file
from collections import defaultdict
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import time

def calculate_auroc(row_factor,col_factor,already_rated_idx,test_sparse,n_items):


	prediction_matrix=row_factor.dot(col_factor.T)

	positive_items_dic=defaultdict(list)

	auroc_dic={}
	for i in range(len(test_sparse.row)):
		user_idx = test_sparse.row[i]
		item_idx = test_sparse.col[i]
		positive_items_dic[user_idx].append(item_idx)


	start_time = time.time()
	for user_idx, positive_items_idxs in positive_items_dic.items():
		already_rated_item_idxs=already_rated_idx[user_idx]
		num_of_positive_items=len(positive_items_idxs)

		negative_items_idxs=[]
		while len(negative_items_idxs) < num_of_positive_items:
			negative_items_idxs=np.random.choice(n_items,num_of_positive_items,replace=True)
			negative_items_idxs=[ idx for idx in negative_items_idxs if idx not in already_rated_item_idxs]

		negative_items_idxs=negative_items_idxs[:num_of_positive_items]

		y_true = np.concatenate((np.ones_like(positive_items_idxs), np.zeros_like(negative_items_idxs)))
		y_pred = prediction_matrix[user_idx][np.concatenate((positive_items_idxs,negative_items_idxs))]
	
		auroc_=roc_auc_score(y_true, y_pred)
		auroc_dic[user_idx]=auroc_

	elapsed_time = time.time() - start_time
#	print('calculating the whole auroc takes %f' % elapsed_time)
	
	return np.mean(list(auroc_dic.values()))
