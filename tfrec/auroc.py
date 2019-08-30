
from collections import defaultdict

import model
import random
import numpy as np
import time 

from sklearn.metrics import roc_auc_score

def calculate_total_auroc(row_factor,col_factor,already_rated_idx,test_sparse,n_items):
	
	positive_items,negative_items=construct_positive_negative(already_rated_idx,test_sparse,n_items)



	def calculate_auroc(user_idx):
		positive_items_idx=positive_items[user_idx]
		negative_items_idx=negative_items[user_idx]
		y_true=np.concatenate((np.ones_like(positive_items_idx),np.zeros_like(negative_items_idx)))
		y_pred=(np.concatenate((col_factor[positive_items_idx],col_factor[negative_items_idx]))).dot(row_factor[user_idx])


		return roc_auc_score(y_true,y_pred)
		
	aurocs={}
	for user_idx in positive_items.keys():
		aurocs[user_idx]=calculate_auroc(user_idx)

#
	auroc_values=list(aurocs.values())

	_avg=np.mean(auroc_values)
	_max=np.max(auroc_values)
	_min=np.min(auroc_values)
	
	print('AVG[%f]' %_avg)
	return aurocs,_avg



	
	

	

def construct_positive_negative(already_rated_idx,test_sparse,n_items):

	positive_items=defaultdict(list)
	for i in range(len(test_sparse.row)):
		user_idx=test_sparse.row[i]
		item_idx=test_sparse.col[i]
		
		positive_items[user_idx].append(item_idx)
	
	negative_items=defaultdict(list)
	
	for user_idx, positive_item_list in positive_items.items():
		neg_items=[]
		already_rated_items=already_rated_idx[user_idx]
		pos_len=len(positive_item_list)
			
		while len(neg_items) < pos_len:
			negative_item_idx_list=np.random.choice(n_items,pos_len,replace=False)
			negative_item_idx_list=[ item_idx for item_idx in negative_item_idx_list if item_idx not in already_rated_items]
			neg_items=neg_items+negative_item_idx_list
		
		negative_items[user_idx]=neg_items[:pos_len]


		
	return positive_items,negative_items
		
	
