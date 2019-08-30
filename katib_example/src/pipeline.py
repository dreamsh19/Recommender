import train
import model
import wals
import faster_auroc
import numpy as np
import datetime
import argparse

np.seterr(divide='ignore', invalid='ignore')


TESTSET_RATIO=0.1
 
LOG_RATINGS = 0
LINEAR_RATINGS = 1


DATA_FILE='lastfm_data/user_artists_log.dat'


DEFAULT_PARAMS = {
    'use_weight': True,
	'latent_factors': 5,
	'num_iters': 20,
	'reg': 0.07,
	'unobserved_weight': 0.01,
	'weight_type': LOG_RATINGS,
    'feature_weight_lin': 130.0,
    'feature_weight_exp': 0.08
}

OPTIMIZED_PARAMS = {
    'use_weight': True,
    'latent_factors': 20,
    'num_iters': 10,
	'reg': 0.07,
	'unobserved_weight': 0.001,
    'weight_type': LINEAR_RATINGS,
    'feature_weight_lin': 130.0,
    'feature_weight_exp': 0.08
}


MAX_NUM_OF_RECOMMEND_ITEMS=10


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--latent_factors',default=DEFAULT_PARAMS['latent_factors'],type=int)
	parser.add_argument('--CV',default=10,type=int)
	parser.add_argument('--reg',default=DEFAULT_PARAMS['reg'],type=float)


	args=parser.parse_args()

	DEFAULT_PARAMS['latent_factors']=args.latent_factors
	DEFAULT_PARAMS['reg']=args.reg



	print('===============pipeline===============')



	CV = args.CV

	for cv in range(CV):
		print('CV=%d' % cv)
		already_rated_idx, user_map, item_map, train_sparse, test_sparse = model.ratings_train_test(DATA_FILE)
		row_factor, col_factor = train.train(DEFAULT_PARAMS, train_sparse)
		auroc = faster_auroc.calculate_auroc(row_factor, col_factor, already_rated_idx, test_sparse, len(item_map))

		print('AUROC=%f' % np.mean(auroc))
		print('AUROC_MID=%f' %np.median(auroc))


if __name__=='__main__':
	main()




	




