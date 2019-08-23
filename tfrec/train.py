import wals


def train(params,train_sparse):

	latent_factors = params['latent_factors']
	num_iters = params['num_iters']
	reg = params['reg']
	unobserved_weight = params['unobserved_weight']
	use_weight=params['use_weight']
	weight_type = params['weight_type']
	feature_weight_exp = params['feature_weight_exp']
	feature_weight_lin = params['feature_weight_lin']


	input_tensor, row_factor, col_factor, model= wals.wals_model(train_sparse,latent_factors,reg,
																unobserved_weight,use_weight,
																weight_type,
																feature_weight_exp,
																feature_weight_lin)
		
	sess=wals.simple_train(model,input_tensor,num_iters)
	
	user_factor=row_factor.eval(session=sess)
	item_factor=col_factor.eval(session=sess)

	sess.close()

	return user_factor,item_factor


