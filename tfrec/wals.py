import numpy as np

import tensorflow as tf
from tensorflow.contrib.factorization.python.ops import factorization_ops

LINEAR_RATINGS=0
LOG_RATINGS=1

def wals_model(data, latent_factor, unobserved_weight, reg, use_weight, weight_type, feature_weight_exp, feature_weight_lin):

	row_weights=None
	col_weights=None

	num_rows=data.shape[0]
	num_cols=data.shape[1]


	if use_weight:
		assert feature_weight_exp is not None
		row_weights=np.ones(num_rows)
		col_weights=make_weights(data,weight_type,feature_weight_lin,feature_weight_exp,0)
	
	row_factor=None
	col_factor=None
	
	
	with tf.Graph().as_default():

		input_tensor=tf.SparseTensor(indices=list(zip(data.row,data.col)),
									values=(data.data).astype(np.float32),
									dense_shape=data.shape)
		
		model=factorization_ops.WALSModel(num_rows,num_cols,latent_factor,
											unobserved_weight=unobserved_weight,
											regularization=reg,
											row_weights=row_weights,
											col_weights=col_weights)
		row_factor=model.row_factors[0]
		col_factor=model.col_factors[0]

	return input_tensor,row_factor,col_factor,model


def make_weights(data,weight_type,feature_weight_lin,feature_weight_exp,axis):
	
	frac=np.array(1.0/(data>0.0).sum(axis))
	
	frac[np.ma.masked_invalid(frac).mask]=0.0
	if weight_type==LOG_RATINGS:
		weights=np.array(np.power(frac,feature_weight_exp)).flatten()
	else:
		weights=np.array(feature_weight_lin*frac).flatten()

	assert np.isfinite(weights).sum()==weights.shape[0]

	return weights


def simple_train(model, input_tensor, num_iters):
  
#	sess = tf.Session(graph=input_tensor.graph)
	sess = tf.compat.v1.Session(graph=input_tensor.graph)
	
	with input_tensor.graph.as_default():
		row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
		col_update_op = model.update_col_factors(sp_input=input_tensor)[1]
		
		sess.run(model.initialize_op)
		sess.run(model.worker_init)

		for _ in range(num_iters):
			sess.run(model.row_update_prep_gramian_op)
			sess.run(model.initialize_row_update_op)
			sess.run(row_update_op)
			sess.run(model.col_update_prep_gramian_op)
			sess.run(model.initialize_col_update_op)
			sess.run(col_update_op)
	
	return sess
