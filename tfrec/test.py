import recommender

rec_unit=recommender.recommender()

for uid in range(100):
	print(uid)
	res=rec_unit.predict([[uid]],'feature_name')
	print(res)
