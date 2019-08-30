import random
import os
import recommender as rc

users=2100
rcTemp=rc.recommender()

testCase=1

for i in range(testCase):

	uid_random=random.randint(1,users)
	X=[[uid_random]]
	results=rcTemp.predict(X,'featureName')
	
	print('USER ID [%d]' %uid_random)
	print(results)

