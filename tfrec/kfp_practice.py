import kfp
from kfp import dsl


def run(UID):
	return dsl.ContainerOp(
				name='run_name',
				image='docker-registry.linecorp.com/lp60409/tfrec:0.3',
				command=['python3', 'recommender.py'],
				arguments=[UID]
#				file_outputs={'results':'/ymp/results'}

			)
@dsl.pipeline( # dsl decoreator
  name='simple_rec_pipeline',
  description='simple rec pipeline'
)
def run_pipeline(uid):
	run(uid)


kfp.compiler.Compiler().compile(run_pipeline,'run_pipeline2.zip')
