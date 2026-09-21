from balanced_fit import *
arms=np.array([.025,.015,.025,.015,.03,.015,.02])
for omit in [2,3,4]:
 x=estimate(arms,omit=omit,physical_margin=.0001,joint_weights=[1,1,1,1,2,2,2])
 conf=to_config(x)
 run(conf,f'final method leave out recording {omit}')
final=json.loads(Path('weighted-wrist2.json').read_text())
run(final,'selected final configuration, complete training verification')
