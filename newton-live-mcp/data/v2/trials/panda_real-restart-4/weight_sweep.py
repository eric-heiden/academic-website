from balanced_fit import *
arms=np.array([.025,.015,.025,.015,.03,.015,.02])
for label,weights,margin,huber in [('margin',[1]*7,.0001,0),('wrist2',[1,1,1,1,2,2,2],.0001,0),('wrist4',[1,1,1,1,4,2,3],.0001,0),('robust',[1,1,1,1,2,2,2],.0001,.2)]:
 x=estimate(arms,physical_margin=margin,joint_weights=weights,huber=huber)
 conf=to_config(x)
 Path(f'weighted-{label}.json').write_text(json.dumps(conf,indent=2)+'\n')
 print('scores',label,json.dumps(score(x)),flush=True)
 run(conf,f'balanced {label}')
