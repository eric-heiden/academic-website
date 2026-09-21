from fit_model import *
arm=[.04,.02,.04,.03,.04,.03,.02]
for floor in [0,.2,.3,.4,.5]:
    visc=[1e-7]*7;visc[4]=max(floor,1e-7)
    x,status,obj=solve(1e-5,arm_min=arm,visc_min=visc)
    name='trade_'+str(floor)
    (ROOT/(name+'.json')).write_text(json.dumps(config_from_x(x),indent=2)+'\n')
    np.save(ROOT/(name+'.npy'),x)
    print(name,status,'rmse',score(x),'joint',x[70:].round(5).tolist(),flush=True)
