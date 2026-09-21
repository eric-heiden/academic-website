from fit_model import *
for arm5 in [.02,.0275,.035]:
    for visc5 in [.15,.225,.3]:
        arm=[.025,.01,.025,.015,arm5,.025,.0175];visc=[1e-7]*7;visc[4]=visc5
        x,status,obj=solve(1e-5,arm_min=arm,visc_min=visc)
        name=f'balance_{arm5}_{visc5}'
        (ROOT/(name+'.json')).write_text(json.dumps(config_from_x(x),indent=2)+'\n')
        np.save(ROOT/(name+'.npy'),x)
        print(name,status,'rmse',score(x),flush=True)
