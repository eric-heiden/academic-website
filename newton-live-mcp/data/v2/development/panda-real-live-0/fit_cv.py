from fit_model import *
for arm5,visc5 in [(.02,.3),(.0275,.15),(.0275,.225),(.035,.15)]:
    for e in ids:
        arm=[.025,.01,.025,.015,arm5,.025,.0175];visc=[1e-7]*7;visc[4]=visc5
        x,status,obj=solve(1e-5,train=E!=e,arm_min=arm,visc_min=visc)
        name=f'cv_{arm5}_{visc5}_{e}'
        (ROOT/(name+'.json')).write_text(json.dumps(config_from_x(x),indent=2)+'\n')
        print(name,status,'validation_rmse',score(x)[str(e)],flush=True)
