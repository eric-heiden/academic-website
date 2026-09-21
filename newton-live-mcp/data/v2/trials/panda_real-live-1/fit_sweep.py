from fit_physical import *
for floor in [.01,.02,.03,.04,.05]:
    fit(name=f'floor_{floor}',arm_min=floor,margin=1e-4,reg=1e-5)
for reg in [.0001,.001,.01]:
    fit(name=f'reg_{reg}',arm_min=.02,margin=1e-4,reg=reg)
for e in np.unique(ep):
    fit(name=f'loo_{e}',mask=np.repeat(ep!=e,7),arm_min=.02,margin=1e-4,reg=1e-4)
