from fit_physical import *
floors=np.array([.025,.01,.025,.015,.03,.02,.015])
fit(name='balanced',arm_min=floors,margin=1e-4,reg=1e-4)
for e in np.unique(ep):
    fit(name=f'balanced_loo_{e}',mask=np.repeat(ep!=e,7),arm_min=floors,margin=1e-4,reg=1e-4)
