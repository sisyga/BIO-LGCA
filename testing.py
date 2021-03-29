import numpy as np
from lgca import get_lgca
from lgca.interactions import coll_migr
from matplotlib import pyplot as plt


lgca = get_lgca(geometry='lin', interaction='alignment', restchannels=2, density=.25)
lgca.beta_al = 2
lgca.beta_ag = 0
lgca.beta_s = 3
lgca.n_crit = 4
lgca.interaction = coll_migr

lgca.timeevo(100, record=True)

fig, axs = plt.subplots(ncols=2, sharey=True)
plt.sca(axs[0])
lgca.plot_density(cbar=False, cmap='viridis', vmax=lgca.n_crit)
plt.sca(axs[1])
lgca.plot_flux()
plt.show()