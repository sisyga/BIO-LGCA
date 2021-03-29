import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from base import *
except ModuleNotFoundError:
    from .base import *


class LGCA_1D(LGCA_base):
    """
    1D version of an LGCA.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation', 'parameter_controlled_diffusion',
                    'random_walk', 'persistent_motion', 'birthdeath']
    velocitychannels = 2
    c = np.array([1., -1.])[None, ...]

    def set_dims(self, dims=None, nodes=None, restchannels=0):
        if nodes is not None:
            self.l, self.K = nodes.shape
            self.restchannels = self.K - self.velocitychannels
            self.dims = self.l,
            return

        elif dims is None:
            dims = 100

        if isinstance(dims, int):
            self.l = dims

        else:
            self.l = dims[0]

        self.dims = self.l,
        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels

    def init_nodes(self, density, nodes=None):
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.bool)
        if nodes is None:
            self.random_reset(density)

        else:
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(np.bool)

    def init_coords(self):
        self.nonborder = (np.arange(self.l) + self.r_int,)
        self.xcoords = np.arange(self.l + 2 * self.r_int) - self.r_int

    def propagation(self):
        """
        :return:
        """
        newnodes = np.zeros_like(self.nodes)
        # resting particles stay
        newnodes[:, 2:] = self.nodes[:, 2:]

        # prop. to the right
        newnodes[1:, 0] = self.nodes[:-1, 0]

        # prop. to the left
        newnodes[:-1, 1] = self.nodes[1:, 1]

        self.nodes = newnodes

    def apply_pbc(self):
        self.nodes[:self.r_int, :] = self.nodes[-2 * self.r_int:-self.r_int, :]
        self.nodes[-self.r_int:, :] = self.nodes[self.r_int:2 * self.r_int, :]

    def apply_rbc(self):
        self.nodes[self.r_int, 0] += self.nodes[self.r_int - 1, 1]
        self.nodes[-self.r_int - 1, 1] += self.nodes[-self.r_int, 0]
        self.apply_abc()

    def apply_abc(self):
        self.nodes[:self.r_int, :] = 0
        self.nodes[-self.r_int:, :] = 0

    def nb_sum(self, qty):
        sum = np.zeros(qty.shape)
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        return sum

    def gradient(self, qty):
        return np.gradient(qty, 2)[..., None]

    def channel_weight(self, qty):
        weights = np.zeros(qty.shape + (self.velocitychannels,))
        weights[:-1, ..., 0] = qty[1:, ...]
        weights[1:, ..., 1] = qty[:-1, ...]
        return weights

    def setup_figure(self, tmax, figindex=None, figsize=(8, 8), tight_layout=True):
        if figindex is None:
            fig = plt.gcf()
            fig.set_size_inches(figsize)
            fig.set_tight_layout(tight_layout)

        else:
            fig = plt.figure(num=figindex)
            fig.set_size_inches(figsize)
            fig.set_tight_layout(tight_layout)

        ax = plt.gca()
        xmax = self.xcoords.max() - 0.5 * self.r_int
        xmin = self.xcoords.min() + 0.5 * self.r_int
        ymax = tmax - 0.5
        ymin = -0.5
        plt.xlim(xmin, xmax)
        plt.ylim(ymax, ymin)
        ax.set_aspect('equal')

        plt.xlabel('Lattice node $r \\, (\\varepsilon)$')
        plt.ylabel('Time $k'
                   '\\, (\\tau)$')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_autoscale_on(False)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        return fig, ax

    def plot_density(self, density_t=None, cmap='hot_r', cbar=True, vmax=None, **kwargs):
        if density_t is None:
            density_t = self.dens_t

        if vmax is None:
            K = self.K

        else:
            K = vmax

        tmax = density_t.shape[0]
        fig, ax = self.setup_figure(tmax, **kwargs)
        cmap = cmap_discretize(cmap, vmax+1)
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=vmax, cmap=cmap)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=.3, pad=0.1)
            cbar = colorbar_index(ncolors=3 + self.restchannels, cmap=cmap, use_gridspec=True, cax=cax)
            cbar.set_label('Particle number $n$')
            plt.sca(ax)

        return plot

    def plot_flux(self, nodes_t=None, **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t

        dens_t = nodes_t.sum(-1) / nodes_t.shape[-1]
        tmax, l = dens_t.shape
        flux_t = nodes_t[..., 0].astype(int) - nodes_t[..., 1].astype(int)

        rgba = np.zeros((tmax, l, 4))
        rgba[dens_t > 0, -1] = 1.
        rgba[flux_t == 1, 0] = 1.
        rgba[flux_t == -1, 2] = 1.
        rgba[flux_t == 0, :-1] = 0.
        fix, ax = self.setup_figure(tmax, **kwargs)
        plot = ax.imshow(rgba, interpolation='None', origin='upper')
        plt.xlabel(r'Lattice node $r \, (\varepsilon)$', )
        plt.ylabel(r'Time $k \, (\tau)$')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.tick_top()
        plt.tight_layout()
        return plot
