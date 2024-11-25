import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as matplotlib_inline
import matplotlib.ticker as ticker
import os
import sys

src_path = os.path.abspath('../..')
if src_path not in sys.path:
    sys.path.append(src_path)

matplotlib_inline.rcParams.update({
    'text.usetex': True,             # Use LaTeX for all text rendering
    'font.family': 'libertine',          # Use LaTeX's default font
    'font.serif': ['Computer Modern'], # Match LaTeX default serif font
    'font.size': 12,                 # Adjust as needed to match your LaTeX document
    'axes.labelsize': 12,            # Match LaTeX font sizes for axes labels
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

from enum import Enum
plt.rcParams['svg.fonttype'] = 'none'
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
from src.tools.tol_colors import tol_cmap, tol_cset

def plot_reconstruction(u_star, u_bar, saveAsSVG=False):
    cmap = tol_cset('bright')
    timePoints = np.linspace(0, 1, len(u_star[:,0]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))
    ax1.spines[["left", "bottom"]].set_position("zero")
    ax1.spines[['top', 'right']].set_visible(False)
    #ax1.plot(1, 0, ">k", transform=ax1.get_yaxis_transform(), clip_on=False)
    #ax1.plot(0, 1, "^k", transform=ax1.get_xaxis_transform(), clip_on=False)
    
    ax2.spines[["left", "bottom"]].set_position("zero")
    ax2.spines[['top', 'right']].set_visible(False)
    #ax2.plot(1, 0, ">k", transform=ax2.get_yaxis_transform(), clip_on=False)
    #ax2.plot(0, 1, "^k", transform=ax2.get_xaxis_transform(), clip_on=False)
    
    ax1.plot(timePoints, u_star[:, 0], label='$u^*_1$', color=cmap[0], clip_on=False)
    ax1.plot(timePoints, u_bar[:, 0], label='$\\bar{u}_1$', color=cmap[1], clip_on=False)
    ax2.plot(timePoints, u_star[:, 1], label='$u^*_2$', color=cmap[0], clip_on=False)
    ax2.plot(timePoints, u_bar[:, 1], label='$\\bar{u}_2$', color=cmap[1], clip_on=False)
    
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))

    ax1.set_xlabel(r'$t$')
    ax2.set_xlabel(r'$t$')
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    if saveAsSVG:
        fig.savefig(r'/mnt/d/HU/Masterarbeit/Plots/reconstruction.svg')
    fig.show()
        
def plot_duals(array, type, saveAsSVG=False):
    cmap = tol_cset('bright')
    timePoints = np.linspace(0, 1, len(array))
    f = plt.figure()
    f.set_figheight(5)

    ax = plt.gca()  # Get the current Axes instance
    ax.spines[["left", "bottom"]].set_position("zero")
    ax.spines[['top', 'right']].set_visible(False)
    #ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))

    plt.xlabel(r'$t$')
    plt.axhline(y=1, linestyle='-', color='grey')
    plt.plot(timePoints, array, color=cmap[0])
    if saveAsSVG:
        if type == 'P':
            plt.savefig(r'/mnt/d/HU/Masterarbeit/Plots/secondDual.svg')
        if type == 'p':
            plt.savefig(r'/mnt/d/HU/Masterarbeit/Plots/firstDual.svg')
    plt.show()
        
def plot_activeSet(array, saveAsSVG=False):
    cmap = tol_cset('bright')
    iterations = np.arange(0, len(array))
    f = plt.figure()
    f.set_figheight(5)

    ax = plt.gca()  # Get the current Axes instance
    #ax.spines[["left", "bottom"]].set_position("zero")
    ax.spines[['top', 'right']].set_visible(False)
    #ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel(r'$k$')
    plt.plot(iterations, array, color=cmap[0])
    if saveAsSVG:
            plt.savefig(r'/mnt/d/HU/Masterarbeit/Plots/activeSet.svg')
    plt.show()
    
def plot_convergence(constraintViolation, residual, saveAsSVG=False):
    cmap = tol_cset('bright')
    iterations = np.arange(0, len(constraintViolation))
    f = plt.figure()
    f.set_figheight(5)

    ax = plt.gca()  # Get the current Axes instance
    #ax.spines[["left", "bottom"]].set_position("zero")
    ax.spines[['top', 'right']].set_visible(False)
    #ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '' if val == 0 else f'{val:g}'))
    #ax.set_xlim(left=0)
    #ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel(r'$k$')
    ax.set_yscale('log')
    plt.plot(iterations, constraintViolation, label=r'$\Psi(u_k)$', color=cmap[0])
    plt.plot(iterations, residual, label=r'$\bar{r_j}(u_k)$', color=cmap[1])
    plt.legend(loc="best")
    if saveAsSVG:
            plt.savefig(r'/mnt/d/HU/Masterarbeit/Plots/convergence.svg')
    plt.show()