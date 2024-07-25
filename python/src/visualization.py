from dolfinx import fem, mesh, plot, io
import numpy as np
import matplotlib as mpl
import pyvista
from src.solveStateEquation import getSourceTerm
from typing import List

def plot_function(function, T, dt, label='function', idx=None):
    num_steps = int(T/dt)
    function_array = np.zeros((num_steps, len(function(0))))
    timesteps = np.linspace(0, T, num_steps)
    for i, t in enumerate(timesteps):
        function_array[i,:] = function(t)
    if idx == None:
        plot_array(function_array, T, label=label)
    else:
        plot_array(function_array[idx], T, label=label)

def plot_array(array: np.ndarray, T, label='function'):
    timepoints = np.linspace(0, T, len(array))
    mpl.pyplot.figure(figsize=(10, 6))
    mpl.pyplot.plot(timepoints, array, marker='o', linestyle='-', color='b')
    mpl.pyplot.xlabel('Time')
    mpl.pyplot.ylabel(label)
    mpl.pyplot.grid(True)
    mpl.pyplot.show()

def printControlFunction(V: fem.FunctionSpace, s1, s2, x1, x2, T=1, dt=0.01, alpha=0.1, slowMoFactor=1):
    g1 = getSourceTerm(V, x1, alpha)
    g2 = getSourceTerm(V, x2, alpha)
    g = []
    interval = np.linspace(0, T, int(T / dt))
    for t in interval:
        local = fem.Function(V)
        local.x.array[:] = s1(t) * g1.x.array + s2(t) * g2.x.array
        g.append(local)
    timeDependentVariableToGif(g, 'control_signal.gif', 'control_function', T=T, slowMoFactor=slowMoFactor)

def timeDependentVariableToGif(data: List[fem.Function], filename, varname="function", slowMoFactor=1, T=1):
    MAX_FPS = 30
    def getMaximum(data):
        value = -100
        for function in data:
            value = max(function.x.array.max(), value)
        return value

    def getMinimum(data):
        value = 100
        for function in data:
            value = min(function.x.array.min(), value)
        return value
    
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(data[0].function_space))
    grid.point_data[varname] = data[0].x.array
    grid.set_active_scalars(varname)
    warped = grid.warp_by_scalar(varname, factor=1)
    plotter = pyvista.Plotter()
    dt = T / float(len(data))
    fps = T / dt / slowMoFactor
    if fps > 30:
        sparse_fps_rate = np.floor(fps / 30)
        fps = fps / sparse_fps_rate
    else:
        sparse_fps_rate = 1
    plotter.open_gif(filename, fps=int(fps))
    plotter.set_scale(zscale=0.1)
    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)

    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                        position_x=0.1, position_y=0.8, width=0.8, height=0.1)
    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[getMinimum(data),getMaximum(data)])
    
    for idx, function in enumerate(data):
        if (idx % sparse_fps_rate != 0):
            continue
        new_warped = grid.warp_by_scalar(varname, factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data[varname][:] = function.x.array
        plotter.write_frame()

    plotter.show()
    plotter.close()