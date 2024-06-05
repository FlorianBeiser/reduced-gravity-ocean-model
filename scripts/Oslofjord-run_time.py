# %% [markdown]
# ```
# This script runs the 6h simulations to caluclate wall time expenses
# Copyright (C) 2022 - 2023 SINTEF Digital
# Copyright (C) 2022 - 2023 Norwegian Meteorological Institute
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ```

# %%
import sys
gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
repo = git.Repo(gpuocean_path)
print("GPUOcean code from:", repo.head.object.hexsha, "on branch", repo.active_branch.name)

# %% [markdown]
# # Oslofjord
# Testing of Nils projected files

# %%

import os
import sys
import time

#Import packages we need
import numpy as np
from netCDF4 import Dataset
import datetime, copy
from IPython.display import display


# %%
from gpuocean.utils import Common, NetCDFInitialization

# %%
gpu_ctx = Common.CUDAContext()

# %% 
# Path to the test file
source_url = "/sintef/data/OsloFjord/test_polstere_1h_0007.nc"

# %%
import xarray as xr
ds = xr.open_dataset(source_url)

from netCDF4 import Dataset
nc = Dataset(source_url)

# %%
# ## Generating GPUOcean Simulation from Input
data_args = {}

# %%
dimY, dimX = ds.h.data.shape
x0, x1, y0, y1 = 5, dimX-5, 175, dimY-5

# %%
import pyproj
proj_str = nc["projection"].proj4

proj = pyproj.Proj(proj_str)

lat_rho = nc.variables['lat_rho'][y0:y1, x0:x1]
lon_rho = nc.variables['lon_rho'][y0:y1, x0:x1]
x_rho, y_rho = proj(lon_rho, lat_rho, inverse = False)
x, y = x_rho[0], y_rho[:,0]

data_args['nx'] = (x1-x0)-4
data_args['ny'] = (y1-y0)-4

NX = data_args["nx"]+4
NY = data_args["ny"]+4

data_args['dx'] = np.average(x[1:] - x[:-1])
data_args['dy'] = np.average(y[1:] - y[:-1])

# %%
data_args["dx"], data_args["dy"]

# %%
data_args['angle'] = nc["angle"][y0:y1, x0:x1]
from gpuocean.utils import OceanographicUtilities
data_args['latitude'] = OceanographicUtilities.degToRad(lat_rho)
data_args["f"] = 0.0

# %%
data_args["t"] = 0

# %%
T = 2*24*3600 

# %%
mld = NetCDFInitialization.MLD(source_url, 1024, min_mld=1.5, max_mld=40, x0=x0, x1=x1, y0=y0, y1=y1, t=0)
mld = NetCDFInitialization.fill_coastal_data(mld)

# %% [markdown]
# #### Baroclinic model

# %%
H = 0.0

# %%
H_m_data = nc.variables['h'][y0-1:y1+1, x0-1:x1+1]
H_m_mask = (H_m_data == 0.0)
H_m = np.ma.array(H_m_data, mask=H_m_mask)
H_i = OceanographicUtilities.midpointsToIntersections(H_m, land_value=0.0, iterations=10)[0]

data_args["H"] = np.ma.array(H*np.ones_like(H_i), mask=H_i.mask.copy(), dtype=np.float32)


# %% [markdown]
# Artifical data

# %%
mld = np.ma.array( 15.0*np.ones_like(mld), mask=copy.copy(mld.mask) )

data_args["eta0"] = np.ma.array(mld.data - H, mask=copy.copy(mld.mask))

# %%
# Starting from lake at rest
data_args["hu0"] = np.ma.array(np.zeros_like(mld), mask=copy.copy(mld.mask))
data_args["hv0"] = np.ma.array(np.zeros_like(mld), mask=copy.copy(mld.mask))

# %%
data_args["g"] = 0.1
data_args["r"] = 1.0e-3#3.0e-3


# %%
data_args['boundary_conditions'] = Common.BoundaryConditions(north=3, south=3, east=3, west=3, spongeCells={'north':20, 'south': 20, 'east': 20, 'west': 20})

# %%
t_step = 3600
T_steps = int(np.ceil(T/t_step)+1)

ts = data_args["t"] + np.arange(0, T+1, step=t_step)
ts_steps = ts-ts[0]

# %%
ampl_v = 0.3 #Input
freq = 12*3600 
bc_v_ref = ampl_v * np.sin(2*np.pi*ts_steps/freq)[:,np.newaxis] 

bc_v = np.zeros((T_steps, NX))
bc_v[:,165:] = np.ones((T_steps, NX-165)) * bc_v_ref

# %%
bc_h = mld[0].data*np.ones((T_steps,NX)) #np.tile(mld[0], (T_steps,1))

ampl_h = 4.0
bc_h[:,165:] = bc_h[:,165:] + ampl_h*np.ones_like(bc_h[:,165:])*np.sin(2*np.pi*ts_steps/freq)[:,np.newaxis]

bc_hv = bc_h*bc_v

bc_h = bc_h - H

# %%
south = Common.SingleBoundaryConditionData(h=bc_h.astype(np.float32), hu=np.zeros((T_steps, NX), dtype=np.float32), hv=bc_hv.astype(np.float32))
north = Common.SingleBoundaryConditionData(h=np.zeros((T_steps, NX), dtype=np.float32), hu=np.zeros((T_steps, NX), dtype=np.float32), hv=np.zeros((T_steps, NX), dtype=np.float32))
east  = Common.SingleBoundaryConditionData(h=np.zeros((T_steps, NY), dtype=np.float32), hu=np.zeros((T_steps, NY), dtype=np.float32), hv=np.zeros((T_steps, NY), dtype=np.float32))
west  = Common.SingleBoundaryConditionData(h=np.zeros((T_steps, NY), dtype=np.float32), hu=np.zeros((T_steps, NY), dtype=np.float32), hv=np.zeros((T_steps, NY), dtype=np.float32))

data_args["boundary_conditions_data"] = Common.BoundaryConditionsData(ts, north=north, south=south, east=east, west=west)

# %% [markdown]
# Run simulation

# %%
from gpuocean.SWEsimulators import CDKLM16
sim = CDKLM16.CDKLM16(gpu_ctx, dt=0.0,  **NetCDFInitialization.removeMetadata(data_args), write_netcdf=False)

# %%
bc_tic = time.time()
sim.step(T)
bc_toc = time.time()

print("BC wall time: ", (bc_toc-bc_tic)/48)

# %% [markdown]
# ### Barotropic Partner Sim

# %%
bt_data_args = copy.deepcopy(data_args)

# %%
H_m_data = nc.variables['h'][y0-1:y1+1, x0-1:x1+1]
H_m_mask = (H_m_data == 0.0)
H_m = np.ma.array(H_m_data, mask=H_m_mask)
H_i = OceanographicUtilities.midpointsToIntersections(H_m, land_value=0.0, iterations=10)[0]

bt_data_args["H"] = np.ma.array(H_i, mask=H_i.mask.copy(), dtype=np.float32)

# %%
bt_data_args["g"] = 9.81

# %%
bt_data_args["eta0"] = np.ma.array(np.zeros_like(data_args["eta0"]), mask=copy.copy(data_args["eta0"].mask))

# %%
ampl_v = 0.1 #Input
freq = 12*3600 
bc_v_ref = ampl_v * np.sin(2*np.pi*ts_steps/freq)[:,np.newaxis] 

bc_v = np.zeros((T_steps, NX))
bc_v[:,165:] = np.ones((T_steps, NX-165)) * bc_v_ref

# %%
bc_h = H_m[1,1:-1].data*np.ones((T_steps,NX))

ampl_h = 0.15
bc_eta = np.zeros_like(bc_h)
bc_eta[:,165:] = ampl_h*np.ones_like(bc_h[:,165:])*np.sin(2*np.pi*ts_steps/freq)[:,np.newaxis]

bc_hv = bc_h*bc_v

# %%
south = Common.SingleBoundaryConditionData(h=bc_eta.astype(np.float32), hu=np.zeros((T_steps, NX), dtype=np.float32), hv=bc_hv.astype(np.float32))
north = Common.SingleBoundaryConditionData(h=np.zeros((T_steps, NX), dtype=np.float32), hu=np.zeros((T_steps, NX), dtype=np.float32), hv=np.zeros((T_steps, NX), dtype=np.float32))
east  = Common.SingleBoundaryConditionData(h=np.zeros((T_steps, NY), dtype=np.float32), hu=np.zeros((T_steps, NY), dtype=np.float32), hv=np.zeros((T_steps, NY), dtype=np.float32))
west  = Common.SingleBoundaryConditionData(h=np.zeros((T_steps, NY), dtype=np.float32), hu=np.zeros((T_steps, NY), dtype=np.float32), hv=np.zeros((T_steps, NY), dtype=np.float32))

bt_data_args["boundary_conditions_data"] = Common.BoundaryConditionsData(ts, north=north, south=south, east=east, west=west)

# %%
bt_sim = CDKLM16.CDKLM16(gpu_ctx, dt=0.0,  **NetCDFInitialization.removeMetadata(bt_data_args), write_netcdf=False)

# %%
bt_tic = time.time()
bt_sim.step(T)
bt_toc = time.time()

print("BT wall time: ", (bt_toc-bt_tic)/48)
