# %% [markdown]
# ```
# This notebook runs barotropic drifters at the North Cape
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
# # Barents Sea

# %%
import os
import sys

#Import packages we need
import numpy as np
from netCDF4 import Dataset
import datetime, copy
from IPython.display import display

#For plotting
import matplotlib
from matplotlib import pyplot as plt

# %%
from gpuocean.utils import Common, NetCDFInitialization, WindStress, OceanographicUtilities

from gpuocean.SWEsimulators import CDKLM16

# %%
barotropic_gpu_ctx = Common.CUDAContext()

# %% 
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

os.makedirs("barents_figs/"+timestamp, exist_ok=True)
os.makedirs("barents_pickles/"+timestamp, exist_ok=True)

# %% [markdown]
# Path to the test file

# %%
source_url = "/sintef/data/NorFjords160/A12/norfjords_160m_his.nc4_2020020101-2020020200"

# %% [markdown]
# ## Inspecting file structure and content

# %%
from netCDF4 import Dataset
nc = Dataset(source_url)

# %%
import xarray as xr
ds = xr.open_dataset(source_url)
ds

# %% [markdown]
# ## Generating GPUOcean Simulation from Input

# %% [markdown]
# General parameters

# %%
dimY, dimX = ds.h.data.shape
x0, x1, y0, y1 = 1100, dimX-350, 315, dimY-585

# %% [markdown]
# Simulation span: 6h!

# %%
t_start = 9
t_stop =  t_start + 6

T = (t_stop-t_start)*3600  #Input
timestep_indices = [list(np.arange(t_start, t_stop+1))]

# %% [markdown]
# Generating wind fields

# %%
from gpuocean.utils import WindStress

def rotate_wind_field(wind, angle, plot=False):
    radians = (angle/360)*2*np.pi
    wind_u = wind.wind_u.copy()
    wind_v = wind.wind_v.copy()
    t = wind.t.copy()
    #print(t)

    c = np.cos(radians)
    s = np.sin(radians)
    wind_u_new = wind_u * c - wind_v * s
    wind_v_new = wind_u * s + wind_v * c

    if plot:
        x0, x1 = 200, 220
        y0, y1 = 200, 220

        fig = plt.figure()
        plt.quiver(wind_u[3, y0:y1, x0:x1], wind_v[3, y0:y1, x0:x1])

        fig = plt.figure()
        plt.quiver(wind_u_new[3, y0:y1, x0:x1], wind_v_new[3, y0:y1, x0:x1])
    return WindStress.WindStress(t=t, wind_u=wind_u_new, wind_v=wind_v_new)


# %% [markdown]
# ### Initial conditions

# %%
ref_barotropic_data_args = NetCDFInitialization.getInitialConditions(source_url, x0, x1, y0, y1, timestep_indices=timestep_indices, norkyst_data=False, land_value=0.0, download_data=False)

# %%
ref_barotropic_sim = CDKLM16.CDKLM16(barotropic_gpu_ctx, **NetCDFInitialization.removeMetadata(ref_barotropic_data_args), dt=0.0, write_netcdf=True)

# %%
subt = 3600
for runt in range(int(T/subt)):
    ref_barotropic_sim.step(subt)


# %% [markdown]
# ### Barotropic Drifters

# %% [markdown]
# Perturbations

# %%
ref_wind = ref_barotropic_data_args["wind"]

barotropic_wind_directions = np.array([-10,-5,0,5,10]) # np.random.normal(0, 15, 5) #np.arange(-20, 21, 5)
barotropic_wind_samples = [None]*len(barotropic_wind_directions)
print(barotropic_wind_directions)
for i in range(len(barotropic_wind_directions)):
    barotropic_wind_samples[i] = rotate_wind_field(ref_wind, barotropic_wind_directions[i])

# %%
max_shift_t = 9
shift_t_step = 3

time_shifted_data_args = []

for shift_t in range(-max_shift_t,max_shift_t,shift_t_step):

    shifted_timestep_indices = [np.arange(t_start+shift_t, t_stop+shift_t+1)]
    time_shifted_data_args_i = NetCDFInitialization.getInitialConditions(source_url, x0, x1, y0, y1, timestep_indices=shifted_timestep_indices, norkyst_data=False, land_value=5.0, download_data=False)

    time_shifted_data_args.append(time_shifted_data_args_i)

# %% [markdown]
# Creating simulators

# %%
import pandas as pd 

barotropic_sims = []
bt_table = pd.DataFrame(columns=["barotropic_id", "wind_rotation_id", "time_shift_id"]).set_index("barotropic_id")

for i_w in range(len(barotropic_wind_samples)):
    for i_t in range(len(time_shifted_data_args)):
        data_args = copy.copy(time_shifted_data_args[i_t])
        data_args["wind"] = barotropic_wind_samples[i_w]
        barotropic_sims.append( CDKLM16.CDKLM16(barotropic_gpu_ctx, **NetCDFInitialization.removeMetadata(data_args), dt=0.0) )

        bt_table.loc[len(bt_table)] = [i_w, i_t]

# %%
bt_table.head()

# %% [markdown]
# Creating drifters

# %%
windage_samples = np.array([0.03275772649313289, 0.036538865092169207, 0.014530555906561415, 0.031149217005290726, 0.0019454308392691487, 0.03615934828448532, 0.044317464017243464, 0.04610899216621601, 0.023221881093816385, 0.01649549488869334]) #np.maximum(0, np.random.normal(0.03, 0.015, 10)) #np.arange(0.0, 0.051, 0.005)

# %%
file = open("barents_figs/"+timestamp+"/log.txt", 'w')
file.write("Barotropic simulations:\n")
file.write("init (time shifted): " + ", ".join([str(v) for v in np.arange(-max_shift_t,max_shift_t,shift_t_step)])+"\n")
file.write("wind: " + ", ".join([str(v) for v in barotropic_wind_directions])+"\n")
file.write("\n")
file.write("Drifter advection:\n")
file.write("windage: " + ", ".join([str(v) for v in windage_samples])+"\n")
file.close()

# %%
import pandas as pd
ref_table = pd.DataFrame(columns=["drifter_id", "barotropic_id", "windage_id"]).set_index("drifter_id")

for bt in range(len(barotropic_sims)):
    for windage in range(len(windage_samples)):
        ref_table.loc[len(ref_table.index)] = [bt, windage]

# %%
ref_table.head()

# %%
from gpuocean.utils import Observation
from gpuocean.drifters import GPUDrifterCollection
from gpuocean.dataassimilation import DataAssimilationUtils as dautils

# %%
observation_type = dautils.ObservationType.UnderlyingFlow 
    
observation_args = {'observation_type': observation_type,
                'nx': ref_barotropic_sim.nx, 'ny': ref_barotropic_sim.ny,
                'domain_size_x': ref_barotropic_sim.nx*ref_barotropic_sim.dx,
                'domain_size_y': ref_barotropic_sim.ny*ref_barotropic_sim.dy,
                'land_mask': ref_barotropic_sim.getLandMask()
                }

trajectories = Observation.Observation(**observation_args)

# %%
initx = [47500, 60000, 72500, 80000, 72500, 50000, 30000]
inity = [42500, 30500, 37500, 27500, 15000,  7500,  5000]

num_drifters = len(initx)

# %%
crossprod_trajectories = []
for cp in range(len(ref_table)):
    crossprod_trajectories.append(copy.deepcopy(trajectories))

# %%
barotropic_wind_samples[bt_table.iloc[ref_table.iloc[2].barotropic_id].wind_rotation_id]

# %%
crossprod_drifters = []
for cp in range(len(ref_table)):
    drifters = GPUDrifterCollection.GPUDrifterCollection(barotropic_gpu_ctx, # OBS: This is used for wind drift! 
                                                    num_drifters,
                                                    boundaryConditions = ref_barotropic_sim.boundary_conditions,
                                                    domain_size_x = trajectories.domain_size_x,
                                                    domain_size_y = trajectories.domain_size_y,
                                                    gpu_stream = ref_barotropic_sim.gpu_stream,
                                                    wind = barotropic_wind_samples[bt_table.iloc[ref_table.iloc[cp].barotropic_id].wind_rotation_id],
                                                    wind_drift_factor = windage_samples[ref_table.iloc[cp].windage_id])          

    drifter_pos_init = np.array([initx, inity]).T
    drifters.setDrifterPositions(drifter_pos_init)
    crossprod_drifters.append(drifters)

# %%
from itertools import compress

for bt in range(len(barotropic_sims)):
    barotropic_sims[bt].attachCrossProductDrifters( 
        list(compress(crossprod_drifters, ref_table["barotropic_id"] == bt)), 
        int(np.sum(ref_table["barotropic_id"] == bt))* [None] )

# %%
for d in range(len(crossprod_drifters)):
    crossprod_trajectories[d].add_observation_from_drifters(crossprod_drifters[d], 0.0)

# %%
for bt in range(len(barotropic_sims)):
    print(bt, "of ", len(barotropic_sims))
    drifter_ids = ref_table.index[ref_table["barotropic_id"]==bt].tolist()
    while barotropic_sims[bt].t < T:
        barotropic_sims[bt].step(360)
        for d in drifter_ids:
            crossprod_trajectories[d].add_observation_from_drifters(crossprod_drifters[d], barotropic_sims[bt].t)

# %%
os.makedirs("barents_pickles/"+timestamp, exist_ok=True)
for cp in range(len(crossprod_trajectories)):
    crossprod_trajectories[cp].to_pickle("barents_pickles/"+timestamp+"/bt_trajectory"+str(cp))

