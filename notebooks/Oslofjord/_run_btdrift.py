# %% [markdown]
# ```
# This notebook sets up and runs a test case for analyzing Kelvin waves
# Copyright (C) 2018 - 2022 SINTEF Digital
# Copyright (C) 2018 - 2022 Norwegian Meteorological Institute
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

#Import packages we need
import numpy as np
from netCDF4 import Dataset
import datetime, copy
from IPython.display import display

#For plotting
import matplotlib
from matplotlib import pyplot as plt

plt.rcParams["image.origin"] = "lower"

# %%
from gpuocean.utils import IPythonMagic, Common, NetCDFInitialization, WindStress, OceanographicUtilities

from gpuocean.SWEsimulators import CDKLM16

# %%
barotropic_gpu_ctx = Common.CUDAContext()

# %% 
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

os.makedirs("figs/"+timestamp, exist_ok=True)
os.makedirs("pickles/"+timestamp, exist_ok=True)

# %% [markdown]
# Path to the test file

# %%
source_url = "/sintef/data/OsloFjord/test_polstere_1h_0007.nc"

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
x0, x1, y0, y1 = 5, dimX-5, 175, dimY-5

# %% [markdown]
# Simulation span: 6h!

# %%
t_start = 5*24
t_stop =  t_start + 6

T = (t_stop-t_start)*3600  #Input
timestep_indices = [list(np.arange(t_start, t_stop+1))]

# %% [markdown]
# Generating wind fields

# %%
from gpuocean.utils import WindStress

def generate_wind_field(wind_angle_deg, t_start_index, T):
    """Generating WindStress object with spatially constant, hourly varying wind 
    according to the standard shape in Oslofjord (see Oslofjord-ArtificialParameters.ipynb)
    
    wind_angle_deg - angle of wind field towards north in degree!
    t_start_index  - index for the time when generated wind field is supposed to start
    T              - time span for which the wind field is generated in seconds
    """

    wind_angle = np.deg2rad(wind_angle_deg)

    # Reference wind speed for 0 - 24h 
    wind_u_ref = np.zeros(24)
    wind_u_ref[:10] = -0.5*(np.sin(2*np.pi*(np.arange(0,10)-2.5)/10)+1)
    wind_u_ref[10:] = 5*(np.sin(2*np.pi*(np.arange(10,24)+0.5)/14)+1)

    # Wind fields for 0 - 24h
    wind_u = list(np.cos(wind_angle)*wind_u_ref[:,np.newaxis][:,np.newaxis].astype(np.float32))
    wind_v = list(np.sin(wind_angle)*wind_u_ref[:,np.newaxis][:,np.newaxis].astype(np.float32))

    # Shifting that reference hours align with t_0
    shift = datetime.datetime.utcfromtimestamp(nc["ocean_time"][t_start_index]).hour
    for shift_hour in range(shift):
        wind_u.append(wind_u.pop(0))
        wind_v.append(wind_v.pop(0))

    # Repeat for several days if necessary
    wind_u = wind_u * int(np.ceil((T/3600+1)/24))
    wind_v = wind_v * int(np.ceil((T/3600+1)/24))

    # Cut relevant time span
    wind_u = wind_u[0:int(T/3600+1)]
    wind_v = wind_v[0:int(T/3600 +1)]

    # Construct time array in simulator time (starting with 0)
    ts = np.arange(T+1, step=3600)

    # Init WindStress object
    wind = WindStress.WindStress(t=ts, wind_u=wind_v, wind_v=wind_u)
    wind.compute_wind_stress_from_wind()

    return wind 

# %% [markdown]
# ### Initial conditions

# %%
ref_barotropic_data_args = NetCDFInitialization.getInitialConditions(source_url, x0, x1, y0, y1, timestep_indices=timestep_indices, norkyst_data=False, land_value=0.0, download_data=False)

# %% [markdown]
# #### Barotropic Simulations

# %%
ref_barotropic_data_args["wind"] = generate_wind_field(45.0, t_start, T)

# %%
ref_barotropic_sim = CDKLM16.CDKLM16(barotropic_gpu_ctx, **NetCDFInitialization.removeMetadata(ref_barotropic_data_args), dt=0.0, write_netcdf=True)

# %%
subt = 3600
for runt in range(int(T/subt)):
    ref_barotropic_sim.step(subt)


# %% [markdown]
# ### Barotropic Drifters

# %%
barotropic_wind_angles_degs = np.arange(35, 56, 10)
barotropic_wind_samples = [None]*len(barotropic_wind_angles_degs)

for i in range(len(barotropic_wind_angles_degs)):
    barotropic_wind_samples[i] = generate_wind_field(barotropic_wind_angles_degs[i], t_start, T)

# %%
# Barotropic data args by starting from earlier timesteps and doing a sufficient spin-up
barotropic_data_args = []

spinup_t = 24

spinup_barotropic_data_args = NetCDFInitialization.getInitialConditions(source_url, x0, x1, y0, y1, timestep_indices=[np.arange(t_start-spinup_t, t_start+1)], norkyst_data=False, land_value=0.0, download_data=False)
spinup_barotropic_data_args.pop("wind")

for wind_direction in barotropic_wind_angles_degs:
    print(wind_direction)
    wind = WindStress.WindStress(t=[0], wind_u=[np.array([[np.cos(np.deg2rad(wind_direction))*10]]).astype(np.float32)], wind_v=[np.array([[np.sin(np.deg2rad(wind_direction))*10]]).astype(np.float32)])
    spinup_barotropic_sim = CDKLM16.CDKLM16(barotropic_gpu_ctx, **NetCDFInitialization.removeMetadata(spinup_barotropic_data_args), dt=0.0, wind=wind)
    spinup_barotropic_sim.step(spinup_t*3600)
    eta, hu, hv = spinup_barotropic_sim.download()

    spunup_barotropic_data_args = copy.copy(ref_barotropic_data_args)
    spunup_barotropic_data_args["eta0"] = eta
    spunup_barotropic_data_args["hu0"] = hu
    spunup_barotropic_data_args["hv0"] = hv
    spunup_barotropic_data_args["wind"] = generate_wind_field(wind_direction, t_start, T)
    barotropic_data_args.append(spunup_barotropic_data_args)

# %%
barotropic_sims = []

for i in range(len(barotropic_data_args)):
        barotropic_sims.append( CDKLM16.CDKLM16(barotropic_gpu_ctx, **NetCDFInitialization.removeMetadata(barotropic_data_args[i]), wind=barotropic_wind_samples[i],  dt=0.0))

# %%
windage_samples = np.arange(0.0, 0.051, 0.025)


# %%
import pandas as pd
ref_table = pd.DataFrame(columns=["drifter_id", "barotropic_id", "windage_id"]).set_index("drifter_id")

for bt in range(len(barotropic_sims)):
    for windage in range(len(windage_samples)):
        ref_table.loc[len(ref_table.index)] = [bt, windage]


# %% [markdown]
# Collecting drifter and observation objects

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
initx = [ 6000, 8000, 5000,  12000, 15000, 4000]
inity = [12000, 3000, 7500, 22000, 16000, 15000]

num_drifters = len(initx)

# %%
crossprod_trajectories = []
for cp in range(len(ref_table)):
    crossprod_trajectories.append(copy.deepcopy(trajectories))

# %%
crossprod_drifters = []
for cp in range(len(ref_table)): 
    drifters = GPUDrifterCollection.GPUDrifterCollection(barotropic_gpu_ctx, # OBS: This is used for wind drift! 
                                                    num_drifters,
                                                    boundaryConditions = ref_barotropic_sim.boundary_conditions,
                                                    domain_size_x = trajectories.domain_size_x,
                                                    domain_size_y = trajectories.domain_size_y,
                                                    gpu_stream = ref_barotropic_sim.gpu_stream,
                                                    wind = barotropic_wind_samples[ref_table.iloc[cp].barotropic_id], # using the barotropic wind
                                                    wind_drift_factor = windage_samples[ref_table.iloc[cp].wind_drift_factor_id])          

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
    drifter_ids = ref_table.index[ref_table["baroclinic_id"]==bt].tolist()
    while barotropic_sims[bt].t < T:
        barotropic_sims[bt].step(360)
        for d in drifter_ids:
            crossprod_trajectories[d].add_observation_from_drifters(crossprod_drifters[d], barotropic_sims[bt].t)

# %%
def plot_cp_trajectories(crossprod_trajectories, drifter_id):
    with plt.rc_context({'lines.color':'black', 
                            'text.color':'black', 
                            'axes.labelcolor':'black', 
                            'xtick.color':'black',
                            'ytick.color':'black'}):
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.tick_params(axis='both', which='major', labelsize=28)
        domain_extent = [0, ref_barotropic_sim.nx*ref_barotropic_sim.dx/1000, 0, ref_barotropic_sim.ny*ref_barotropic_sim.dy/1000]

        bg_cmap = copy.deepcopy(plt.cm.Blues)
        bg_cmap.set_bad("grey", alpha = 1.0)

        ax.imshow(ref_barotropic_sim[0].download()[0] > -10, interpolation="none", origin='lower', 
                        cmap=bg_cmap,  
                        extent=domain_extent)

        path = crossprod_trajectories[0].get_drifter_path(drifter_id, 0, T, in_km = True)[0]
        start_pos = path[0,:]
        circ_start = matplotlib.patches.Circle((start_pos[0], start_pos[1]), 
                                                0.01, color = 'black', fill=True, zorder=10)
        ax.add_patch(circ_start)

        for path in [t.get_drifter_path(drifter_id, 0, T, in_km = True)[0] for t in crossprod_trajectories]:
            
            ax.plot(path[:,0], path[:,1], color="C0", ls="-", zorder=5, alpha=0.1)
            
            end_pos = path[-1,:]
            ax.scatter(end_pos[0], end_pos[1], marker='x', color='black', s=100, linewidths=2, alpha=0.5)

        plt.tight_layout()

        plt.savefig("figs/"+timestamp+"/CPdrift"+str(drifter_id)+".pdf", bbox_inches="tight")

# %%
for cp in range(len(crossprod_trajectories)):
    crossprod_trajectories[cp].to_pickle("pickles/"+timestamp+"/cp_trajectory"+str(cp))


# %%
for drifter_id in range(num_drifters):
    plot_cp_trajectories(crossprod_trajectories, drifter_id)


