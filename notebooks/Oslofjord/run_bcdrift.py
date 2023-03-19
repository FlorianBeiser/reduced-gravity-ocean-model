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

# %%
from gpuocean.utils import Common, NetCDFInitialization, WindStress, OceanographicUtilities

from gpuocean.SWEsimulators import CDKLM16

# %%
baroclinic_gpu_ctx = Common.CUDAContext()

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
# #### Baroclinic Simulations

# %%
_, ref_baroclinic_data_args = NetCDFInitialization.getCombinedInitialConditions(source_url, x0, x1, y0, y1, 1024.0, timestep_indices=timestep_indices, norkyst_data=False, land_value=0.0, download_data=False)

# %%
ref_baroclinic_data_args["wind"] = generate_wind_field(45.0, t_start, T)
ref_baroclinic_data_args["wind_stress_factor"] = 0.3

# %%
ref_baroclinic_sim = CDKLM16.CDKLM16(baroclinic_gpu_ctx, **NetCDFInitialization.removeMetadata(ref_baroclinic_data_args), dt=0.0, write_netcdf=True)

# %%
subt = 3600
for runt in range(int(T/subt)):
    ref_baroclinic_sim.step(subt)

# %% [markdown]
# ## Baroclinic Drifters

# %%
import pandas as pd 

# %%
baroclinic_wind_angles_degs = np.arange(35, 56, 5)
baroclinic_wind_samples = [None]*len(baroclinic_wind_angles_degs)

for i in range(len(baroclinic_wind_angles_degs)):
    baroclinic_wind_samples[i] = generate_wind_field(baroclinic_wind_angles_degs[i], t_start, T)

# %%
# Mixed layer depth (MLD) 
# Can be explored coupled or decoupled with the reduced gravity constant
mld_dens_samples = np.arange(1022.5, 1024.6, 0.50)
mld_samples_data_args = [None]*len(mld_dens_samples)

for i in range(len(mld_dens_samples)):
    _, mld_samples_data_args[i] = NetCDFInitialization.getCombinedInitialConditions(source_url, x0, x1, y0, y1, 
                                                                                    mld_dens_samples[i], timestep_indices=timestep_indices, norkyst_data=False, land_value=0.0, download_data=False)
    print(mld_samples_data_args[i]["g"])


# %%
wind_stress_samples = np.arange(0.1, 0.51, 0.1)

# %%
friction_samples = np.arange(0, 0.0051, 0.001)

# %%
windage_samples = np.arange(0.0, 0.031, 0.005)

# %% 
file = open("figs/"+timestamp+"/log.txt", 'w')
file.write("CROSS PRODUCT SIMULATION\n")
file.write("\n")
file.write("Baroclinic simulations:\n")
file.write("MLD: " + ", ".join([str(v) for v in mld_dens_samples])+"\n")
file.write("friction: " + ", ".join([str(v) for v in friction_samples])+"\n")
file.write("wind stress: " + ", ".join([str(v) for v in wind_stress_samples])+"\n")
file.write("wind: " + ", ".join([str(v) for v in baroclinic_wind_angles_degs])+"\n")
file.write("\n")
file.write("Drifter advection:\n")
file.write("windage: " + ", ".join([str(v) for v in windage_samples])+"\n")
file.close()

# %% [markdown]
# #### Creating simulators

# %%
baroclinic_sims = []
bc_table = pd.DataFrame(columns=["baroclinic_id", "wind_angle_id", "wind_stress_factor_id", "friction_id", "mld_id"]).set_index("baroclinic_id")

for i_w in range(len(baroclinic_wind_samples)):
    for i_ws in range(len(wind_stress_samples)):
        for i_f in range(len(friction_samples)):
            for i_mld in range(len(mld_samples_data_args)):
                baroclinic_data_args = copy.copy(mld_samples_data_args[i_mld])
                baroclinic_data_args["wind"] = baroclinic_wind_samples[i_w]
                baroclinic_data_args["wind_stress_factor"] = wind_stress_samples[i_ws]
                baroclinic_data_args["r"] = friction_samples[i_f]

                baroclinic_sims.append( CDKLM16.CDKLM16(baroclinic_gpu_ctx, **NetCDFInitialization.removeMetadata(baroclinic_data_args),  dt=0.0))
                
                bc_table.loc[len(bc_table.index)] = [i_w, i_ws, i_f, i_mld]

# %%
bc_table

# %%
ref_table = pd.DataFrame(columns=["drifter_id", "baroclinic_id", "windage_id"]).set_index("drifter_id")


for bc in range(len(baroclinic_sims)):
    for windage in range(len(windage_samples)):
        ref_table.loc[len(ref_table.index)] = [bc, windage]

ref_table

# %% [markdown]
# Collecting drifter and observation objects

# %%
from gpuocean.utils import Observation
from gpuocean.drifters import GPUDrifterCollection
from gpuocean.dataassimilation import DataAssimilationUtils as dautils

# %%
observation_type = dautils.ObservationType.UnderlyingFlow 
    
observation_args = {'observation_type': observation_type,
                'nx': ref_baroclinic_sim.nx, 'ny': ref_baroclinic_sim.ny,
                'domain_size_x': ref_baroclinic_sim.nx*ref_baroclinic_sim.dx,
                'domain_size_y': ref_baroclinic_sim.ny*ref_baroclinic_sim.dy,
                'land_mask': ref_baroclinic_sim.getLandMask()
                }

trajectories = Observation.Observation(**observation_args)


# %%
crossprod_trajectories = []
for cp in range(len(ref_table)):
    crossprod_trajectories.append(copy.deepcopy(trajectories))

# %%
initx = [ 6000, 8000, 5000,  12000, 15000, 4000]
inity = [12000, 3000, 7500, 22000, 16000, 15000]

num_drifters = len(initx)

# %%
crossprod_drifters = []
for cp in range(len(ref_table)): 
    drifters = GPUDrifterCollection.GPUDrifterCollection(baroclinic_gpu_ctx, # OBS: This is used for wind drift! 
                                                    num_drifters,
                                                    boundaryConditions = ref_baroclinic_sim.boundary_conditions,
                                                    domain_size_x = trajectories.domain_size_x,
                                                    domain_size_y = trajectories.domain_size_y,
                                                    gpu_stream = ref_baroclinic_sim.gpu_stream, # Get re-set in attachment to simulation!
                                                    wind = baroclinic_wind_samples[bc_table.iloc[ref_table.iloc[cp].baroclinic_id].wind_angle_id], #wind = ref_baroclinic_data_args["wind"],
                                                    wind_drift_factor = windage_samples[ref_table.iloc[cp].windage_id]
                                                    )           

    drifter_pos_init = np.array([initx, inity]).T
    drifters.setDrifterPositions(drifter_pos_init)
    crossprod_drifters.append(drifters)

# %% [markdown]
# Attach CPdrifters

# %%
from itertools import compress

# %%
for bc in range(len(baroclinic_sims)):
    baroclinic_sims[bc].attachCrossProductDrifters( 
        list(compress(crossprod_drifters, ref_table["baroclinic_id"] == bc)), 
        [None]*len(windage_samples) )

# %%
for d in range(len(crossprod_drifters)):
    crossprod_trajectories[d].add_observation_from_drifters(crossprod_drifters[d], 0.0)

# %% [markdown]
# Run the drift experiment

# %%
for bc in range(len(baroclinic_sims)):
    drifter_ids = ref_table.index[ref_table["baroclinic_id"]==bc].tolist()
    while baroclinic_sims[bc].t < T:
        baroclinic_sims[bc].step(360)
        for d in drifter_ids:
            crossprod_trajectories[d].add_observation_from_drifters(crossprod_drifters[d], baroclinic_sims[bc].t)

# %%
def plot_cp_trajectories(crossprod_trajectories, drifter_id):
    with plt.rc_context({'lines.color':'black', 
                            'text.color':'black', 
                            'axes.labelcolor':'black', 
                            'xtick.color':'black',
                            'ytick.color':'black'}):
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.tick_params(axis='both', which='major', labelsize=28)
        domain_extent = [0, ref_baroclinic_sim.nx*ref_baroclinic_sim.dx/1000, 0, ref_baroclinic_sim.ny*ref_baroclinic_sim.dy/1000]

        bg_cmap = copy.deepcopy(plt.cm.Blues)
        bg_cmap.set_bad("grey", alpha = 1.0)

        ax.imshow(baroclinic_sims[0].download()[0] > -10, interpolation="none", origin='lower', 
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
