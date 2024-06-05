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
# # Boknafjorden

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
baroclinic_gpu_ctx = Common.CUDAContext()

# %%
drifter_gpu_ctx = Common.CUDAContext()

# %% 
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

os.makedirs("bokna_figs/"+timestamp, exist_ok=True)
os.makedirs("bokna_pickles/"+timestamp, exist_ok=True)
# %% [markdown]
# Path to the test file

# %%
source_url = "/sintef/data/NorFjords160/A03/norfjords_160m_his.nc4_2019110101-2019110200"

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
x0, x1, y0, y1 = 685, dimX-360, 335, dimY-330

# %% [markdown]
# Simulation span: 6h!

# %%
t_start = 10
t_stop =  t_start + 6

T = (t_stop-t_start)*3600  #Input
timestep_indices = [list(np.arange(t_start, t_stop+1))]

# %% [markdown]
# ### Initial conditions

# %%
mld_density = 1023
sponge_cells = {'north': 5, 'south': 5, 'east': 5, 'west': 5}
ref_barotropic_data_args, ref_baroclinic_data_args = NetCDFInitialization.getCombinedInitialConditions(source_url, x0, x1, y0, y1, mld_density, timestep_indices=timestep_indices, norkyst_data=False, land_value=5.0, download_data=False, sponge_cells=sponge_cells)

# %% [markdown]
# #### Barotropic Simulations

# %%
ref_barotropic_sim = CDKLM16.CDKLM16(barotropic_gpu_ctx, **NetCDFInitialization.removeMetadata(ref_barotropic_data_args), dt=0.0, write_netcdf=True)

# %%
subt = 3600
for runt in range(int(T/subt)):
    ref_barotropic_sim.step(subt)

# %% [markdown]
# #### Baroclinic Simulations

# %%
ref_baroclinic_data_args["wind_stress_factor"] = 0.3

# %%
ref_baroclinic_sim = CDKLM16.CDKLM16(baroclinic_gpu_ctx, **NetCDFInitialization.removeMetadata(ref_baroclinic_data_args), dt=0.0, write_netcdf=True)

# %%
subt = 3600
for runt in range(int(T/subt)):
    ref_baroclinic_sim.step(subt)

# %% [markdown]
# #### Combined deterministic simulation with drifters
# 
# Just for reference

# %%
from gpuocean.SWEsimulators import CombinedCDKLM16

ref_barotropic_sim = CDKLM16.CDKLM16(barotropic_gpu_ctx, **NetCDFInitialization.removeMetadata(ref_barotropic_data_args), dt=0.0, write_netcdf=False)
ref_baroclinic_sim = CDKLM16.CDKLM16(baroclinic_gpu_ctx, **NetCDFInitialization.removeMetadata(ref_baroclinic_data_args), dt=0.0, write_netcdf=False)

ref_sims = CombinedCDKLM16.CombinedCDKLM16(barotropic_sim=ref_barotropic_sim, baroclinic_sim=ref_baroclinic_sim)

# %%
from gpuocean.utils import Observation
from gpuocean.drifters import GPUDrifterCollection
from gpuocean.dataassimilation import DataAssimilationUtils as dautils

# %%
observation_type = dautils.ObservationType.UnderlyingFlow 
    
observation_args = {'observation_type': observation_type,
                'nx': ref_sims.nx, 'ny': ref_sims.ny,
                'domain_size_x': ref_sims.nx*ref_sims.dx,
                'domain_size_y': ref_sims.ny*ref_sims.dy,
                'land_mask': ref_sims.getLandMask()
                }

trajectories = Observation.Observation(**observation_args)

# %%
initx = np.array([ 2500,  6000, 11000,  7500])
inity = np.array([10000,  7500,  7000, 12000])

num_drifters = len(initx)

drifters = GPUDrifterCollection.GPUDrifterCollection(drifter_gpu_ctx, num_drifters,
                                                    boundaryConditions = ref_sims.boundary_conditions,
                                                    domain_size_x = trajectories.domain_size_x,
                                                    domain_size_y = trajectories.domain_size_y,
                                                    gpu_stream = ref_sims.gpu_stream,
                                                    wind=ref_barotropic_data_args["wind"],
                                                    wind_drift_factor=0.02/2 # halfend since drift kernel called twice
                                                    )

drifter_pos_init = np.array([initx, inity]).T
drifters.setDrifterPositions(drifter_pos_init)

# %%
ref_sims.attachDrifters(drifters)
trajectories.add_observation_from_sim(ref_sims)

# %%
ref_sims.combinedStep(T, trajectory_dt=60, trajectories=trajectories)

# %% [markdown]
# ## Cross Pert Drifters

# %% [markdown]
# The test scenario should be
# - X barotropic simulations (different `wind`)
# - Y wind drift factors (connected to barotropic sims, since wind response of baroclinic model is already well covered)
# - Z baroclinic simulations (2 different `wind`, with 2 different `wind_stress_factors` and 3 different `g`)

# %%
import pandas as pd 

# %%
# Wind 
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
       

# %%
# wind_directions = np.random.normal(0, 10, 5) #np.arange(-15, 16, 5)
# print(wind_directions)

# wind_samples = [None]*len(wind_directions)
# for i in range(len(wind_directions)):
#     wind_samples[i] = rotate_wind_field(ref_baroclinic_data_args["wind"], wind_directions[i])

# %%
barotropic_wind_directions = np.array([-10,-5,0,5,10])#np.array([15.20564991, -4.48302138,  1.57755535,  4.63672119, -3.08141148]) #TEMP np.random.normal(0, 10, 5) #np.arange(-15, 16, 5)
print(barotropic_wind_directions)

barotropic_wind_samples = [None]*len(barotropic_wind_directions)
for i in range(len(barotropic_wind_directions)):
    barotropic_wind_samples[i] = rotate_wind_field(ref_baroclinic_data_args["wind"], barotropic_wind_directions[i])

# %%
baroclinic_wind_directions = np.array([-10,-5,0,5,10])#np.array([ -9.48045535,  14.23118069,  -0.9100276 ,   6.52690427, -10.57726405]) #TEMP np.random.normal(0, 10, 5) #np.arange(-15, 16, 15)
print(baroclinic_wind_directions)

baroclinic_wind_samples = [None]*len(baroclinic_wind_directions)
for i in range(len(baroclinic_wind_directions)):
    baroclinic_wind_samples[i] = rotate_wind_field(ref_baroclinic_data_args["wind"], baroclinic_wind_directions[i])

# %% [markdown]
# Collect perturbations

# %%
barotropic_shift_times = np.arange(-6,7,6)
barotropic_time_shift_args = [None]*len(barotropic_shift_times)

for i in range(len(barotropic_shift_times)):
    timestep_indices = [list(np.arange(t_start+barotropic_shift_times[i], t_stop+1+barotropic_shift_times[i]))]
    barotropic_time_shift_args[i] = NetCDFInitialization.getInitialConditions(source_url, x0, x1, y0, y1, 
                                                                            timestep_indices=timestep_indices, 
                                                                            norkyst_data=False, land_value=5.0, download_data=False, 
                                                                            sponge_cells=sponge_cells)

# %%
# Mixed layer depth (MLD) 
# Can be explored coupled or decoupled with the reduced gravity constant
mld_dens_samples = np.arange(1022.75, 1023.51, 0.1)
mld_samples_data_args = [None]*len(mld_dens_samples)

for i in range(len(mld_dens_samples)):
    _, mld_samples_data_args[i] = NetCDFInitialization.getCombinedInitialConditions(source_url, x0, x1, y0, y1, 
                                                                                    mld_dens_samples[i], timestep_indices=timestep_indices, norkyst_data=False, land_value=5.0, download_data=False)
    print(mld_samples_data_args[i]["g"])




# %%
# windage_samples = np.maximum(0, np.random.normal(0.03, 0.015, 1)) #TEMP np.maximum(0, np.random.normal(0.03, 0.015, 10)) #np.arange(0.0, 0.051, 0.005)


# %% [markdown]
# Generate all contextes

# %%
bc_gpu_ctxs = []
for i in range(len(baroclinic_wind_samples)):
    bc_gpu_ctxs.append( Common.CUDAContext() )

bt_gpu_ctxs = []
for i in range(len(barotropic_wind_samples)):
    bt_gpu_ctxs.append( Common.CUDAContext() )

# %% [markdown]
# #### Creating simulators

# %%
barotropic_sims = []
bt_table = pd.DataFrame(columns=["barotropic_id", "wind_rotation_id", "time_shift_id"]).set_index("barotropic_id")

for i_w in range(len(barotropic_wind_samples)):
    for i_t in range(len(barotropic_time_shift_args)):
        data_args = copy.copy(barotropic_time_shift_args[i_t])
        data_args["wind"] = barotropic_wind_samples[i_w]
        barotropic_sims.append( CDKLM16.CDKLM16(bt_gpu_ctxs[i_w], **NetCDFInitialization.removeMetadata(data_args), dt=0.0) )

        bt_table.loc[len(bt_table)] = [i_w, i_t]

# %% 
baroclinic_sims_per_wind = 36

# %%
baroclinic_sims = []
bc_table = pd.DataFrame(columns=["baroclinic_id", "wind_rotation_id", "wind_stress_factor", "friction", "mld"]).set_index("baroclinic_id")

for i_w in range(len(baroclinic_wind_samples)):
    mld_idxs = np.random.randint(low=0, high=len(mld_dens_samples), size=baroclinic_sims_per_wind)
    wind_stress_samples = np.minimum(np.maximum(0, np.random.normal(0.35,0.25, size=baroclinic_sims_per_wind)), 0.7)
    friction_samples = np.maximum(0, np.random.normal(0.0025,0.001, size=baroclinic_sims_per_wind))
    for i_other in range(baroclinic_sims_per_wind):
        baroclinic_data_args = copy.copy(mld_samples_data_args[mld_idxs[i_other]])
        baroclinic_data_args["wind"] = baroclinic_wind_samples[i_w]
        baroclinic_data_args["wind_stress_factor"] = wind_stress_samples[i_other]
        baroclinic_data_args["r"] = friction_samples[i_other]

        baroclinic_sims.append( CDKLM16.CDKLM16(bc_gpu_ctxs[i_w], **NetCDFInitialization.removeMetadata(baroclinic_data_args),  dt=0.0))
        
        bc_table.loc[len(bc_table.index)] = [i_w, wind_stress_samples[i_other], friction_samples[i_other], mld_idxs[i_other]]

bc_table["wind_rotation_id"] = bc_table["wind_rotation_id"].astype(int)

# %% [markdown]
# #### Cross Product Table 

#%%
windage_samples_per_sim = 10

# %%
ref_table = pd.DataFrame(columns=["drifter_id", "barotropic_id", "baroclinic_id", "windage"]).set_index("drifter_id")

# %%
for bt in range(len(barotropic_sims)):
    for bc in range(len(baroclinic_sims)):
        windage_samples = np.maximum(0, np.random.normal(0.03, 0.015, size=windage_samples_per_sim))
        for i_windage in range(windage_samples_per_sim):
            ref_table.loc[len(ref_table.index)] = [bt, bc, windage_samples[i_windage]]

ref_table["baroclinic_id"] = ref_table["baroclinic_id"].astype(int)
ref_table["barotropic_id"] = ref_table["barotropic_id"].astype(int)

# %% 
file = open("bokna_figs/"+timestamp+"/log.txt", 'w')
file.write("CROSS PRODUCT SIMULATION\n")
file.write("\n")
file.write("Barotropic simulations:\n")
file.write("wind: " + ", ".join([str(v) for v in barotropic_wind_directions])+"\n")
file.write("shifted init: " + ", ".join([str(v) for v in barotropic_shift_times]) + "\n")
file.write("\n")
file.write("Baroclinic simulations:\n")
file.write("wind: " + ", ".join([str(v) for v in baroclinic_wind_directions])+"\n")
file.write("simulation samples per wind: " + str(baroclinic_sims_per_wind)+"\n")

file.write("\n")
file.write("Drifter advection:\n")
file.write("windage samples per simulation: " + str(windage_samples_per_sim)+"\n")
file.close()


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
crossprod_trajectories = []
for cp in range(len(ref_table)):
    crossprod_trajectories.append(copy.deepcopy(trajectories))

# %%
crossprod_drifters = []
for cp in range(len(ref_table)): 
    drifters = GPUDrifterCollection.GPUDrifterCollection(bt_gpu_ctxs[bt_table.wind_rotation_id.iloc[ref_table.barotropic_id.iloc[cp]]], # OBS: This is used for wind drift! 
                                                    num_drifters,
                                                    boundaryConditions = ref_barotropic_sim.boundary_conditions,
                                                    domain_size_x = trajectories.domain_size_x,
                                                    domain_size_y = trajectories.domain_size_y,
                                                    gpu_stream = barotropic_sims[ref_table.barotropic_id.iloc[cp]].gpu_stream, # OBS!
                                                    wind = barotropic_wind_samples[bt_table.wind_rotation_id.iloc[ref_table.barotropic_id.iloc[cp]]],
                                                    wind_drift_factor = ref_table.iloc[cp].windage/2 # drift is called twice per step, but we only want one wind contribution
                                                    )           

    drifter_pos_init = np.array([initx, inity]).T
    drifters.setDrifterPositions(drifter_pos_init)
    crossprod_drifters.append(drifters)

# %% [markdown]
# Attach CPdrifters

# %%
from itertools import compress

# %%
for bt in range(len(barotropic_sims)):
    barotropic_sims[bt].attachCrossProductDrifters( 
        list(compress(crossprod_drifters, ref_table["barotropic_id"] == bt)), 
        [baroclinic_sims[i] for i in list(ref_table[ref_table["barotropic_id"] == bt].baroclinic_id)] )

# %% [markdown]
# Let's fan out the drifters!

# %%
write_dt = 60.0 
write_t  = 0.0

# %%
bc_dt = min([bc.dt for bc in baroclinic_sims])

# %%
for bc in baroclinic_sims:
    bc.step(bc_dt)

# %%
for bt in barotropic_sims:
    bt.step(bc_dt)

# %%
for d in range(len(crossprod_drifters)):
    crossprod_trajectories[d].add_observation_from_drifters(crossprod_drifters[d], bt.t)

# %%
time_file = open("bokna_figs/"+timestamp+"/time_log.txt", 'w')

while bt.t < T:
    print(bt.t)
    time_file.write(str(bt.t)+"\n")

    bc_dt = min([bc.dt for bc in baroclinic_sims])
    if bt.t % 3600 != 0:
        bc_dt = min(bc_dt, np.ceil(bt.t/3600)*3600 - bt.t)

    for bc in baroclinic_sims:
        bc.step(bc_dt)

    for bt in barotropic_sims:
        bt.step(bc_dt)

    if bt.t > write_t + write_dt:
        for d in range(len(crossprod_drifters)):
            crossprod_trajectories[d].add_observation_from_drifters(crossprod_drifters[d], bt.t)
        write_t = bt.t

# %%
for cp in range(len(crossprod_trajectories)):
    time_file.write(str(cp)+"\n")
    crossprod_trajectories[cp].to_pickle("bokna_pickles/"+timestamp+"/cp_trajectory"+str(cp))
