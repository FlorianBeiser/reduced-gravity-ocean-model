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

plt.rcParams["lines.color"] = "w"
plt.rcParams["text.color"] = "w"
plt.rcParams["axes.labelcolor"] = "w"
plt.rcParams["xtick.color"] = "w"
plt.rcParams["ytick.color"] = "w"

plt.rcParams["image.origin"] = "lower"

from IPython.display import clear_output
from matplotlib import animation, rc
plt.rcParams["animation.html"] = "jshtml"
from mpl_toolkits.axes_grid1 import make_axes_locatable


from gpuocean.utils import NetCDFInitialization


def plotSolution(fig, 
                 eta, hu, hv, h, dx, dy, 
                 t, red_grav_mode=False,
                 comment = "Oslo",
                 h_min=-0.25, h_max=0.25, 
                 uv_min=-5, uv_max=5,
                 ax=None, sp=None, quiv=None, frequ=10):


    from datetime import timedelta
    fig.suptitle("Time = " + str(datetime.datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')) + " " + comment, 
                 fontsize=18,
                 horizontalalignment='left')
    
    ny, nx = eta.shape
    domain_extent = [0, nx*dx, 0, ny*dy]
    
    x_plots = 4
    y_plots = 1
   
    labels = ["eta", "hv", "hu"]
    if red_grav_mode:
        labels = ["MLD","hv","hu"]

    # Prepare quiver
    u = hu/(h+eta)
    v = hv/(h+eta)
    velocity = np.ma.sqrt(u*u + v*v)
    
    frequency_x = frequ
    frequency_y = frequ
    x = np.arange(0, velocity.shape[1], frequency_x)*dx
    y = np.arange(0, velocity.shape[0], frequency_y)*dy
    qu = u[::frequency_y, ::frequency_x]
    qv = v[::frequency_y, ::frequency_x]

    if red_grav_mode:
        eta = -(h+eta)
        h_min = -15
        h_max = 0

    if (ax is None):
        ax = [None]*x_plots*y_plots
        sp = [None]*x_plots*y_plots

        uv_cmap = copy.copy(plt.cm.coolwarm)
        uv_cmap.set_bad("grey", alpha = 1.0)
        
        h_cmap = copy.copy(plt.cm.coolwarm)
        h_cmap.set_bad("grey", alpha = 1.0)
        if red_grav_mode:
            h_cmap = copy.copy(plt.cm.Blues_r)
            h_cmap.set_bad("grey", alpha = 1.0)

        velo_cmap = copy.copy(plt.cm.Oranges)
        velo_cmap.set_bad("grey", alpha = 1.0)

        ax[0] = plt.subplot(y_plots, x_plots, 1)
        sp[0] = ax[0].imshow(eta, interpolation="none", origin='lower', 
                             cmap=h_cmap, 
                             vmin=h_min, vmax=h_max, 
                             extent=domain_extent)
        plt.axis('image')
        plt.title("$"+labels[0]+"$")
        divider0 = make_axes_locatable(ax[0])
        cax0 = divider0.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sp[0],cax=cax0)


        ax[1] = plt.subplot(y_plots, x_plots, 2)
        sp[1] = ax[1].imshow(hu, interpolation="none", origin='lower', 
                            cmap=uv_cmap, 
                            vmin=uv_min, vmax=uv_max, 
                            extent=domain_extent)
        plt.axis('image')
        plt.title("$"+labels[1]+"$")
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sp[1],cax=cax1)



        ax[2] = plt.subplot(y_plots, x_plots, 3)
        sp[2] = ax[2].imshow(hv, interpolation="none", origin='lower', 
                             cmap=uv_cmap, 
                             vmin=uv_min, vmax=uv_max, 
                             extent=domain_extent)
        plt.axis('image')
        plt.title("$"+labels[2]+"$")
        divider2 = make_axes_locatable(ax[2])
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sp[2],cax=cax2)

        ax[3] = plt.subplot(y_plots, x_plots, 4)
        sp[3] = ax[3].imshow(velocity, interpolation="none", origin='lower', 
                             cmap="Reds", 
                             vmin=0, vmax=1.0, 
                             extent=domain_extent)
        quiv = ax[3].quiver(x,y,qu,qv, scale=5)
        plt.axis('image')
        plt.title("velocity")
        divider2 = make_axes_locatable(ax[3])
        cax3 = divider2.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sp[3],cax=cax3)

        plt.tight_layout()
            
    else:        
        #Update plots
        fig.sca(ax[0])
        sp[0].set_data(eta)

        fig.sca(ax[1])
        sp[1].set_data(hu)
        
        fig.sca(ax[2])
        sp[2].set_data(hv)

        fig.sca(ax[3])
        sp[3].set_data(velocity)
        quiv.set_UVC(qu, qv)
    
    return ax, sp, quiv


def ncAnimation(source_url, nctype, t_range=[0, None], ROMS_upper_layer=None, ROMS_coord=[0,None,0,None], comment="",**kwargs):
    #Create figure and plot initial conditions
    fig = plt.figure(figsize=(20, 6))

    ncfile = Dataset(source_url)

    red_grav_mode = False

    if nctype == "ROMS":
        t_start = t_range[0]
        t_stop  = t_range[1]
        t = ncfile.variables['ocean_time'][t_start:t_stop]

        x0, x1 = ROMS_coord[0], ROMS_coord[1]
        y0, y1 = ROMS_coord[2], ROMS_coord[3]

        if x0 == 0:
            x0 = 1
        if x1 is None:
            x1 = ncfile["zeta"][0].shape[1] - 1
        if y0 == 0:
            y0 = 1
        if y1 is None:
            y1 = ncfile["zeta"][0].shape[0] -1 

        H_m = np.ma.array(ncfile["h"][y0:y1,x0:x1], mask=[1-ncfile["mask_rho"][y0:y1,x0:x1]])

        if not ROMS_upper_layer:
            eta = np.ma.array(ncfile["zeta"][t_start:t_stop,y0:y1,x0:x1], mask=len(t)*[1-ncfile["mask_rho"][y0:y1,x0:x1]])
            try:
                u = np.ma.array( 0.5*(ncfile["ubar"][t_start:t_stop,y0:y1,x0+1:x1+1]+ncfile["ubar"][t_start:t_stop,y0:y1,x0-1:x1-1]), mask=len(t)*[1-ncfile["mask_rho"][y0:y1,x0:x1]])
                v = np.ma.array( 0.5*(ncfile["vbar"][t_start:t_stop,y0+1:y1+1:,x0:x1]+ncfile["vbar"][t_start:t_stop,y0-1:y1-1,x0:x1]), mask=len(t)*[1-ncfile["mask_rho"][y0:y1,x0:x1]])

                hu = u*H_m
                hv = v*H_m
            except:
                u = 0.5*(ncfile["u"][t_start:t_stop,:,1:-1,1:]+ncfile["u"][t_start:t_stop,:,1:-1,:-1])
                v = 0.5*(ncfile["v"][t_start:t_stop,:,1:,1:-1]+ncfile["v"][t_start:t_stop,:,:-1,1:-1])
                
                integrator = NetCDFInitialization.MLD_integrator(source_url, H_m, x0=1, x1=-1, y0=1, y1=-1)
                hu = np.ma.array(np.sum(integrator * u, axis=1), mask=len(t)*[1-ncfile["mask_rho"][1:-1,1:-1]])
                hv = np.ma.array(np.sum(integrator * v, axis=1), mask=len(t)*[1-ncfile["mask_rho"][1:-1,1:-1]])
        else:
            H_m = 0.0
            eta = []
            hu  = []
            hv  = []

            for t_idx in range(t_start, t_stop):

                u = 0.5*(ncfile["u"][t_idx,:,y0:y1,x0:x1]+ncfile["u"][t_idx,:,y0:y1,x0+1:x1+1])
                v = 0.5*(ncfile["v"][t_idx,:,y0:y1,x0:x1]+ncfile["v"][t_idx,:,y0+1:y1+1,x0:x1])

                mld = NetCDFInitialization.MLD(source_url, 1024, min_mld=1.5, max_mld=40, x0=x0, x1=x1, y0=y0, y1=y1, t=t_idx)
                integrator = NetCDFInitialization.MLD_integrator(source_url, mld, x0=x0, x1=x1, y0=y0, y1=y1)

                eta.append( mld )
                
                hu.append( np.sum(integrator * u, axis=0) )
                hv.append( np.sum(integrator * v, axis=0) )

            eta = np.ma.array(eta, mask = len(t)*[1-ncfile["mask_rho"][y0:y1,x0:x1]])
            hu  = np.ma.array(hu, mask = len(t)*[1-ncfile["mask_rho"][y0:y1,x0:x1]])
            hv  = np.ma.array(hv, mask = len(t)*[1-ncfile["mask_rho"][y0:y1,x0:x1]])
            red_grav_mode = True


    elif nctype == "gpuocean":
        t = ncfile["time"][:]

        eta = ncfile["eta"][:]
        hu  = ncfile["hu"][:]
        hv  = ncfile["hv"][:]

        H_m = ncfile["Hm"][:]

    elif nctype == "gpuocean-reduced_grav":
        t = ncfile["time"][:]

        eta = ncfile["eta"][:]
        hu  = ncfile["hu"][:]
        hv  = ncfile["hv"][:]

        H_m = ncfile["Hm"][:]
        
        red_grav_mode = True

        

    movie_frames = len(t)

    dx = 50
    dy = 50
    
    ax, sp, quiv = plotSolution(fig, 
                            eta[0],
                            hu[0],
                            hv[0],
                            H_m,
                            dx, dy, 
                            t[0], 
                            red_grav_mode,
                            comment=comment,
                            **kwargs)


    #Helper function which simulates and plots the solution    
    def animate(i):
        t_now = t[0] + (i / (movie_frames-1)) * (t[-1] - t[0]) 

        k = np.searchsorted(t, t_now)
        if (k >= eta.shape[0]):
            k = eta.shape[0] - 1
        j = max(0, k-1)
        if (j == k):
            k += 1
        s = (t_now - t[j]) / (t[k] - t[j])

        plotSolution(fig, 
                        ((1-s)*eta[j] + s*eta[k]), 
                        ((1-s)*hu[j]  + s*hu[k]), 
                        ((1-s)*hv[j]  + s*hv[k]), 
                        H_m, 
                        dx, dy, 
                        t_now, 
                        red_grav_mode,
                        comment=comment,
                        **kwargs, ax=ax, sp=sp, quiv=quiv)

        clear_output(wait = True)
        #print(progress.getPrintString(i / (movie_frames-1)))

    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(movie_frames), interval=250)
    plt.close(fig)
    
    return anim



################################################################
################################################################
################################################################




def plotMLD(fig, 
                 mld, 
                 t,
                 comment = "Oslo",
                 ax=None, sp=None):


    from datetime import timedelta
    fig.suptitle("Time = " + str(datetime.datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')) + " " + comment, 
                 fontsize=18,
                 horizontalalignment='left')

    if (ax is None):

        cmap = plt.cm.Blues
        cmap.set_bad("grey", alpha = 1.0)

        ax = plt.subplot(1, 1, 1)
        sp = ax.imshow(mld, interpolation="none", origin='lower', 
                             cmap=cmap, aspect="auto",
                             vmin=0, vmax=10)
        plt.axis('image')
        plt.title("MLD")
        fig.colorbar(sp,ax=ax, label="[m]")
            
    else:        
        #Update plots
        fig.sca(ax)
        sp.set_data(mld)

    
    return ax, sp


def mldAnimation(source_url, t_range=[0, None], ROMS_coord=[0,None,0,None], comment="",**kwargs):
    #Create figure and plot initial conditions
    fig = plt.figure(figsize=(10, 6))

    x0, x1 = ROMS_coord[0], ROMS_coord[1]
    y0, y1 = ROMS_coord[2], ROMS_coord[3]

    mlds = []

    for t in range(t_range[0],t_range[1]):
        mlds.append(NetCDFInitialization.MLD(source_url, 1024, min_mld=1.5, max_mld=40, x0=x0, x1=x1, y0=y0, y1=y1, t=t))

    t_start = t_range[0]
    t_stop  = t_range[1]
    t = Dataset(source_url).variables['ocean_time'][t_start:t_stop]

    movie_frames = len(mlds)

    ax, sp = plotMLD(fig, 
                            mlds[0],
                            t[0], 
                            comment="MLD (FjordOS)",
                            **kwargs)


    #Helper function which simulates and plots the solution    
    def animate(i):
        t_now = t[0] + (i / (movie_frames-1)) * (t[-1] - t[0]) 

        k = np.searchsorted(t, t_now)
        if (k >= len(mlds)):
            k = len(mlds) - 1
        j = max(0, k-1)
        if (j == k):
            k += 1
        s = (t_now - t[j]) / (t[k] - t[j])

        plotMLD(fig, 
                        ((1-s)*mlds[j] + s*mlds[k]), 
                        t_now, 
                        comment=comment,
                        **kwargs, ax=ax, sp=sp)

        clear_output(wait = True)
        #print(progress.getPrintString(i / (movie_frames-1)))

    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(movie_frames), interval=250)
    plt.close(fig)
    
    return anim



################################################################
################################################################
################################################################



def plotDens(fig, 
                 dens, 
                 t,
                 comment = "Oslo",
                 ax=None, sp=None):


    from datetime import timedelta
    fig.suptitle("Time = " + str(datetime.datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')) + " " + comment, 
                 fontsize=18,
                 horizontalalignment='left')

    dens = dens.repeat(10,axis=1)

    if (ax is None):

        ax = plt.subplot(1, 1, 1)
        sp = ax.imshow(dens.T, interpolation="none", origin='upper', 
                             cmap="plasma", aspect="auto",
                             vmin=1020, vmax=1025)
        plt.axis('image')
        plt.title("Potential Density")
        ax.set_ylabel("Depth in dm")
        fig.colorbar(sp,ax=ax)
            
    else:        
        #Update plots
        fig.sca(ax)
        sp.set_data(dens.T)

    
    return ax, sp


def densAnimation(source_url, y_cut, t_range=[0, None], ROMS_coord=[0,None,0,None], comment="",**kwargs):
    #Create figure and plot initial conditions
    fig = plt.figure(figsize=(10, 6))

    x0, x1 = ROMS_coord[0], ROMS_coord[1]
    y0, y1 = ROMS_coord[2], ROMS_coord[3]

    s_pot_densities_shows = []

    for t in range(t_range[0],t_range[1]):

        s_pot_densities = NetCDFInitialization.potentialDensities(source_url, t=t, x0=x0, x1=x1, y0=y0, y1=y1)

        depth_show = 25
        s_pot_densities_show = np.ma.array(np.zeros((s_pot_densities.shape[1],depth_show)))
        s_pot_densities_show[:,0] =  s_pot_densities[-1][:,y_cut]
        for d in range(1,depth_show):
            dens_low = np.sum(NetCDFInitialization.MLD_integrator(source_url, np.ma.array(d*np.ones((y1-y0,x1-x0)),mask=False), t=t, x0=x0, x1=x1, y0=y0, y1=y1) * s_pot_densities, axis=0)
            if d == 1:
                dens_up = 0.0
            else:
                dens_up  = np.sum(NetCDFInitialization.MLD_integrator(source_url, np.ma.array((d-1)*np.ones((y1-y0,x1-x0)),mask=False), t=t, x0=x0, x1=x1, y0=y0, y1=y1) * s_pot_densities, axis=0)
            s_pot_densities_show[:,d] =  (dens_low - dens_up)[:,y_cut]
        s_pot_densities_show.mask = (s_pot_densities_show<1000.0)
    
        s_pot_densities_shows.append(s_pot_densities_show)

    t_start = t_range[0]
    t_stop  = t_range[1]
    t = Dataset(source_url).variables['ocean_time'][t_start:t_stop]

    movie_frames = len(s_pot_densities_shows)

    ax, sp = plotDens(fig, 
                            s_pot_densities_shows[0],
                            t[0], 
                            comment="Potential densities (FjordOS)",
                            **kwargs)


    #Helper function which simulates and plots the solution    
    def animate(i):
        t_now = t[0] + (i / (movie_frames-1)) * (t[-1] - t[0]) 

        k = np.searchsorted(t, t_now)
        if (k >= len(s_pot_densities_shows)):
            k = len(s_pot_densities_shows) - 1
        j = max(0, k-1)
        if (j == k):
            k += 1
        s = (t_now - t[j]) / (t[k] - t[j])

        plotDens(fig, 
                        ((1-s)*s_pot_densities_shows[j] + s*s_pot_densities_shows[k]), 
                        t_now, 
                        comment=comment,
                        **kwargs, ax=ax, sp=sp)

        clear_output(wait = True)
        #print(progress.getPrintString(i / (movie_frames-1)))

    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(movie_frames), interval=250)
    plt.close(fig)
    
    return anim