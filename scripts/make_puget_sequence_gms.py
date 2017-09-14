import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs


from aftershock_gms import *


# data prep
print('preparing data')


#fault_geojson = '../data/fault_zone_traces.geojson'
fault_geojson = '../data/puget_sound_ruptures.geojson'

with open(fault_geojson, 'r') as f:
    trace_dict = json.load(f)
    traces = trace_dict['features']


# study area
print('making sites')
lons = np.arange(-124., -121., 0.01)
lats = np.arange(47., 49., 0.01)

lons, lats = np.meshgrid(lons, lats)
mesh = RectangularMesh(lons, lats, depths=None)
sites = SiteCollection([Site(location=Point(lon, lat), vs30=760.,
                             vs30measured=True, z1pt0=40., z2pt5=1.)
                        for lon, lat in zip(mesh.lons.flatten(),
                                            mesh.lats.flatten())])

# mainshock ruptures
print('making ruptures')
mainshock_ruptures = {trace['properties']['event']: trace_to_rupture(trace)
                      for trace in traces}


print('calculating mainshock ground motions')
t_gm_0 = time.time()
mainshock_gms = {k: ground_motion_from_rupture(rup, sites=sites)
                 for k, rup in mainshock_ruptures.items()}
t_gm_1 = time.time()
print('done with ground motion calcs in {0:.1f} s'.format(t_gm_1-t_gm_0))

# aftershocks
n_days = 1000 # days after event
cutoff_mag = 4.5

print('making aftershock sequences')

aftershock_ruptures = {k: make_aftershock_rupture_sequence(rup, n_days,
                                                     min_return_mag=cutoff_mag)
                       for k, rup in mainshock_ruptures.items()}

# ground motions in parallel
print('calculating aftershock ground motions')
t_gm_2 = time.time()
#for k in aftershock_ruptures.keys():
#    calc_aftershock_gms(aftershock_ruptures[k], sites, n_jobs=-1, _joblib=False)
t_gm_3 = time.time()
print('done with ground motion calcs in {0:.1f} m'.format((t_gm_3-t_gm_2)/60))


sites = pd.read_csv('../results/nw_washington_sites.csv',
                    names=['lon', 'lat'])

plt.figure(figsize=(10,6))

ax = plt.axes(projection=ccrs.PlateCarree())

gms = ax.pcolormesh(lons, lats, 
              mainshock_gms['West_Point_Sewer_Log_Death'][PGA()].reshape(
                  lons.shape),
              cmap='viridis', alpha=0.6)
plt.colorbar(gms, shrink=0.5, label='PGA')

ax.coastlines(resolution='10m')

ax.scatter(sites.lon, sites.lat, c='k', s=1)

ax.set_extent((-124.5, -121., 46.5, 49.2))

plt.show()
