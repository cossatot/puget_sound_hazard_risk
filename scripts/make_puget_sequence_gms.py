import json
import time
import numpy as np
#import matplotlib.pyplot as plt

from aftershock_gms import *


# data prep
print('preparing data')
with open('../data/fault_zone_traces.geojson', 'r') as f:
    trace_dict = json.load(f)
    traces = trace_dict['features']

traces[0]['properties']['Mw'] = 7.32
traces[1]['properties']['Mw'] = 7.71
traces[2]['properties']['Mw'] = 7.04
traces[3]['properties']['Mw'] = 6.52
traces[4]['properties']['Mw'] = 6.94
traces[5]['properties']['Mw'] = 6.62
traces[6]['properties']['Mw'] = 6.31
traces[7]['properties']['Mw'] = 6.55
traces[8]['properties']['Mw'] = 7.32
traces[9]['properties']['Mw'] = 7.08
traces[10]['properties']['Mw'] = 7.10


# study area
print('making sites')
lons = np.arange(-124., -121., 0.1)
lats = np.arange(47., 49., 0.1)

lons, lats = np.meshgrid(lons, lats)
mesh = RectangularMesh(lons, lats, depths=None)
sites = SiteCollection([Site(location=Point(lon, lat), vs30=760.,
                             vs30measured=True, z1pt0=40., z2pt5=1.)
                        for lon, lat in zip(mesh.lons.flatten(),
                                            mesh.lats.flatten())])

# mainshock ruptures
print('making ruptures')
mainshock_ruptures = {trace['properties']['fault']: trace_to_rupture(trace)
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
for k in aftershock_ruptures.keys():
    parallel_aftershock_gms(aftershock_ruptures[k], sites, n_jobs=-1)
t_gm_3 = time.time()
print('done with ground motion calcs in {0:.1f} m'.format((t_gm_3-t_gm_2)/60))

