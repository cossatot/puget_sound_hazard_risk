#import sys; sys.path.append('/home/rstyron/src/GEM/oq-engine')
import sys; sys.path.append('/Users/itchy/src/oq-engine')


from openquake.hazardlib.source import BaseRupture
from openquake.hazardlib.geo import Point, Line, SimpleFaultSurface, Mesh, RectangularMesh
from openquake.hazardlib.calc import ground_motion_fields
from openquake.hazardlib.gsim.campbell_bozorgnia_2008 import CampbellBozorgnia2008
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.imt import PGA
from openquake.hazardlib.correlation import JB2009CorrelationModel
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014
from openquake.hazardlib.calc import ground_motion_fields


from openquake.hazardlib.geo.surface import PlanarSurface
from openquake.hazardlib.geo.geodetic import point_at
from openquake.hazardlib.scalerel.wc1994 import WC1994

import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

try:
    from joblib import Parallel, delayed
    _joblib = True
except ImportError:
    _joblib = False



#####
# Mainshock ruptures from fault traces
#####

def trace_to_surface(trace, **kwargs):
    coords = trace['geometry']['coordinates'][0]
    point_list = [Point(*c) for c in coords]
    
    sfs_args = kwargs
    
    if 'dip' not in sfs_args:
        sfs_args['dip'] = trace['properties']['dip']
        
    sfs = SimpleFaultSurface.from_fault_data(fault_trace=Line(point_list),
                                             **sfs_args)
    return sfs


def trace_surface_to_rupture(trace, sfs, **kwargs):
    rupture = BaseRupture(mag=trace['properties']['Mw'],
                          rake=trace['properties']['rake'],
                          surface=sfs,
                          hypocenter=sfs.get_middle_point(),
                          **kwargs
                          )
    return rupture


def trace_to_rupture(trace, trace_args=None, rupture_args=None):
    
    if trace_args is None:
        trace_args = {}
        
    if rupture_args is None:
        rupture_args = {}
    
    if 'tectonic_region_type' not in rupture_args:
        rupture_args['tectonic_region_type'] = 'Active Shallow Crust'
    if 'source_typology' not in rupture_args:
        rupture_args['source_typology'] = None
    
    if 'upper_seismogenic_depth' not in trace_args:
        trace_args['upper_seismogenic_depth'] = 0.
    if 'lower_seismogenic_depth' not in trace_args:
        trace_args['lower_seismogenic_depth'] = 20.
    if 'mesh_spacing' not in trace_args:
        trace_args['mesh_spacing'] = 2.
    
    surface = trace_to_surface(trace, **trace_args)
    rupture = trace_surface_to_rupture(trace, surface, **rupture_args)
    
    return rupture



#####
# Aftershock generation and rupture making
#####

aft_distance_probs = np.hstack(((0.5/np.linspace(1,20,100)[::-1] + 0.5), 
                                1/np.linspace(1,10,50)))
aft_distance_probs = np.hstack([aft_distance_probs[::-1], 
                                aft_distance_probs])
aft_axis_distance = np.linspace(-0.75, 0.75, len(aft_distance_probs))


def sample_aftershock_coords(mainshock, n_aftershocks, min_width=5,
                             min_depth=4, max_depth=20.):
    main_midpt_lat = mainshock.surface.get_middle_point().latitude
    main_midpt_lon = mainshock.surface.get_middle_point().longitude
    
    width = mainshock.surface.get_width()
    length = mainshock.surface.get_area() / width
    surface_width = width * np.sin(np.radians(mainshock.surface.get_dip()))
    if surface_width < min_width:
        surface_width = min_width
        
    along_strike_distance = length * inverse_transform_sample(aft_axis_distance,
                                                          aft_distance_probs,
                                                          n_aftershocks)
        
    strike_perp_distance = width * inverse_transform_sample(aft_axis_distance,
                                                            aft_distance_probs,
                                                            n_aftershocks)
    aftershock_dists = np.sqrt(along_strike_distance**2 
                               + strike_perp_distance**2)
    
    aftershock_angle_from_strike = np.degrees(np.arctan2(strike_perp_distance,
                                                         along_strike_distance))
    
    aftershock_az =  (aftershock_angle_from_strike 
                      + mainshock.surface.get_strike())
    
    aftershock_lons, aftershock_lats = point_at(main_midpt_lon, main_midpt_lat,
                                                aftershock_az, aftershock_dists)
    
    aftershock_depths = np.random.uniform(min_depth, max_depth, n_aftershocks)
    
    return aftershock_lons, aftershock_lats, aftershock_depths


def make_aftershock_sequence(mainshock, num_days, min_mag=3.,
                             min_return_mag=4.):
    '''
    Aftershock time and magnitude parameters from
    Shcherbakov
    '''
    days = np.arange(num_days)
    Mms = mainshock.mag
    p = 1.1
    b=1.
    B = 1.2
    del_M = 1.1
    Ms = np.linspace(min_mag, Mms - del_M)
    
    def c(M, c_m_star=0.03, B=B, Mms=Mms, p=p, b=b, del_M=del_M):
        #return c_m_star * 10**((B-b) * (Mms - del_M - M) / (p-1))
        return c_m_star

    def r(t, M, p=p, del_M=del_M, Mms=Mms):
        num = (p-1) * 10**(Mms - del_M - M)
        denom = c(M) * (1 + t / c(M))**p
        
        return num / denom
    
    def sample_aftershocks_at_t(t, Ms=Ms):
    
        # Get total number of aftershocks for that day
        
        # N is a float. We will take the integer and then
        # potentially add 1 more based on the probability
        # of the remainder.
                
        rt = r(t, Ms)
                
        N = rt[0]
        N_ = int(N)
        n_ = N - N_
        N_eqs = N_ + 1 if np.random.rand() <= n_ else N_
        
        dr_dm = -np.gradient(rt)
        
        M_samps = inverse_transform_sample(Ms, dr_dm, N_eqs)
        
        zeros = [M_samps == 0.]
        
        while np.sum(zeros) > 0:
            zeros = [M_samps == 0.]
            new_samps = inverse_transform_sample(Ms, dr_dm, np.sum(zeros))
            M_samps[zeros] = new_samps
            zeros = [M_samps == 0.]
        
        return M_samps
        
        
    def sample_aftershock_sequence(days, Ms):
        
        sample_list = []
        
        for t in days:
            dMs = sample_aftershocks_at_t(t, Ms)
            eq_arr = np.vstack([np.ones(len(dMs)) * t, dMs]).T
            sample_list.append(eq_arr)
            
        return np.vstack(sample_list)
    
    aftershock_sequence = sample_aftershock_sequence(days, Ms)
    # return aftershocks greater than min_return_mag:
    aft_filter = aftershock_sequence[aftershock_sequence[:,1] > min_return_mag]
    return aft_filter


def make_aftershock_dict(mainshock, Mw, lon, lat, depth):
    aft_d = {}
    aft_d['Mw'] = Mw
    aft_d['lat'] = lat
    aft_d['lon'] = lon
    aft_d['depth'] = depth
    
    width = mainshock.surface.get_width()
    length = mainshock.surface.get_area() / width
    aft_d['aspect_ratio'] = length / width
    aft_d['rake'] = mainshock.rake
    aft_d['strike'] = mainshock.surface.get_strike()
    aft_d['dip'] = mainshock.surface.get_dip()
    
    return aft_d


def make_aftershock_surface(aft_d, scale_relationship=WC1994, mesh_spacing=2.):
    rupture_area = scale_relationship().get_median_area(aft_d['Mw'],
                                                        aft_d['rake'])
    
    length = (rupture_area * aft_d['aspect_ratio'])**0.5
    width = length / aft_d['aspect_ratio']
    
    mid_upper_lon, mid_upper_lat = point_at(aft_d['lon'], aft_d['lat'],
                                      aft_d['strike'] - 90, 
                                      width * np.sin(np.radians(aft_d['dip'])))
    mid_lower_lon, mid_lower_lat = point_at(aft_d['lon'], aft_d['lat'], 
                                      aft_d['strike'] + 90, 
                                      width * np.sin(np.radians(aft_d['dip'])))
    
    ul_lon, ul_lat = point_at(mid_upper_lon, mid_upper_lat, 
                              aft_d['strike']-180, length/2)
    ur_lon, ur_lat = point_at(mid_upper_lon, mid_upper_lat, 
                              aft_d['strike'], length/2)
    ll_lon, ll_lat = point_at(mid_upper_lon, mid_upper_lat, 
                              aft_d['strike']-180, length/2)
    lr_lon, lr_lat = point_at(mid_upper_lon, mid_upper_lat, 
                              aft_d['strike'], length/2)
    
    upper_surface_depth = (aft_d['depth'] - width 
                           * np.cos(np.radians(aft_d['dip'])))

    if upper_surface_depth < 0:
        upper_surface_depth = 0.

    lower_surface_depth = (aft_d['depth'] + width 
                           * np.cos(np.radians(aft_d['dip'])))
    
    ul_corner = Point(ul_lon, ul_lat, upper_surface_depth)
    ll_corner = Point(ll_lon, ll_lat, lower_surface_depth)
    ur_corner = Point(ur_lon, ur_lat, upper_surface_depth)
    lr_corner = Point(lr_lon, lr_lat, lower_surface_depth)
    
    return PlanarSurface.from_corner_points(mesh_spacing, ul_corner, ur_corner,
                                            lr_corner, ll_corner)


def make_aftershock_rupture_from_dict(aft_d):
    surf = make_aftershock_surface(aft_d)
    
    rupture = BaseRupture(mag=aft_d['Mw'],
                          rake=aft_d['rake'],
                          surface=surf,
                          hypocenter=surf.get_middle_point(),
                          #**kwargs
                          source_typology=None,
                          tectonic_region_type='Active Shallow Crust'
                          )
    return rupture


def make_aftershock_rupture_from_params(mainshock, Mw, lon, lat, depth):
    aft_d = make_aftershock_dict(mainshock, Mw, lon, lat, depth)
    aft_d['rupture'] = make_aftershock_rupture_from_dict(aft_d)
    return aft_d


def make_aftershock_rupture_sequence(mainshock, num_days, min_return_mag=4.):
    aftershock_days_mags = make_aftershock_sequence(mainshock, num_days,
                                                min_return_mag=min_return_mag)
    n_eqs = aftershock_days_mags.shape[0]
    
    aft_lons, aft_lats, aft_depths = sample_aftershock_coords(mainshock, n_eqs)
    
    aftershocks = {}
    
    for i in range(n_eqs):
        Mw = aftershock_days_mags[i,1]
        d = aftershock_days_mags[i,0]
        lon = aft_lons[i]
        lat = aft_lats[i]
        depth = aft_depths[i]
        aftershocks[i] = make_aftershock_rupture_from_params(mainshock, Mw,
                                                             lon, lat, depth)
        aftershocks[i]['day'] = d
        
    return aftershocks

#####
# ground motions
#####

def ground_motion_from_rupture(rupture, sites=None, imts=[PGA()],
                               gsim=ChiouYoungs2014(),
                               truncation_level=0, realizations=1, **kwargs):

    gm = ground_motion_fields(rupture=rupture,
                              sites=sites,
                              imts=imts,
                              gsim=gsim,
                              truncation_level=truncation_level,
                              realizations=realizations)
    return gm


def calc_aftershock_gms(aftershock_dict, sites, n_jobs=-1, verbose=2,
                        _joblib=_joblib, **kwargs):
    if _joblib == True:
        return parallel_aftershock_gms(aftershock_dict, sites, n_jobs=n_jobs,
                                       verbose=verbose)
    else:
        return serial_aftershock_gms(aftershock_dict, sites, **kwargs)


def serial_aftershock_gms(aftershock_dict, sites, **kwargs):
    afts = aftershock_dict.keys()
    gm_list = [ground_motion_from_rupture(aftershock_dict[i]['rupture'], 
                                          sites=sites, **kwargs)
               for i in afts]

    for i in afts:
        aftershock_dict[i]['ground_motion'] = gm_list[i]

    return


def parallel_aftershock_gms(aftershock_dict, sites, n_jobs=-1, verbose=2,
                            **kwargs):
    afts = aftershock_dict.keys()
    gm_list = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(
                                                ground_motion_from_rupture)
                                                (aftershock_dict[i]['rupture'], 
                                                 sites=sites, **kwargs)
                                                for i in afts)
    for i in afts:
        aftershock_dict[i]['ground_motion'] = gm_list[i]

    return


#####
# PDF sampling functions
#####

def normalize_pmf(x, px):
    '''
    Normalizes a probability mass function
    given x and p(x) values
    '''
    if x[0] > x[1]:
        denom = np.trapz(px[::-1], x[::-1])
    else:
        denom = np.trapz(px, x)

    if denom != 0.:
        px_norm = px / denom
    else: px_norm = px * 0.

    return x, px_norm


def Pdf(x, px, normalize=True):
    """
    Returns an `interp1d` class based on the
    relative probabilities x and px
    """
    if np.isscalar(x):
        def _pdf(interp_x, x):
            if interp_x == x:
                return 1.
            else:
                return 0.
    else:
        if normalize == True:
            x, px = normalize_pmf(x, px)

        _pdf = interp1d(x, px, bounds_error=False, fill_value=0.)
    return _pdf


def Cdf(x, px, normalize=True):
    """docstring"""
    _cdf = interp1d(x, cumtrapz(px, initial=0.) / np.sum(px), 
                    fill_value=1., bounds_error=False)

    return _cdf


def inverse_transform_sample(x, px, n_samps):
    """
    lots o' docs
    """
    if len(x) == 1:
        return np.ones(n_samps) * px

    else:
        cdf = Cdf(x, px)

        cdf_interp = interp1d(cdf(x), x, bounds_error=False,
                              fill_value=0.)

        samps = np.random.rand(n_samps)

    return cdf_interp(samps)



