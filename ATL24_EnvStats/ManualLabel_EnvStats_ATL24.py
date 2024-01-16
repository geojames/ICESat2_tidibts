#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'James T. Dietrich'
__contact__ = 'james.dietrich@austin.utexas.edu'
__copyright__ = '(c) James Dietrich 2022'
__license__ = 'MIT'
__date__ = '2022-Nov-01'
__version__ = '0.1'
__status__ = "initial release"
__url__ = "..."

"""
Name:           ManualLabel_EnvStats_ATL24.py.py
Compatibility:  Python 3.11
Description:    A description of your program

Requires:       Modules/Libraries required

Dev ToDo:       1)

AUTHOR:         James T. Dietrich
ORGANIZATION:   University of Texas @ Austin
                3D Geospatial Labratory
                Center for Space Research
Contact:        james.dietrich@austin.utexas.edu
"""
#%%

import numpy as np
import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
import sys
import warnings
import os
import h5py
#import json
from datetime import datetime as dt
from glob import glob
#import shutil

import argparse
import requests
import rasterio as rio

warnings.filterwarnings("ignore")
     

#%%

def find_scuba(h5_file,scuba_dir):
    
    pattern = "/**/all_segment_points*%s*.csv"%os.path.basename(h5_file)[:-10]
    
    return(glob(scuba_dir + pattern,recursive=True))

def path_fix(folder):
    
    if (folder[-1] == '/') or (folder[-1] == '\\'):
        folder = folder[:-1]
        
    return folder

def get_kd_values(date, outpath = 'c:/temp', keep=False):
    '''
    Downloads an 8-day NOAA20 VIIRS Kd490 image from NASA OceanColor

    Parameters
    ----------
    date : string data from ATL-03 file "%Y-%m-%d_%H%MZ"
        suggested 
        start_dt_obj = dt.strptime(h5['ancillary_data/data_start_utc'][0].decode(),
                                           "%Y-%m-%dT%H:%M:%S.%fZ")
        atl03_date = dt.strftime(start_dt_obj,"%Y-%m-%d_%H%MZ")
    out : file path
        file path for output of the KD images. The default is 'c:/temp'.
    keep : boolean, optional
        True = files will be kept and stored with their orignal filename.
        False = files will be overwritten with the filename kd_490.nc
        The default is False.

    Returns
    -------
    kd_out = file path to the downloaded KD image

    '''
    
    # check outpath for trailing slash
    kd_out = path_fix(outpath)
    
    # OpenDAP file structure:
    # http://oceandata.sci.gsfc.nasa.gov/opendap/VIIRSJ1/L3SMI/2019/0101/JPSS1_VIIRS.20190101_20190108.L3m.8D.KD.Kd_490.4km.nc.dap.nc4

    # get start datetime from the granule, establish the 8-day cycle start dates for
    #   the year, generate the time deltas to the granule date
    h5_date = dt.strptime(date,"%Y-%m-%d_%H%MZ")
    eight_day = pd.date_range(start='1/1/%s'%h5_date.year, end='12/31/%s'%h5_date.year,freq='8d')
    td = eight_day - h5_date

    # calculate the closest starting point of the 8-day cycle based on the
    #   granule time - infer the ending point +7 days
    dl_start = eight_day[td[td<=pd.Timedelta(0)].argmax()]
    dl_end = dl_start + pd.Timedelta('7d')
    
    if dl_end.year != dl_start.year:
        dl_end = dt.strptime("%s-12-31"%dl_start.year,"%Y-%m-%d")

    # build the download string
    oc_start = 'http://oceandata.sci.gsfc.nasa.gov/opendap/VIIRSJ1/L3SMI/%s/%s/'%(dl_start.year,dl_start.strftime('%m%d'))
    oc_dates = 'JPSS1_VIIRS.%s_%s'%(dl_start.strftime('%Y%m%d'), dl_end.strftime('%Y%m%d'))
    oc_end = ".L3m.8D.KD.Kd_490.4km.nc"
    dap_end = '.dap.nc4'
    oc_file = oc_start + oc_dates + oc_end + dap_end

    # set the file_put path and filename
    if keep:
        file_out = kd_out + "/" + oc_dates + oc_end
    else:
        file_out = kd_out + "/kd_490_temp.nc"

    # submit download request, if OK (no errors) wrtie the file out
    #   if no file, kill everything
    
    r = requests.get(oc_file)

    if r.ok:
        with open(file_out,'wb') as f:
          
            # write the contents of the response (r.content)
            # to a new file in binary mode.
            f.write(r.content)
            
            return(file_out, r.ok)
    else:
        print('* Trying NRT *')
        oc_end = ".L3m.8D.KD.Kd_490.4km.NRT.nc"
        dap_end = '.dap.nc4'
        oc_file = oc_start + oc_dates + oc_end + dap_end
        r = requests.get(oc_file)
        
        if r.ok:
            with open(file_out,'wb') as f:
              
                # write the contents of the response (r.content)
                # to a new file in binary mode.
                f.write(r.content)
                
                return(file_out, r.ok)
        else:
            print("unable to download file %s | Status: %i, %s"%(oc_dates + oc_end + dap_end,
                                                                 r.status_code, r.reason))
            return(file_out, r.ok)

#%%
def clip_to_aoi(data, aoi):
    '''
        Clip a profile dataframe to the associated AOI, ensuring that whole
        along track segments are retained during the clip process.

        Argument:
                data: Profile.data dataframe.
                aoi: Openoceans GeoAOI object.

        Returns:
                clipped_data: clipped Profile.data dataframe.

    '''
    pts_segs = data.segment_id[aoi.contains(data.geometry)].unique()
    
    if pts_segs.shape[0] == 0:
        raise Exception("No Valid Data in AOI")

    return data[data.segment_id.isin(pts_segs)]
    
    

def _convert_seg_to_ph_res(segment_res_data):
    '''
    Upsamples segment-rate data from h5 file to the photon-rate
    - gtx/geolocation and gtx/geophysical_corr

    Parameters
    ----------
    segment_res_data : array
        an array of segment rate values from the H5 file. Must include segment_ph_cnt

    Returns
    -------
    dataframe with size (total photon count x input columns) with upsampled 
        segment level data

    '''

    # trim away segments without photon data (seg_ph_cnt = 0)

    segs_with_data = segment_res_data['segment_ph_cnt'] != 0
    segment_clip = segment_res_data[segs_with_data].copy()
    
    # Calculate the cumulative sum array trimmed seg_ph_counts
    #   initialize photon resolution array
    segment_clip['seg_cumsum'] = segment_clip['segment_ph_cnt'].cumsum()
    seg_max = segment_clip['seg_cumsum'].iloc[-1]
    ph_res_data = pd.DataFrame(np.arange(1,seg_max+1),columns=['idx'])
    
    
    # using the cumsum array as the indicies for mapping the segment rate
    #   data to the photon resolution array
    ph_res_data = pd.merge(ph_res_data, segment_clip,
                           left_on='idx',right_on='seg_cumsum', how='left')
    ph_res_data.bfill(inplace=True)
    
    # clean up temp columns (commnet out for testing)
    ph_res_data.drop(['idx','seg_cumsum'],axis=1,inplace=True)
    
    return ph_res_data
    
    
def read_h5(f, gtxx, aoi, verbose = False):
    
    dt_start = dt.now()
    
    # nested dictonary of beam spot mapping
    #   top: sc_orient - 0 or 1, beam_name - from h5 input
    beam_info = {0:{'gt1l':{'beam_strength':'strong','atlas_spot':1,'track_pair':1},
                    'gt1r':{'beam_strength':'weak','atlas_spot':2,'track_pair':1},
                    'gt2l':{'beam_strength':'strong','atlas_spot':3,'track_pair':2},
                    'gt2r':{'beam_strength':'weak','atlas_spot':4,'track_pair':2},
                    'gt3l':{'beam_strength':'strong','atlas_spot':5,'track_pair':3},
                    'gt3r':{'beam_strength':'weak','atlas_spot':6,'track_pair':3}},
                 1:{'gt1l':{'beam_strength':'weak','atlas_spot':6,'track_pair':1},
                    'gt1r':{'beam_strength':'strong','atlas_spot':5,'track_pair':1},
                    'gt2l':{'beam_strength':'weak','atlas_spot':4,'track_pair':2},
                    'gt2r':{'beam_strength':'strong','atlas_spot':3,'track_pair':2},
                    'gt3l':{'beam_strength':'weak','atlas_spot':2,'track_pair':3},
                    'gt3r':{'beam_strength':'strong','atlas_spot':1,'track_pair':3}}}
    
    # day/night array for profile info
    #   Day = Solar_angle > 6° above the horizon
    #   Twilight = Sunrise/sunset (6°) to civil twilight (-6°)
    #   Night >= -7°
    day_night_ang = np.array([6,-6,-7])
    day_night_name = np.array(['day','twilight','night'])
    
    height_keys = ['delta_time', 'dist_ph_along', 'h_ph', 'lat_ph', 'lon_ph', 
                   'signal_conf_ph']
    
    geoloc_keys = ['segment_dist_x','segment_id', 'segment_length', 'segment_ph_cnt','solar_elevation','solar_azimuth']
    # geoloc_keys = ['knn', 'ph_index_beg', 'segment_dist_x', 
    #                'segment_id', 'segment_length', 'segment_ph_cnt',
    #                'sigma_across', 'sigma_along', 'sigma_h', 'sigma_lat', 
    #                'sigma_lon', 'solar_azimuth', 'solar_elevation','surf_type']

    geophys_keys = ['geoid']
    
    if verbose:
        t1 = (dt.now()-dt_start).total_seconds()
        print('Time Check 1: ', t1)
        print('  Strating h5 read...')
        
    #pho = gpd.GeoDataFrame()
    pho = pd.DataFrame()
    #pre-test lat/long read
    try:
        lat = np.array(f[gtxx + '/heights/lat_ph'])
        lon = np.array(f[gtxx + '/heights/lon_ph'])
    except KeyError as e:
        if str(e) == "'Unable to open object (component not found)'":
            raise MissingTrackError(f"H5 file does not contain data for {gtxx}. Try another track or make sure the ATL03 file has data.")
            
        else:
            raise e

    except Exception as e:
        raise e
        
    # establish photon index for future use (zero base)
    pho['photon_index'] = np.arange(0,lat.shape[0])
    
    #track info read
    start_dt_obj = dt.strptime(f['ancillary_data/data_start_utc'][0].decode(),
                                       "%Y-%m-%dT%H:%M:%S.%fZ")
    start_dt = dt.strftime(start_dt_obj,"%Y-%m-%d_%H%MZ")
    # end_dt = dt.strftime(dt.strptime(f['ancillary_data/data_end_utc'][0].decode(),
    #                                  "%Y-%m-%dT%H:%M:%S.%fZ"),"%Y-%m-%d %H:%M:%S.%f")
    
    # use region to determine track direction
    region = f['/ancillary_data/start_region'][0]
    if (region <= 3) or (region >= 12):
        orbit_dir= 'ASCENDING'

    elif (region == 4) or (region == 11):
        orbit_dir = 'POLE'

    else:
        orbit_dir = 'DESCENDING'
    
    solar_ang = f[gtxx + '/geolocation/solar_elevation'][:]
    med_solar_ang = np.nanmedian(solar_ang[solar_ang < 360])
    day_night = day_night_name[np.abs(day_night_ang - med_solar_ang).argmin()].upper()
    
    date = start_dt
    rgt = str(f['/orbit_info/rgt'][0]).zfill(4)
    cc = str(f['/orbit_info/cycle_number'][0]).zfill(2)
    reg = str(f['/ancillary_data/start_region'][0]).zfill(2)
    rel = f['/ancillary_data/release'][0].decode('UTF-8').replace(" ", "")
    # beam info from lookup dict
    beam_info_dict = beam_info[f['/orbit_info/sc_orient'][0]][gtxx]
     
    alt03_str = "{} | RGT{} | {}-{} | CYC{} | {} | {}".format(date, rgt, gtxx.upper(),
                                                      beam_info_dict['beam_strength'].upper(),cc, 
                                                      day_night, orbit_dir[:3]) 
    pho['timeofday'] = day_night
    pho['beam_strength'] = beam_info_dict['beam_strength'].upper()
    
    # feedback
    if verbose:
        t2 = (dt.now()-dt_start).total_seconds()
        print('Time Check 2: %0.3f [Δt %0.3f]'%(t2,t2-t1))
        print('  Finished info > building pho geometry')
        
    # create the main photon dataframe
    #   initailize with geopandas geometry from point lat,long
    #   set gdf crs to WGS84
    # pho.set_geometry(gpd.points_from_xy(np.array(pho['lon_ph']),
    #                                      np.array(pho['lat_ph'])),inplace=True)
    # pho.set_crs("EPSG:4326", inplace=True)
    # pts_geom = shapely.MultiPoint(np.array([[pho['lon_ph'].values],
    #                                [pho['lat_ph'].values]]).T)
    # #pho.set_crs("EPSG:4326", inplace=True)
    
    
    
    # feedback
    if verbose:
        t3 = (dt.now()-dt_start).total_seconds()
        print('Time Check 3: %0.3f [Δt %0.3f]'%(t3,t3-t2))
        print('  Finished geometry > reading h5 keys')
             

    # read all the height keys and create dataframe columns for each
    #   signal_conf_ph requires spliting into land, ocean, inland water
    for i,k in enumerate(height_keys):
        if verbose:
            print('\t -heights/',k)
        if k == 'signal_conf_ph':
            # ocean conf
            pho['signal_conf'] = np.array(f[gtxx + '/heights/' + k][:,1])
            
        else:
            pho[height_keys[i]] = np.array(f[gtxx + '/heights/' + k][:])
            
    # initalize a dataframe with segment level data values
    seg_df = pd.DataFrame()
    
    # read all the geolocation keys and create dataframe columns for each
    #   all segment level data is upsampled with _convert_seg_to_ph_res
    #   - surf_type requires spliting into land, ocean, inland water
    #   - velocity is calculated as the hypotenuse of the vectors
    for i,k in enumerate(geoloc_keys):
        if verbose:
            print('\t -geoloc/',k)
        
        if k == 'surf_type':
            surf_typ = np.array(f[gtxx + '/geolocation/' + k][:])
            seg_df['surf_type_land'] = surf_typ[:,0]
            seg_df['surf_type_ocean'] = surf_typ[:,1]
            seg_df['surf_type_inlandwater'] = surf_typ[:,4]

        else:
            seg_df[geoloc_keys[i]] = np.array(f[gtxx + '/geolocation/' + k][:])
    
    # read all the geophys_corr keys and create dataframe columns for each
    #   all segment level data is upsampled with _convert_seg_to_ph_res
    for i,k in enumerate(geophys_keys):
        
        if verbose:
            print('\t -geophys/',k)

        seg_df[geophys_keys[i]] = np.array(f[gtxx + '/geophys_corr/' + k][:])
    
    # feedback
    if verbose:
        t4 = (dt.now()-dt_start).total_seconds()
        print('Time Check 4: %0.3f [Δt %0.3f]'%(t4,t4-t3))
        print('  Finished reading keys > converting segment data - total photons: %i'%pho.shape[0])
        
    seg_upsample = _convert_seg_to_ph_res(seg_df)
    
    pho_merge = pd.merge(pho,seg_upsample,left_index=True,right_index=True)
    
    # feedback
    if verbose:
        t5 = (dt.now()-dt_start).total_seconds()
        print('Time Check 5: %0.3f [Δt %0.3f]'%(t5,t5-t4))
        print('  Finished converting seg > adding extra fields...')
               
    # calculate geodetic heights
    #   ellipsoidal height (heights/h_ph) - geoid (geophys/geoid)
    pho_merge['z_ph'] = pho_merge.h_ph - pho_merge.geoid
    
    if verbose:
        t6 = (dt.now()-dt_start).total_seconds()
        print('Time Check 6: %0.3f [Δt %0.3f]'%(t6,t6-t5))
        print('  Finished from_h5 > cliping to AOI')
       
        
    #pho_clip = clip_to_aoi(pho_merge,aoi)
        
    # returns the pho, info_dict, and aoi to the class
    #return pho_clip, alt03_str, date, beam_info_dict
    return pho_merge, alt03_str, date, beam_info_dict           

def read_atl09(data_03, data_09, track_pair):
    
    profile = 'profile_%i'%track_pair
    
    atm = pd.DataFrame()
    
    with h5py.File(data_09, 'r') as f:
        
        atm['a_deltatime'] = f[profile + '/high_rate/delta_time'][:]
        atm['asr'] = f[profile + '/high_rate/apparent_surf_reflec'][:]
        atm['c_pct'] = f[profile + '/high_rate/asr_cloud_probability'][:]
        atm['backg_c'] = f[profile + '/high_rate/backg_c'][:]
        atm['backg_theoret'] = f[profile + '/high_rate/backg_theoret'][:]
        atm['c_flag'] = f[profile + '/high_rate/cloud_flag_asr'][:]
        atm['cloud_fold'] = f[profile + '/high_rate/cloud_fold_flag'][:]
        atm['od_asr'] = f[profile + '/high_rate/column_od_asr'][:]
        atm['msw_flag'] = f[profile + '/high_rate/msw_flag'][:]
        atm['ocean_surf_reflec'] = f[profile + '/high_rate/ocean_surf_reflec'][:]
        atm['a_lat'] = f[profile + '/high_rate/latitude'][:]
        atm['px'] = f[profile + '/high_rate/prof_dist_x'][:]
        atm['py'] = f[profile + '/high_rate/prof_dist_y'][:]
        atm['seg_id'] = f[profile + '/high_rate/segment_id'][:]

    atm_seg = pd.merge(pd.Series(data_03.delta_time.unique(),name='dm_delta_time'), atm,left_on='dm_delta_time',
                        right_on='a_deltatime', how='left')

    atm_seg['c_flag_interp'] = atm_seg['c_flag'].interpolate(method='linear', axis=0)
    atm_seg['c_pct_interp'] = atm_seg['c_pct'].interpolate(method='linear', axis=0)

    atm_seg.ffill(inplace=True)
    
    data_atm = pd.merge(data_03, atm_seg, left_on='delta_time',
                        right_on='dm_delta_time', how='left')

    
    return data_atm

    

#%% MAIN

if __name__ == '__main__':
    
    # baem list for iterative processing
    beam_list = ['gt1l','gt1r','gt2l', 'gt2r', 'gt3l', 'gt3r']
    
    # command line parser
    parser = argparse.ArgumentParser(description = 'SCuBA Environmental Stats Generator')
    
    parser.add_argument("-g","--gran", type=str,
                    help="Folder path to ICEsat-2 *.h5 files")
    parser.add_argument("-l","--label", type=str,
                    help='Top level folder path to labeled files')

    parser.add_argument("-d","--dem", type=str,
                    help="Path to Reference Elevation DEM - Full path")
    parser.add_argument("-o","--outpath", type=str,
                    help="Path to output files",default='')

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    
    # fill in starting variables from argparse
    h5_filepath = args.gran

    label_filepath = args.label
    # aoi_file = args.aoi
    # aoi_name = args.name
    ref_dem = args.dem
    argoutpath = args.outpath
    
    # check paths
    path_ck = []
    # check for correct file paths
    if (os.path.exists(h5_filepath)==False):
        path_ck.append(1)  
    elif (os.path.exists(label_filepath)==False):
        path_ck.append(2)
    # elif (os.path.exists(aoi_file)==False):
    #     path_ck.append(3)
    elif (os.path.exists(ref_dem)==False):
        path_ck.append(3)

    if path_ck != []:
        sys.exit("Check file paths...one folder/file path does not exist %s"%path_ck)
    
    # fix path endings for consistantcy
    h5_filepath = path_fix(h5_filepath)
    label_filepath = path_fix(label_filepath)  
    
    lbl_files = glob(label_filepath + '/**/*.h5',recursive=True)
    
    # atl09_files = glob(h5_filepath + '/ATL09/ATL09*.h5') 
    
    # for each file in the labeled data folder
    for file_lbl in lbl_files:
        
        # progess feedback
        print('\n* %s'%file_lbl)
        
        # timer
        dt_file = dt.now()
        
        # find ATL03 granule file for each labeled file
        atl_bn = os.path.basename(file_lbl)[:33]
        atl03_h5 = ''
        
        if atl_bn[-3:] == '005':
            atl03_h5 = glob(h5_filepath + '/REL005/**/%s*.h5'%atl_bn,recursive=True)[0]
            
        elif atl_bn[-3:] == '006':
            atl03_h5 = glob(h5_filepath + '/REL006/**/%s*.h5'%atl_bn,recursive=True)[0]
        
        # if the number of found granules files == 0 pass on the current granule
        if atl03_h5 == '':
            print("*** No ATL03 found %s"%atl_bn)
            continue
        else:
            pass

        kd_flag = False
        
        for i,b in enumerate(beam_list):
            
            # feedback
            print('\tBeam: %s'%b)
                
            # prep output file path/name and check for already processed files
            # find the common path to the scuba files, build the save file path,
            #   export bathy points to CSV
            if argoutpath == '':
                common_path = os.path.commonpath(h5_filepath) + '/env_stats'
                if os.path.exists(common_path) == False:
                    os.mkdir(common_path)
            else:
                common_path = argoutpath + '/env_stats'
                
                if os.path.exists(argoutpath) == False:
                    os.mkdir(argoutpath)
                    
                if os.path.exists(common_path) == False:
                    os.mkdir(common_path)
            
            save_path = common_path + '/%s_%s_envSTATS.csv'%(os.path.basename(file_lbl)[:-3],b)
            
            if os.path.exists(save_path):
                print("%s already processed"%b)
                continue
            
            # --- label read first ---
            #   try to read labeled file first
            
            try:
                with h5py.File(file_lbl) as f:
                    
                    if args.verbose:
                        print('\tReading Labels...')
                    
                    if b in f.keys():
                    
                        labels = pd.DataFrame(np.array([f[b+'/granule_photon_indices'][:],
                                           f[b+'/manual_label'][:].astype(str)]).T,columns=['lbl_idx','label'])
    
                        labels.lbl_idx = labels.lbl_idx.astype(int)
                        
                        labels['TRUTH'] = 0
                        labels.TRUTH[labels.label == 'Bathymetry'] = 4
                        labels.TRUTH[labels.label == 'Sea Surface'] = 5
                        
                    else:
                        print(' ******  No Labels in %s'%b)
                        continue
                
            except Exception as e:
                continue
                print(' ******  No Labels in %s > %s'%(b,e))
            
            # --- read h5 ---
            # read in h5 photon data for the current beam 
            try:
                with h5py.File(atl03_h5, 'r') as f_h5:
                    profile, alt03_str, profile_date, beam_info_d = read_h5(f_h5, b, aoi=None, verbose=args.verbose)

            except Exception as e:
                print(' ******  Problem in H5 Read ', e)
                continue
            
            # ATL09 wildcard filename
            atl09_wild = "/REL%s/**/ATL09_%s*_%s*.h5"%(atl_bn.split("_")[3],
                                                       atl_bn.split("_")[1][:6],
                                                       atl_bn.split("_")[2][:6])
            
            f_09 = glob(h5_filepath + atl09_wild,recursive=True)
            
            # if ATL09 file availible, read and merge - else merge 03 and lebels
            if len(f_09) > 0:
                atm_prof = read_atl09(profile,f_09[0],beam_info_d['track_pair'])
                profile_l = pd.merge(atm_prof,labels,left_on='photon_index',right_on='lbl_idx')
            else:
                profile_l = pd.merge(profile,labels,left_on='photon_index',right_on='lbl_idx')
            
            if args.verbose:
                print('\t ### Cols_L: ',profile_l.columns)
            
            # min max latitude for labels
            label_latmax = profile_l[profile_l.TRUTH.between(3,6)].lat_ph.max()
            label_latmin = profile_l[profile_l.TRUTH.between(3,6)].lat_ph.min()
            
            if args.verbose:
                print('\t ### LMM: %0.2f - %0.2f'%(label_latmin,label_latmax))
            
            # clip to just merged data frame to the labeled data
            profile_m = profile_l[profile_l.lat_ph.between(label_latmin,label_latmax)]
            
            if args.verbose:
                print('\t ### Cols_M: ',profile_m.shape)
                
            del profile, profile_l 
            
            # --- KD490 Download ---
            if args.verbose:
                print('\tKD_490 Download...')
            
            if kd_flag == False:
                kd_out, req_ok = get_kd_values(profile_date,outpath = common_path,
                                           keep=False)
                if req_ok == True:
                    kd_flag = True
                else:
                    print('********** KD ERROR ************')
                    continue
            
            if args.verbose:
                print('\tKD / DEM sampleing...')
                
            try:
                bathy_points = profile_m.copy(deep=True)
                
                if args.verbose:
                    print('\t ### Bathy points: %i'%bathy_points.shape[0])
        
                # bathy_points['sclass'] = 0
                # bathy_points['sclass'][bathy_points.sea_surf_conf >=0.66] = 5
                # bathy_points['sclass'][bathy_points.bathy_conf > args.bathy_thresh] = 4
        
                coord_list = [(x, y) for x, y in zip(bathy_points.lon_ph, bathy_points.lat_ph)]
                dt1 = dt.now()
                
                if args.verbose:
                    print('\tKD Sample...')
                with rio.open('netcdf:%s:Kd_490'%kd_out) as kd_src:
                    #print('\t\tOpen')
                    bathy_points["kd_490"] = [x[0] for x in kd_src.sample(coord_list)]
                    #print('\t\tRead')
                    bathy_points["kd_490"] = bathy_points["kd_490"] * kd_src.scales[0] + kd_src.offsets[0]
                    #print('\t\tKD532')
                    bathy_points["kd_532"] = 0.68 * (bathy_points["kd_490"] - 0.03) + 0.054
                    bathy_points["kd_zsd"] = 1.15 / (bathy_points["kd_532"] - 0.03)
                
                if args.verbose:
                    print('\tDEM Sample...')
                with rio.open(ref_dem) as dem_src:
                    bathy_points["ref_z"] = [x[0] for x in dem_src.sample(coord_list)]
                    bathy_points["kd_secchiRatio"] = bathy_points["kd_zsd"] / np.abs(bathy_points["ref_z"])
        
                if args.verbose:
                    print('\tStats...')
                
                # basic refraction correction
                #bathy_points.drop(['geometry'],axis=1,inplace=True)
                bathy_points['z_refract_geoid'] = np.nan
                bathy_points.z_refract_geoid.loc[bathy_points.TRUTH == 4] = bathy_points.z_ph * 0.7521117
                
                bathy_points["bathy_error"] = np.nan
                bathy_points["bathy_error"] = bathy_points[bathy_points.TRUTH==4].z_refract_geoid - bathy_points[bathy_points.TRUTH==4].ref_z
                
                # basic surface elevation estimate
                bathy_points["z_surf"] =  np.nan
                bathy_points.z_surf.loc[bathy_points.TRUTH==5] = bathy_points.z_ph
                
                # truth bathy flag
                bathy_points['tb'] = np.nan
                bathy_points.tb[bathy_points.TRUTH==4] = 1
        
        
                # setup statistacs for each geosegment
                bp_stats = bathy_points.groupby('segment_id')
                
                # roughness (surface st dev)
                bathy_sd = bp_stats.z_refract_geoid.aggregate("std").rename('bathy_sd')
                surf_sd = bp_stats.z_surf.aggregate("std").rename('surf_sd')
                
                # point count/density
                # bathy_ptcnt = bp_stats.z_refract_geoid.count().rename('bathy_ptcnt')
                # bathy_ptdens = (bp_stats.z_refract_geoid.count() / bp_stats.segment_length.mean()).rename('bathy_ptdens')
                bathy_t_ptcnt = bp_stats.tb.count().rename('bathy_t_ptcnt')
                bathy_t_ptdens = (bp_stats.tb.count() / bp_stats.segment_length.mean()).rename('bathy_t_ptdens')
                surf_ptdens = (bp_stats.z_surf.count() / bp_stats.segment_length.mean()).rename('surf_ptdens')
                
                #availible bathymety based on input DEM
                avail_bathy = bp_stats.ref_z.aggregate("min").rename('avail_bathy')
                avg_depth = (bp_stats.z_surf.mean() - bp_stats.z_refract_geoid.mean()).rename('avg_depth')
                avg_surf = (bp_stats.z_surf.mean()).rename('avg_surf')
                
                #Concat all segment stats
                bp_concat = pd.concat([bathy_sd,surf_sd,bathy_t_ptdens,bathy_t_ptcnt,
                                       surf_ptdens,avail_bathy,avg_surf,avg_depth],axis=1)
        
                bathy_points = pd.merge(bathy_points,bp_concat,left_on='segment_id',right_index=True,how='outer')
                bathy_points['pt_depth'] = bathy_points.avg_surf - bathy_points.z_refract_geoid
                
                # slope 
                bs_stats = bathy_points[bathy_points.TRUTH==4].groupby('segment_id')
                slope = ((bs_stats.z_ph.max() - bs_stats.z_ph.min())/ bs_stats.segment_length.mean()).rename('slope')
                bathy_points = pd.merge(bathy_points,slope,left_on='segment_id',right_index=True,how='outer')
                bathy_points.loc[(bathy_points.slope == 0) & (bathy_points.bathy_t_ptcnt <= 1), 'slope'] = -0.1
                
                
                if args.verbose:
                    print('\tAvailFlag...')
                
                # bathy_points['avail_flag'] = 0
                # bathy_points.loc[(bathy_points.avail_bathy.between(-42,0)) & (bathy_points.bathy_ptcnt == 0), 'avail_flag'] = 1
                # bathy_points.loc[(bathy_points.avail_bathy.between(-42,0)) & (bathy_points.bathy_ptcnt > 0), 'avail_flag'] = 3
                # bathy_points.loc[(bathy_points.avail_flag == 0) & (bathy_points.bathy_t_ptcnt > 0), 'avail_flag'] = 1
                
                # availibility flag (1 = availible, no labeled, 3 = availible and labeled)
                bathy_points['avail_flagT'] = 0
                bathy_points.loc[(bathy_points.avail_bathy.between(-42,0)) & (bathy_points.bathy_t_ptcnt == 0), 'avail_flagT'] = 1
                bathy_points.loc[(bathy_points.avail_bathy.between(-42,0)) & (bathy_points.bathy_t_ptcnt > 0), 'avail_flagT'] = 3        
                # bathy_points.loc[(bathy_points.avail_flag == 1) & (bathy_points.bathy_t_ptcnt > 0), 'avail_flagT'] = 3
                # bathy_points.loc[(bathy_points.avail_flag == 0) & (bathy_points.bathy_t_ptcnt > 0), 'avail_flagT'] = 3
                
                
                if args.verbose:
                    print('\tReason...')
                # reason
                seg_cut = bathy_points.segment_id.unique()
                seg_num = 10
                min_z = np.array([])
                for i,s in enumerate(seg_cut[::seg_num]):
                    min_z = np.append(min_z,bathy_points[bathy_points.segment_id.between(s,s+(seg_num-1))].z_ph.min())
        
                min_ser = pd.Series(min_z,index=seg_cut[::seg_num],name='min_z_ph')
        
                bathy_points = pd.merge(bathy_points,min_ser,left_on='segment_id',right_index=True,how='outer')
                bathy_points.min_z_ph = bathy_points.min_z_ph.ffill()
                
                if args.verbose:
                    print('\tReason Flag...')

                bathy_points.loc[(bathy_points.bathy_t_ptcnt == 0) & (bathy_points.avail_bathy.between(-42,0)) & (bathy_points.avail_bathy < bathy_points.min_z_ph), 'avail_flagT'] = 2  #telem window
                #bathy_points.loc[(bathy_points.avail_flag == 1) & (bathy_points.avail_bathy.between(-42,0)) & (bathy_points.avail_bathy < bathy_points.min_z_ph), 'avail_flag'] = 2  #telem window     
                
    # -- sig waves heights --
                if args.verbose:
                    print('\tWaves...')
        
                seg_cut = bathy_points.segment_id.unique()
        
                waves = np.array([])
        
                seg_num = 15
                for i,s in enumerate(seg_cut[::seg_num]):
                    waves = np.append(waves,bathy_points[bathy_points.segment_id.between(s,s+(seg_num-1))].z_surf.std())
        
                wv_ser = pd.Series(waves*4,index=seg_cut[::seg_num],name='sig_waveH')
        
                bathy_points = pd.merge(bathy_points,wv_ser,left_on='segment_id',right_index=True,how='outer')
                bathy_points.sig_waveH = bathy_points.sig_waveH.ffill()
            
                # --- output ---
                # CSV output of the resulting stats dataframe (BIG)
                
                if args.verbose:
                    print('\tOutput...')
                
                # Export to CSV 
                bathy_points.to_csv(save_path, index=False)
                
                if args.verbose:
                    print('\t\t%s'%save_path)
                    
            except Exception as e:
                print(' ******  Problem stats section ', e)
                continue
                    
            print((dt.now() - dt_file).total_seconds())