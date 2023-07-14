#!/usr/bin/env python3

import os
import numpy as np
import h5py
#from astropy.cosmology import Planck13
from astropy.cosmology import FlatLambdaCDM
from FIRE_routine import *

print('===============================================================================')
run_name = 'm12_elvis_RomeoJuliet_res3500'
host_num = 2 # 1 for m12 runs; 2 for ELVIS runs
save_path = os.path.join('/export/nfs0home/omyrtaj/data/romeo_juliet', run_name)
print('***Start the routine for ' + run_name + '***')

output_path = os.path.join('/data17/grenache/aalazar/FIRE/GVB',run_name,'output/hdf5')
snapdir_paths = get_snapdir_path(output_path)
for snapdir_path in snapdir_paths[-2:]: #romeo_juliet has 22 snapshots
    snapdir_num = snapdir_path[-3:] # save the last three digit
    print('========================================')
    print('Start reading the snapshot_'+snapdir_num)
    part, header = read_part(snapdir_path)
    print('Data load!')

    print('H0 = '+str(header['hubble']))
    print('Om0 = '+str(header['omega_matter']))
    print('Build the FlatLambdaCDM based on the above params...')
    cosmo = FlatLambdaCDM(
        H0=header['hubble'], 
        Om0=header['omega_matter'], 
        )
    snap_lbt = np.array((cosmo.lookback_time(header['redshift']))) / 100.0 # Unit: Gyr
    print('Calculated lookback time for the snapshot: '+str(snap_lbt)[:5]+'Gyr')
    
    print('Start calculating the center...')
    host_center, host_velocity = assign_hosts_coordinates_from_particles( 
        part, header, host_number = host_num
        )
    print('Center located for '+ str(host_num)+'host(s)!')
    print(host_center)
    print(host_velocity)
    print('!!!!!!ABOVE check!!!!!!')
    if host_num != host_center.shape[0]:
        raise ValueError('# of host(s) assigned not consistent with the # of the coords/vel got for the host(s)!')
    
    for ii in range(0,host_num):
        
        print('Start calculation for host'+str(ii))
        part['star']['radius'] = header['scalefactor'] * coord_to_r(
            part['star']['position'], 
            cen_deduct = True, 
            cen_coord = host_center[ii],
            ) # Unit: kpc physical
        part['gas']['radius'] = header['scalefactor'] * coord_to_r(
            part['gas']['position'], 
            cen_deduct = True, 
            cen_coord = host_center[ii],
            ) # Unit: kpc physical
        part['dark']['radius'] = header['scalefactor'] * coord_to_r(
            part['gas']['position'], 
            cen_deduct = True, 
            cen_coord = host_center[ii],
            ) # Unit: kpc physical
        r_vs_M_file = get_r_vs_M(part, r_max = 400.0)
        print('Mass profile calculated!')
        print(r_vs_M_file)
        print('!!!!!!!ABOVE!!!!!!!')
        star_coord_host = header['scalefactor'] * (part['star']['position'] - host_center[ii]) #unit: kpc physical!!!
        star_vel_host = part['star']['velocity'] - host_velocity[ii]
        host_mask = part['star']['radius']<=20.0 #CHANGED THIS LINE FROM 10 TO 20 ON JULY 26, 5:47 PM, grabs stars within 20 kpc
        
        star_jnet_host = calculate_ang_mom(
            part['star']['mass'][host_mask],
            header['scalefactor'] * star_coord_host[host_mask],
            star_vel_host[host_mask],
            )
        r_matrix = cal_rotation_matrix( star_jnet_host ,np.array((0.0,0.0,1.0)))
        print('ROTATION MATRIX???')
        print(r_matrix)
        print('!!!!!!!ABOVE!!!!!!!')
        star_coord_host = rotate_matrix(star_coord_host,r_matrix)
        star_vel_host = rotate_matrix(star_vel_host,r_matrix)

        print('Start calculating Jz/Jc for host'+ str(ii))
        JzJc_host, trash = get_Jz_over_Jc(
            part['star']['radius'][host_mask],
            part['star']['position'][host_mask],
            star_coord_host[host_mask],
            star_vel_host[host_mask],
            r_vs_M_file,
            recal_jnet = False,
            )
        print('Jz/Jc for host'+ str(ii)+' done!')


        print('Start calculating J/Jc for host'+ str(ii))
        JJc_host, trash = get_J_over_Jc(
            part['star']['radius'][host_mask],
            part['star']['position'][host_mask],
            star_coord_host[host_mask],
            star_vel_host[host_mask],
            r_vs_M_file,
            recal_jnet = False,
            )
        print('J/Jc for host'+ str(ii)+' done!')

        print('All calculation done. Start saving data now...')

        print('Select the particles to save based on SFT...')
        sft_lbt = np.array((cosmo.lookback_time( 1.0/part['star']['sft_a'][host_mask] - 1.0 ))) / 100.0
        print('(Only save particles younger than 100Myr)')
        young_mask = ( sft_lbt <= (snap_lbt+0.5) ) #changed .1 to .5, july 31 7:33 pm, only grabs stars w/sft <500Myr
        print('Out of '+ str( host_mask[host_mask].shape[0] ) +' host particles:')
        print('!!!!! # of particles saved: '+ str(young_mask[young_mask].shape[0]) + ' !!!!!' )
        
        fff = os.path.join( 
            save_path, 
            'id_jzjc_jjc_' + snapdir_num + '_host'+str(ii)[0] + '_20kpc_500Myr' + '.hdf5' 
            )
        print('Saving data in:'+fff)
        
        df = h5py.File(fff,'w')
        df.create_dataset('id', data=part['star']['id'][host_mask][young_mask] )
	
        df.create_dataset('sft_a', data=part['star']['sft_a'][host_mask][young_mask])
        df.create_dataset('sft_Gyr', data=sft_lbt[young_mask])

        df.create_dataset('jzjc', data=JzJc_host[young_mask])

        df.create_dataset('jjc', data=JJc_host[young_mask])
        
        df.create_dataset('host_center', data = host_center[ii])
        df.close()




