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
save_path = os.path.join('/data17/grenache/omyrtaj/analysis_data', run_name)
print('***Start the routine for ' + run_name + '***')

#output_path = os.path.join('/data17/grenache/aalazar/FIRE/GVB',run_name,'output/hdf5')
output_path = os.path.join('/data17/grenache/omyrtaj/FIRE/',run_name,'output')
snapdir_paths = get_snapdir_path(output_path)
for snapdir_path in snapdir_paths[-39:]: #romeo_juliet has 39 snapshots on my grenache FIRE directory
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
    print('Calculated redshift for the snapshot: ' + str(header['redshift']))
    print('Calculated lookback time for the snapshot: '+str(snap_lbt)[:5]+'Gyr')
    
    print('Start calculating the center...')
    halo = h5py.File('/data17/grenache/omyrtaj/FIRE/'+run_name+'/halo/rockstar_dm/catalog_hdf5/halo_'+snapdir_num+'.hdf5', 'r')
    
    f_star = h5py.File('/data17/grenache/omyrtaj/FIRE/'+run_name+'/halo/rockstar_dm/catalog_hdf5/star_'+snapdir_num+'.hdf5','r')
    halo_pos = np.array(halo['position'])
    halo_vel = np.array(halo['velocity'])
    host1_ind = np.array(halo['host.index'])[0]
    host2_ind = np.array(halo['host2.index'])[0]
    host_center = [halo_pos[host1_ind], halo_pos[host2_ind]]
    host_velocity = [halo_vel[host1_ind], halo_vel[host2_ind]]
    r90_star_rockstar_arr = np.array(f_star['star.radius.90'])
    # Anna's CoM function
    #host_center, host_velocity = assign_hosts_coordinates_from_particles( 
        #part, header, host_number = host_num
        #)
    
    print('Center located for '+ str(host_num)+'host(s)!')
    print(host_center)
    print(host_velocity)
    print('!!!!!!ABOVE check!!!!!!')
    #if host_num != host_center.shape[0]:
        #raise ValueError('# of host(s) assigned not consistent with the # of the coords/vel got for the host(s)!')
    
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
        v_circ_vs_r = get_circ_vel(r_vs_M_file)
        print('Circular velocity profile calculated!')
        accel_vs_r = get_accel(r_vs_M_file)
        print('Acceleration profile calculated!')
        
        m_vir = np.array(halo['mass.vir'])[host1_ind]
        r90_star_rockstar = r90_star_rockstar_arr[host1_ind]
        if ii != 0:
            m_vir = np.array(halo['mass.vir'])[host2_ind]
            r90_star_rockstar = r90_star_rockstar_arr[host2_ind]
        z = 1/header['scalefactor'] - 1
        r_vir = virial_radius(m_vir, z, header['omega_matter'], header['omega_lambda'], header['hubble'])
        print('Virial radius calculated: ' + str(r_vir))
        r90_mask = part['star']['radius'] < 0.1*r_vir
        r90_star = get_r90(part['star']['radius'][r90_mask],
                               part['star']['mass'][r90_mask],
                               np.sum(part['star']['mass']))
        print('r90_star (using 0.1 r_vir): ', r90_star)
        print('r90_star (from star catalog): ', r90_star_rockstar)
        r_peak_over_r90 = get_r_peak_over_r90(v_circ_vs_r[0], v_circ_vs_r[1], r90_star)
        r_peak_over_r90_rockstar = get_r_peak_over_r90(v_circ_vs_r[0], v_circ_vs_r[1], r90_star_rockstar)
        print('r_peak/r90: ', r_peak_over_r90)
        print('r_peak_over_r90_rockstar: ', r_peak_over_r90_rockstar)
        star_coord_host = header['scalefactor'] * (part['star']['position'] - host_center[ii]) #unit: kpc physical!!!
        star_vel_host = part['star']['velocity'] - host_velocity[ii]
        host_mask = part['star']['radius']<=20.0 #CHANGED THIS LINE FROM 10 TO 20 ON JULY 26, 5:47 PM
        
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

        print('Starting v_phi calculation: ')
        v_dot_phihat = calc_cyl_vels(star_vel_host, star_coord_host)
        
        print('Done with v_dot_phihat calculation: ', v_dot_phihat)
        print('v_dot_phihat shape: ', np.shape(v_dot_phihat))

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
        fe_over_h = fe_over_h_ratios(part['star']['metallicity'][:,0][host_mask], #added may 2
                                     part['star']['metallicity'][:,1][host_mask],
                                     part['star']['metallicity'][:,10][host_mask])

        print('Done calculating [Fe/H]')
        mg_over_fe = mg_over_fe_ratios(part['star']['metallicity'][:,6][host_mask],
                                       part['star']['metallicity'][:,10][host_mask])
        print('Done calculating [Mg/Fe]')
        o_over_fe = o_over_fe_ratios(part['star']['metallicity'][:,4][host_mask],
                                     part['star']['metallicity'][:,10][host_mask])
        print('Done calculating [O/Fe]')
        
        print('All calculation done. Start saving data now...')

        print('Select the particles to save based on SFT...')
        sft_lbt = np.array((cosmo.lookback_time( 1.0/part['star']['sft_a'][host_mask] - 1.0 ))) / 100.0
        print('(Only save particles younger than 500Myr)')
        young_mask = ( sft_lbt <= (snap_lbt+0.5) ) #changed .1 to .5, july 31 7:33 pm
        #print('Out of '+ str( host_mask[host_mask].shape[0] ) +' host particles:')
        #print('!!!!! # of particles saved: '+ str(young_mask[young_mask].shape[0]) + ' !!!!!' )
        
        fff = os.path.join( 
            save_path, 
            'id_jzjc_jjc_' + snapdir_num + '_host'+str(ii)[0] + '_20kpc_rockstar_centers' + '.hdf5' 
            )
        print('Saving data in:'+fff)
        
        df = h5py.File(fff,'w')
        df.create_dataset('r_vs_M_file', data = r_vs_M_file)
        df.create_dataset('id', data=part['star']['id'][host_mask])#[young_mask] )
        df.create_dataset('mass', data = part['star']['mass'][host_mask])
        df.create_dataset('r_vir', data = r_vir)
        df.create_dataset('r90_star', data = r90_star)
        df.create_dataset('r90_star_rockstar', data = r90_star_rockstar)
        df.create_dataset('r_peak_over_r90', data = r_peak_over_r90)
        df.create_dataset('r_peak_over_r90_rockstar', data = r_peak_over_r90_rockstar)
        df.create_dataset('fe_over_h', data=fe_over_h)#[young_mask])
        df.create_dataset('mg_over_fe', data = mg_over_fe)
        df.create_dataset('o_over_fe', data = o_over_fe)
        df.create_dataset('sft_a', data=part['star']['sft_a'][host_mask])#[young_mask])
        df.create_dataset('sft_Gyr', data=sft_lbt)#[young_mask])
        df.create_dataset('r_star', data = part['star']['radius'][host_mask])
        df.create_dataset('jzjc', data=JzJc_host)#[young_mask])
        df.create_dataset('jjc', data=JJc_host)#[young_mask]
        df.create_dataset('host_center', data = host_center[ii])
        df.create_dataset('v_dot_phihat', data = v_dot_phihat[host_mask])
        df.create_dataset('v_circ_vs_r', data = v_circ_vs_r)
        df.create_dataset('accel_vs_r', data = accel_vs_r)       
        df.create_dataset('star_coord_host', data = star_coord_host[host_mask])
        df.create_dataset('star_vel_host', data = star_vel_host[host_mask])
        df.close()




