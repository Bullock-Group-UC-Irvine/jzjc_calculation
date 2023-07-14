# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import collections
import h5py
import numpy as np
import scipy
import sys
from scipy import integrate

#===========================================================================================
#===========================Functions to get center coord and vel===========================

def get_header(filepath):
    header_name_dict = {
            'GIZMO_version': 'gizmo.version',
            # 6-element array of number of particles of each type in file
            'NumPart_ThisFile': 'particle.numbers.in.file',
            # 6-element array of total number of particles of each type (across all files)
            'NumPart_Total': 'particle.numbers.total',
            'NumPart_Total_HighWord': 'particle.numbers.total.high.word',
            # number of file blocks per snapshot
            'NumFilesPerSnapshot': 'file.number.per.snapshot',
            # numerical parameters
            'UnitLength_In_CGS': 'unit.length',
            'UnitMass_In_CGS': 'unit.mass',
            'UnitVelocity_In_CGS': 'unit.velocity',
            'Effective_Kernel_NeighborNumber': 'kernel.number',
            'Fixed_ForceSoftening_Keplerian_Kernel_Extent': 'kernel.sizes',
            'Kernel_Function_ID': 'kernel.id',
            'TurbDiffusion_Coefficient': 'diffusion.coefficient',
            'Solar_Abundances_Adopted': 'sun.massfractions',
            'Metals_Atomic_Number_Or_Key': 'atomic.numbers',
            # mass of each particle species, if all particles are same
            # (= 0 if they are different, which is usually true)
            'MassTable': 'particle.masses',
            # time
            'Time': 'time',  # [scale-factor or Gyr/h in file]
            'Redshift': 'redshift',
            # cosmology
            'BoxSize': 'box.length',  # [kpc/h comoving in file]
            'Omega0': 'omega_matter',  # old name convention
            'OmegaLambda': 'omega_lambda',  # old name convention
            'Omega_Matter': 'omega_matter',
            'Omega_Baryon': 'omega_baryon',
            'Omega_Lambda': 'omega_lambda',
            'Omega_Radiation': 'omega_radiation',
            'HubbleParam': 'hubble',
            'ComovingIntegrationOn': 'cosmological',
            # physics flags
            'Flag_DoublePrecision': 'has.double.precision',
            'Flag_Sfr': 'has.star.formation',
            'Density_Threshold_For_SF_CodeUnits': 'sf.density.threshold',
            'Flag_Cooling': 'has.cooling',
            'Flag_StellarAge': 'has.star.age',
            'Flag_Feedback': 'has.feedback',
            'Flag_IC_Info': 'initial.condition.kind',
            'Flag_Metals': 'element.number',
            'Flag_AgeTracers': 'has.agetracer',  # not in newer headers
            # age-tracers to assign elemental abundances in post-processing
            # will have either agetracer.age.min + agetracer.age.max or agetracer.age.bins
            'AgeTracer_NumberOfBins': 'agetracer.age.bin.number',
            'AgeTracerBinStart': 'agetracer.age.min',
            'AgeTracerBinEnd': 'agetracer.age.max',
            'AgeTracer_CustomTimeBins': 'agetracer.age.bins',  # or this array
            'AgeTracerEventsPerTimeBin': 'agetracer.event.number.per.age.bin',
            # level of compression of snapshot file
            'CompactLevel': 'compression.level',
            'Compactify_Version': 'compression.version',
            'ReadMe': 'compression.readme',
        }
    header = {}

    filename = [file for file in os.listdir( filepath ) if file[-4:] == 'hdf5' ][0]

    with h5py.File( os.path.join(filepath, filename ) , 'r') as file_read:
        header_read = file_read['Header'].attrs  # load header dictionary
        for prop_read_name in header_read:
                if prop_read_name in header_name_dict:
                    prop_save_name = header_name_dict[prop_read_name]
                else:
                    prop_save_name = prop_read_name
                header[prop_save_name] = header_read[prop_read_name]  # transfer to my header dict

    if 'cosmological' in header and header['cosmological']:
            assert 0 < header['hubble'] < 1
            assert 0 < header['omega_matter'] <= 1
            assert 0 < header['omega_lambda'] <= 1
    elif (
            0 < header['hubble'] < 1
            and 0 < header['omega_matter'] <= 1
            and 0 < header['omega_lambda'] <= 1
    ):
            header['cosmological'] = True  # compatible with old file headers
    else:
            header['cosmological'] = False

    if header['cosmological']:
        header['scalefactor'] = float(header['time'])
        del header['time']
        header['box.length/h'] = float(header['box.length'])
        header['box.length'] /= header['hubble']  # convert to [kpc comoving]
    else:
        header['time'] /= header['hubble']  # convert to [Gyr]
        header['scalefactor'] = 1.0
    return header

#from ut/basic/array.py
def parse_int_dtype(array_length, dtype=None):
    '''
    Parse input data type, if None, use default int, according to array size.

    Parameters
    ----------
    array_length : int
        length of array
    dtype
        data type to override default selection

    Returns
    -------
    dtype
        data type
    '''
    if dtype is None:
        if array_length > 2147483647:
            dtype = np.int64
        else:
            dtype = np.int32

    return dtype

#from ut/basic/array.py
def get_arange(array_or_length_or_imin=None, i_max=None, dtype=None):
    '''
    Get np.arange corresponding to input limits or input array size.

    Parameters
    ----------
    array_or_length_or_imin : array or int
        array or array length or starting value
    i_max : int
        ending value (if array_or_length_or_imin is starting value)
    dtype
        data type (if None, use size to determine 32 or 64-bit int)

    Returns
    -------
    array
        array with values 0, 1, 2, 3, etc
    '''
    if i_max is None:
        i_min = 0
        if np.isscalar(array_or_length_or_imin):
            i_max = array_or_length_or_imin
        else:
            i_max = len(array_or_length_or_imin)
    else:
        i_min = array_or_length_or_imin

    dtype = parse_int_dtype(i_max, dtype)

    return np.arange(i_min, i_max, dtype=dtype)

#from ut.particle.py
def parse_indices(part_spec, part_indicess, center_index=None):
    '''
    Parse input list of particle indices.
    If none, generate via arange.

    Parameters
    ----------
    part_spec : dict
        catalog of particles of given species
    part_indices : array or list of arrays
        indices of particles
    center_index : int
        index of center/host position, to select from part_indicess (if list)

    Returns
    -------
    part_indices : array
        indices of particles (for single center/host)
    '''
    if part_indicess is None or len(part_indicess) == 0:
        # input null, so get indices of all particles via catalog
        if 'position' in part_spec:
            part_indices = get_arange(part_spec['position'].shape[0])
        elif 'mass' in part_spec:
            part_indices = get_arange(part_spec['mass'].size)
        else:
            raise ValueError('cannot determine particle indices array')
    else:
        assert len(part_indicess) > 0
        if not np.isscalar(part_indicess[0]):
            # input array of particle indices for each center/host
            part_indices = part_indicess[center_index]
        else:
            part_indices = part_indicess

    return part_indices

#from ut/basic/coordinate.py
def get_position_differences(position_difs, periodic_length=None):
    '''
    Get distance / separation vector, in range [-periodic_length/2, periodic_length/2).

    Parameters
    ----------
    position_difs : array
        position difference[s]
    periodic_length : float
        periodicity length (if none, return array as is)
    '''
    if not periodic_length:
        return position_difs
    else:
        if np.isscalar(periodic_length) and periodic_length <= 1:
            print(f'! got unusual periodic_length = {periodic_length}')

    if np.isscalar(position_difs):
        if position_difs >= 0.5 * periodic_length:
            position_difs -= periodic_length
        elif position_difs < -0.5 * periodic_length:
            position_difs += periodic_length
    else:
        position_difs[position_difs >= 0.5 * periodic_length] -= periodic_length
        position_difs[position_difs < -0.5 * periodic_length] += periodic_length

    return position_difs

#from ut/basic/coordinate.py
def get_center_position(
    positions,
    weights=None,
    periodic_length=None,
    position_number_min=32,
    center_position=None,
    distance_max=np.Inf,
):
    '''
    Get position of center of mass, using iterative zoom-in.

    Parameters
    ----------
    positions : array (particle number x dimension number)
        position[s]
    weights : array
        weight for each position (usually mass) - if None, assume all have same weight
    periodic_length : float
        periodic box length
    position_number_min : int
        minimum number of positions within distance to keep zooming in
    center_position : array
        initial center position to use
    distance_max : float
        maximum distance to consider initially

    Returns
    -------
    center_position : array
        position vector of center of mass
    '''
    distance_bins = np.array(
        [
            np.Inf,
            1000,
            700,
            500,
            300,
            200,
            150,
            100,
            70,
            50,
            30,
            20,
            15,
            10,
            7,
            5,
            3,
            2,
            1.5,
            1,
            0.7,
            0.5,
            0.3,
            0.2,
            0.15,
            0.1,
            0.07,
            0.05,
            0.03,
            0.02,
            0.015,
            0.01,
            0.007,
            0.005,
            0.003,
            0.002,
            0.0015,
            0.001,
        ]
    )
    distance_bins = distance_bins[distance_bins <= distance_max]

    if weights is not None:
        assert positions.shape[0] == weights.size
        # normalize weights by median, improves numerical stability
        weights = np.asarray(weights) / np.median(weights)

    if center_position is None or len(center_position) == 0:
        center_position = np.zeros(positions.shape[1], positions.dtype)
    else:
        center_position = np.array(center_position, positions.dtype)

    if positions.shape[0] > 2147483647:
        idtype = np.int64
    else:
        idtype = np.int32
    part_indices = np.arange(positions.shape[0], dtype=idtype)

    for dist_i, dist_max in enumerate(distance_bins):
        # direct method ----------
        distance2s = (
            get_position_differences(positions[part_indices] - center_position, periodic_length)
            ** 2
        )
        distance2s = np.sum(distance2s, 1)

        # get particles within distance max
        masks = distance2s < dist_max ** 2
        part_indices_dist = part_indices[masks]

        # store particles slightly beyond distance max for next interation
        masks = distance2s < (1.5 * dist_max) ** 2
        part_indices = part_indices[masks]

        # kd-tree method ----------
        # if dist_i == 0:
        # create tree if this is the first distance bin
        #    KDTree = spatial.KDTree(positions, boxsize=part.info['box.length'])
        #    particle_number_max = positions.shape[0]

        # distances, indices = KDTree.query(
        #    center_position, particle_number_max, distance_upper_bound=dist_max, workers=2)

        # masks = (distances < dist_max)
        # part_indices_dist = indices[masks]
        # particle_number_max = part_indices_dist.size

        # check whether reached minimum total number of particles within distance
        # but force at least one loop over distance bins to get *a* center
        if part_indices_dist.size <= position_number_min and dist_i > 0:
            return center_position

        if weights is None:
            weights_use = weights
        else:
            weights_use = weights[part_indices_dist]

        # ensure that np.average uses 64-bit internally for accuracy, but returns as input dtype
        center_position = np.average(
            positions[part_indices_dist].astype(np.float64), 0, weights_use
        ).astype(positions.dtype)

    return center_position

#from ut.particle.py
def get_position_differences(position_difs, periodic_length=None):
    '''
    Get distance / separation vector, in range [-periodic_length/2, periodic_length/2).

    Parameters
    ----------
    position_difs : array
        position difference[s]
    periodic_length : float
        periodicity length (if none, return array as is)
    '''
    if not periodic_length:
        return position_difs
    else:
        if np.isscalar(periodic_length) and periodic_length <= 1:
            print(f'! got unusual periodic_length = {periodic_length}')

    if np.isscalar(position_difs):
        if position_difs >= 0.5 * periodic_length:
            position_difs -= periodic_length
        elif position_difs < -0.5 * periodic_length:
            position_difs += periodic_length
    else:
        position_difs[position_difs >= 0.5 * periodic_length] -= periodic_length
        position_difs[position_difs < -0.5 * periodic_length] += periodic_length

    return position_difs

#from ut.particle.py
def get_distances(
    positions_1=None, positions_2=None, periodic_length=None, scalefactor=None, total_distance=False
):
    '''
    Get vector or total/scalar distance[s] between input position vectors.
    If input scale-factors, will convert distance from comoving to physical.

    Parameters
    ----------
    positions_1 : array
        position[s]
    positions_2 : array
        position[s]
    periodic_length : float
        periodic length (if none, not use periodic)
    scalefactor : float or array
        expansion scale-factor (to convert comoving to physical)
    total : bool
        whether to compute total/scalar (instead of vector) distance

    Returns
    -------
    distances : array (object number x dimension number, or object number)
        vector or total/scalar distance[s]
    '''
    if not isinstance(positions_1, np.ndarray):
        positions_1 = np.array(positions_1)
    if not isinstance(positions_2, np.ndarray):
        positions_2 = np.array(positions_2)

    if len(positions_1.shape) == 1 and len(positions_2.shape) == 1:
        shape_pos = 0
    else:
        shape_pos = 1

    distances = get_position_differences(positions_1 - positions_2, periodic_length)

    if total_distance:
        distances = np.sqrt(np.sum(distances ** 2, shape_pos))

    if scalefactor is not None:
        if scalefactor > (1 + 1e-4) or scalefactor <= 0:
            print(f'! got unusual scalefactor = {scalefactor}')
        distances *= scalefactor

    return distances

#from ut.particle.py
def get_center_velocity(
    velocities,
    weights=None,
    positions=None,
    center_position=None,
    distance_max=20,
    periodic_length=None,
):
    '''
    Get velocity of center of mass.
    If no input masses, assume all masses are the same.

    Parameters
    ----------
    velocities : array (particle number x dimension_number)
        velocity[s]
    weights : array
        weight for each position (usually mass) - if None, assume all have same weight
    positions : array (particle number x dimension number)
        positions, if want to select by this
    center_position : array
        center position, if want to select by this
    distance_max : float
        maximum position difference from center to use particles
    periodic_length : float
        periodic box length

    Returns
    -------
    center_velocity : array
        velocity vector of center of mass
    '''
    masks = np.full(velocities.shape[0], True)

    # ensure that use only finite values
    for dimen_i in range(velocities.shape[1]):
        masks *= np.isfinite(velocities[:, dimen_i])

    if positions is not None and center_position is not None and len(center_position) > 0:
        assert velocities.shape == positions.shape
        distance2s = np.sum(
            get_position_differences(positions - center_position, periodic_length) ** 2, 1
        )
        masks *= distance2s < distance_max ** 2

    if weights is not None:
        assert velocities.shape[0] == weights.size
        # normalizing weights by median seems to improve numerical stability
        weights = weights[masks] / np.median(weights[masks])

    if not masks.any():
        print('! cannot compute host/center velocity')
        print('  no positions within distance_max = {:.3f} kpc comoving'.format(distance_max))
        print('  nearest = {:.3f} kpc comoving'.format(np.sqrt(distance2s.min())))
        return np.r_[np.nan, np.nan, np.nan]

    # ensure that np.average uses 64-bit internally for accuracy, but returns as input dtype
    return np.average(velocities[masks].astype(np.float64), 0, weights).astype(velocities.dtype)

#from ut.particle.py
def get_distances_wrt_center(
    part,
    header,
    part_indicess,
    center_position,
    species=['star'],
    host_index=0,
    coordinate_system='cartesian',
    return_single_array=True,
):
    '''
    Get distances (scalar or vector) between input particle species positions and center_position
    (input or stored in particle catalog).

    Parameters
    ----------
    part : dict
        catalog of particles at snapshot
    species : str or list
        name[s] of particle species to compute
    part_indicess : array or list
        indices[s] of particles to compute, one array per input species
    center_position : array
        position of center [kpc comoving]
        if None, will use default center position in particle catalog
    rotation : bool or array
        whether to rotate particles. two options:
        (a) if input array of eigen-vectors, will define rotation axes for all species
        (b) if true, will rotate to align with principal axes defined by input species
    host_index : int
        index of host to get stored position of (if not input center_position)
    coordinate_system : str
        which coordinates to get distances in: 'cartesian' (default), 'cylindrical', 'spherical'
    total_distance : bool
        whether to compute total/scalar distance
    return_single_array : bool
        whether to return single array (instead of dict) if input single species

    Returns
    -------
    dist : array (object number x dimension number) or dict thereof
        [kpc physical]
        3-D distance vectors aligned with default x,y,z axes OR
        3-D distance vectors aligned with major, medium, minor axis OR
        2-D distance vectors along major axes and along minor axis OR
        1-D scalar distances
    OR
    dictionary of above for each species
    '''
    assert coordinate_system in ('cartesian', 'cylindrical', 'spherical')

    rotation=None
    total_distance=True

    # if center_position == None:
    #     raise ValueError('center_position not provided in get_distances_wrt_center()!')
    # #center_position = parse_property(part, 'position', center_position, host_index)
    # if part_indicess == None:
    #     raise ValueError('part_indicess not provided in get_distances_wrt_center()!')
    # #part_indicess = parse_property(species, 'indices', part_indicess)

    dist = {}

    for spec_i, spec_name in enumerate(species):
        if len(species) == 1:
            part_indices = parse_indices(part[spec_name], part_indicess)
        else:
            raise ValueError('cannot get more than one species in get_distances_wrt_center() QAQ')

        dist[spec_name] = get_distances(
            part[spec_name]['position'][part_indices],
            center_position,
            header['box.length'],
            header['scalefactor'],
            total_distance,
        )  # [kpc physical]

        if not total_distance:
            raise ValueError('total_distance set to False in get_distances_wrt_center()!')

    if return_single_array and len(species) == 1:
        dist = dist[species[0]]

    return dist


#from ut.particle.py
def get_center_positions(
    part,
    header,
    species_name='star',
    part_indicess=None,
    weight_property='mass',
    center_number=1,
    exclusion_distance=300,
    center_positions=None,
    distance_max=np.Inf,
    return_single_array=False,
):
    '''
    Get host/center position[s] [kpc comoving] via iterative zoom-in on input particle species,
    weighting particle positions by input weight_property.

    Parameters
    ----------
    part : dict
        dictionary of particles
    species : str
        typically 'star' or 'dark'
    part_indicess : array or list of arrays
        indices of particles to use to compute center position[s]
        if a list, use different particles indices for different centers
    weight_property : str
        property to weight particles by: 'mass'(, 'potential', 'massfraction.metals')
    center_number : int
        number of centers (hosts) to compute
    exclusion_distance : float
        radius around previous center to cut out particles for finding next center [kpc comoving]
    center_positions : array or list of arrays
        initial position[s] to center on
    distance_max : float
        maximum distance around center_positions to use to select particles
    return_single_array : bool
        whether to return single array instead of array of arrays, if center_number = 1

    Returns
    -------
    center_positions : array or array of arrays
        position[s] of center[s] [kpc comoving]
    '''

    part_spec = part[species_name]

    if weight_property != 'mass':
        raise ValueError('Not using particle mass as weight in get_center_positions()!')
    if weight_property not in part_spec:
        raise ValueError('Particle mass not provided in get_center_positions()!')


    center_positions = [center_positions for _ in range(center_number)]
    
    for center_i, center_position in enumerate(center_positions):
        part_indices = parse_indices(part_spec, part_indicess, center_i)

        if center_i > 0 and exclusion_distance is not None and exclusion_distance > 0:
            # cull out particles near previous center
            distances = get_distances_wrt_center(
                part,
                header,
                parse_indices(part_spec, part_indicess, center_i - 1),
                center_positions[center_i - 1],
                return_single_array=True,
            )
            # exclusion distance in [kpc comoving]
            masks = distances > (exclusion_distance * header['scalefactor'])
            part_indices = part_indices[masks]

        center_positions[center_i] = get_center_position(
            part_spec['position'][part_indices],
            part_spec['mass'][part_indices],
            header['box.length'],
            center_position=center_position,
            distance_max=distance_max,
        )

    center_positions = np.array(center_positions)


    if return_single_array and center_number == 1:
        center_positions = center_positions[0]

    return center_positions

#from ut.particle.py
def get_center_velocities(
    part,
    header,
    center_positions,
    species_name='star',
    part_indicess=None,
    weight_property='mass',
    distance_max=10,
    return_single_array=False,
):
    '''
    Get host/center velocity[s] [km / s] of input particle species that are within distance_max of
    center_positions, weighting particle velocities by input weight_property.
    If input multiple center_positions, compute a center velocity for each one.

    Parameters
    ----------
    part : dict
        dictionary of particles
    species_name : str
        name of particle species to use
    part_indicess : array or list of arrays
        indices of particles to use to define center
        use this to exclude particles that you know are not relevant
        if list, use host_index to determine which list element to use
    weight_property : str
        property to weight particles by: 'mass', 'potential', 'massfraction.metals'
    distance_max : float
        maximum radius to consider [kpc physical]
    center_positions : array or list of arrays
        center position[s] [kpc comoving]
        if None, will use default center position[s] in catalog
        if list, compute a center velocity for each center position
    return_single_array : bool
        whether to return single array instead of array of arrays, if input single center position
    verbose : bool
        flag for verbosity in print diagnostics

    Returns
    -------
    center_velocities : array or array of arrays
        velocity[s] of center[s] [km / s]
    '''

    part_spec = part[species_name]

    if weight_property != 'mass':
        raise ValueError('Not using particle mass as weight in get_center_velocities()!')
    if weight_property not in part_spec:
        raise ValueError('Particle mass not provided in get_center_velocities()!')

    if center_positions is None:
        raise ValueError('Galaxies centers not provided in get_center_velocities()!')
    #center_positions = parse_property(part_spec, 'position', center_positions)

    distance_max /= header['scalefactor']  # convert to [kpc comoving] to match positions

    center_velocities = np.zeros(center_positions.shape, part_spec['velocity'].dtype)


    for center_i, center_position in enumerate(center_positions):
        part_indices = parse_indices(part_spec, part_indicess, center_i)

        center_velocities[center_i] = get_center_velocity(
            part_spec['velocity'][part_indices],
            part_spec['mass'][part_indices],
            part_spec['position'][part_indices],
            center_position,
            distance_max,
            header['box.length'],
        )

    if return_single_array and len(center_velocities) == 1:
        center_velocities = center_velocities[0]

    return center_velocities

#from myself
def assign_hosts_coordinates_from_particles(
    part,
    header,
    species_name = 'star',
    method='mass',
    velocity_distance_max=10,
    host_number=1,
    exclusion_distance=300,
):
    host_center = get_center_positions(
        part,
        header,
        weight_property=method,
        center_number=host_number,
        exclusion_distance=exclusion_distance,
    )
    host_velocity = get_center_velocities(
        part,
        header,
        host_center,
        weight_property=method,
        distance_max = velocity_distance_max
    )
    return host_center, host_velocity

#===========================================================================================
#===========================================================================================

#===========================================================================================
#===========================Functions to Rotate the system to z-axis===========================

def coord_to_r(coord, cen_deduct = False, cen_coord = np.zeros(3)):
    
    # Calculate distance given coordinates;
    # set cen_deduct = True to calculate distance to this one center coordinate,
    # otherwise, the center is set to (0,0,0) by default  
    if (len(coord.shape) == 1): 
        return np.sqrt(np.sum(np.square(coord-cen_coord)))
    elif (coord.shape[1]==3):
        return np.sqrt(np.sum(np.square(coord-cen_coord),axis=1))
    else:
        return np.sqrt(np.sum(np.square(coord.T-cen_coord),axis=1))

def cal_vr_vt(coord, vel):
    #Calculate radial velcity and tangential velocity

    if coord.shape!=vel.shape:
        raise ValueError('Coordinates shape does not match velocity shape!')
        return np.nan, np.nan
    else:
        if (coord.shape[1]==3):
            vr = np.sum(coord*vel, axis = 1)/coord_to_r(coord)
        else:
            vr = np.sum(coord*vel, axis = 0)/coord_to_r(coord)
        vt = np.sqrt( np.square(coord_to_r(vel)) - np.square(vr) )
        return vr, vt


def calculate_ang_mom(mass,coord,vel):

    # Calculate angular momentum given mass, coordinates and velocities
    if coord.shape[1]!=3:
        coord = coord.T
        vel = vel.T
    mom = np.zeros((coord.shape[0],3))
    mom[:,0] = mass * ( (coord[:,1]*vel[:,2]) - (coord[:,2]*vel[:,1]) )
    mom[:,1] = mass * ( (coord[:,2]*vel[:,0]) - (coord[:,0]*vel[:,2]) )
    mom[:,2] = mass * ( (coord[:,0]*vel[:,1]) - (coord[:,1]*vel[:,0]) )

    mom1 = np.sum(mom[:,0])
    mom2 = np.sum(mom[:,1])
    mom3 = np.sum(mom[:,2])

    return np.array((mom1,mom2,mom3))/np.sum(mass)

def norm_vec(v):

    # Normalize a 1D vector

    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm

def vrrotvec(a,b):

    # Calculate to rotation vector that can rotate vector a to vector b

    an = norm_vec(a)
    bn = norm_vec(b)
    axb = np.cross(an,bn)
    ac = np.arccos(np.dot(an,bn))
    return np.append(axb,ac)

def vrrotvec2mat(r):

    # Convert rotation vector r to rotation matrix

    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1-c

    n = norm_vec(r[0:3])

    x = n[0]
    y = n[1]
    z = n[2]
    m = np.array( ((t*x*x + c, t*x*y - s*z, t*x*z + s*y),\
        (t*x*y + s*z, t*y*y + c, t*y*z - s*x),\
        (t*x*z - s*y, t*y*z + s*x, t*z*z + c)) )
    return m

def cal_rotation_matrix(a,b):

    # Calculate rotation matrix that can rotate vector a to vector b

    return vrrotvec2mat(vrrotvec(a,b))

def rotate_matrix(data,r):

    # Rotate data with rotation matrix r

    if data.shape[1]!=3:
        return np.dot(r,data)
    else:
        return np.dot(r,data.T).T

def rotate_2axis(data,a,b):

    # Rotate data in such a way that can rotate from vector a to vector b

    r = rotation_matrix(a,b)
    if data.shape[1]!=3:
        return np.dot(r,data)
    else:
        return np.dot(r,data.T).T

#===========================================================================================
#===========================================================================================

#===========================================================================================
#===========================Functions to get Jz/Jc and J/Jc===========================

def get_r_vs_M(part, r_max = None, numpoints = 50000, save_file = None):
    
    #returns r vs M(<r)
    
    try:  
        r_star, r_gas, r_dark = part['star']['radius'], part['gas']['radius'], part['dark']['radius']

        m_star, m_gas, m_dark = part['star']['mass'], part['gas']['mass'], part['dark']['mass']
    
        if r_max is None:
            r_max = 20.0
    
        star_indices, gas_indices, dark_indices = np.argsort(r_star), \
            np.argsort(r_gas), np.argsort(r_dark)
    
        r_star, r_gas, r_dark = r_star[star_indices], r_gas[gas_indices], r_dark[dark_indices]
    
        star_mass_cum, gas_mass_cum, dark_mass_cum = np.cumsum(m_star[star_indices]), \
            np.cumsum(m_gas[gas_indices]), np.cumsum(m_dark[dark_indices]),
    
        # interpolate all of these to the same radii. Don't want to break at 0, so..
        r_grid = np.linspace(0.01, r_max, numpoints)
        Mstar_cum, Mgas_cum, Mdark_cum = np.interp(r_grid, r_star, star_mass_cum), \
            np.interp(r_grid, r_gas, gas_mass_cum), np.interp(r_grid, r_dark, dark_mass_cum)
    
        Mtot_cum = Mstar_cum + Mgas_cum + Mdark_cum
        
    except KeyError:
        # no gas
        r_star, r_dark = part['star']['radius'], part['dark']['radius']

        m_star, m_dark = part['star']['mass'], part['dark']['mass']
    
        if r_max is None:
            r_max = 20.0

        star_indices, dark_indices = np.argsort(r_star), np.argsort(r_dark)
    
        r_star, r_dark = r_star[star_indices], r_dark[dark_indices]
    
        star_mass_cum, dark_mass_cum = np.cumsum(m_star[star_indices]), np.cumsum(m_dark[dark_indices])
    
        # interpolate all of these to the same radii. Don't want to break at 0, so..
        r_grid = np.linspace(0.01, r_max, numpoints)
        Mstar_cum, Mdark_cum = np.interp(r_grid, r_star, star_mass_cum), np.interp(r_grid, r_dark, dark_mass_cum)
        Mtot_cum = Mstar_cum + Mdark_cum
    
    if save_file is not None:
        np.save(save_file, np.vstack((r_grid,Mtot_cum)))
    return np.vstack((r_grid,Mtot_cum))

def get_functional_phi_tck(r_vs_M_file, return_M = False):
    '''
    Creates the kernel to be called by get_functional_phi_splev
    '''
    from scipy.interpolate import splrep
    from scipy.integrate import cumtrapz
    
    G = 4.302e-6 # [kpc M_sun^-1 km^2 s^-2]
    
    # solve Poisson's equation.
    if type(r_vs_M_file) is str:
        r_vs_M = np.load(r_vs_M_file) 
    else:
        r_vs_M = r_vs_M_file
    r_grid, Mtot_cum = r_vs_M[0,:], r_vs_M[1,:]

    #phi = G*cumtrapz(Mtot_cum/r_grid**2, x = r_grid)
    A = G*cumtrapz(Mtot_cum/r_grid**2, x = r_grid)
    B = A[-1]
    r = r_grid[:-1] + 0.5*np.diff(r_grid)
    phi = A - B
    phi -= G*Mtot_cum[-1]/r_grid[-1]
    
    # only defined up to an integration constant. we know that at r = r_max,
    # the potential is GM/r_max, so 
    # phi -= (np.max(phi) + G*Mtot_cum[-1]/r_grid[-1])    
    func_phi = splrep(r, phi)
        
    
    if return_M:
        return func_phi, r_grid, Mtot_cum
    else:
        return func_phi

def get_functional_phi_splev(r_vs_M_file, radii, return_M = False):
    '''
    Faster than other version; uses spline instead of interp1d
    '''
    from scipy.interpolate import splev
    
    if return_M:
        tck, r_grid, M_tot = get_functional_phi_tck(r_vs_M_file, return_M = return_M)
    else:
        tck = get_functional_phi_tck(r_vs_M_file)
    phi = splev(radii, tck)
    if return_M:
        return phi, r_grid, M_tot
    else:
        return phi

def get_specific_energies(radii, mass, coord, vel, r_vs_M_file):

    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
        
    T = 1/2*(vx**2 + vy**2 + vz**2)
    
    phi, r_grid, M_tot = get_functional_phi_splev(r_vs_M_file, radii = radii, 
            return_M = True)
    phi_grid = get_functional_phi_splev(r_vs_M_file, radii = r_grid)
    
    E = phi + T
    
    return E, phi_grid, r_grid, M_tot

def get_Jc_of_E(radii, mass, coord, vel, r_vs_M_file):

    G = 4.302e-6 # [kpc M_sun^-1 km^2 s^-2]

    E, phi_grid, r_grid, M_tot = get_specific_energies(radii, mass, coord, vel, r_vs_M_file)

    E_grid = G*M_tot/(2*r_grid) + phi_grid
    circ_r = np.interp(E, E_grid, r_grid)
    
    this_M = np.interp(circ_r, r_grid, M_tot)
    
    J = np.sqrt(G*this_M*circ_r)
    return J

def get_specific_momentum( mass , coord, vel ):
  
    R = coord
        
    V = vel
    masses = np.ones(len(V))
    P = np.multiply(masses, vel.T).T # elementwise multiplication is inefficient
    L = np.cross(coord, P)

    # in units of M_sun*km*kpc/s
    return L

def get_Jz_over_Jc( radii, mass, coord, vel, r_vs_M_file, recal_jnet = False):


    print('radii in jz/jc function: :', np.sort(radii)[::-1],
	'max of radii in jz/jc function',np.max(radii))
    Ls = get_specific_momentum( mass , coord, vel )
    M = np.sum(mass)

    if recal_jnet:
        j_net = np.sum(Ls, axis = 0)/M
        normed_j_net = j_net/np.linalg.norm(j_net)
    else:
        normed_j_net = np.array((0.0,0.0,1.0))
    Jz = np.dot(Ls, normed_j_net)
    Jc = get_Jc_of_E(radii, mass, coord, vel, r_vs_M_file)
    return Jz/Jc, normed_j_net

def get_J_over_Jc( radii, mass, coord, vel, r_vs_M_file, recal_jnet = False):


    Ls = get_specific_momentum( mass , coord, vel )
    M = np.sum(mass)

    if recal_jnet:
        j_net = np.sum(Ls, axis = 0)/M
        normed_j_net = j_net/np.linalg.norm(j_net)
    else:
        normed_j_net = np.array((0.0,0.0,1.0))
    #Jz = np.dot(Ls, normed_j_net)
    Jc = get_Jc_of_E(radii, mass, coord, vel, r_vs_M_file)
    return np.sqrt(np.sum(np.square(Ls),axis=1))/Jc, normed_j_net


#===========================================================================================
#===========================================================================================


#===========================================================================================
#===========================Functions to get basic data LUL===========================
def get_snapdir_path(output_path, snapshot = None):
    #output_path: path to the "output/" folder
    if snapshot is None:
        #if snapshot not specified, return the list of all the snapdir_* path
        snapdir_name = [snap for snap in os.listdir( output_path ) if snap[:8]== 'snapdir_' ]
        snapdir_num = [int(snap_num[-3:]) for snap_num in snapdir_name ]
        #sort the snapdir according to the final 3 digit
        sort_ii = np.argsort(snapdir_num)
        #return the sorted list of full path to .hdf5 file
        return [os.path.join(output_path,snapdir_name[ii]) for ii in sort_ii]
    else:
        snapdir_name = os.path.join(output_path, 'snapdir_'+snapshot )
        return snapdir_name


def get_data(filepath,key1,key2):
    #filepath: pointing to the folder where there are .hdf5 files
    #make sure you are reading the hdf5 file
    filenames = [file for file in os.listdir( filepath ) if file[-4:] == 'hdf5' ]
    #filenames = os.listdir( filepath )
    num_of_file = len( filenames )
    if num_of_file == 1:
        f = h5py.File( os.path.join(filepath,filenames[0]) , 'r')
        return f[key1][key2][:]
    else:
        for i in range(0,num_of_file):
            f = h5py.File( os.path.join(filepath,filenames[i]) , 'r')
            if ( len(f[key1][key2][:].shape)==1 ):
                if i==0:
                    result = f[key1][key2][:]
                else:
                    result = np.hstack( (result,f[key1][key2][:]) )
            else:
                if i==0:
                    result = f[key1][key2][:]
                else:
                    result = np.vstack( (result,f[key1][key2][:]) )
        return result

def convert_coord_comoving(coord, h):
    return coord / h

def convert_mass(mass,h):
    return mass * np.power(10,10) / h

def convert_velocity(velocity, a):
    return velocity * np.sqrt(a)

def read_part(snapdir_path):
    header = get_header( snapdir_path )
    part = {}
    part['star'] = {}
    part['star']['position'] = convert_coord_comoving(
        get_data(snapdir_path,'PartType4','Coordinates') ,
        header['hubble'],
        ) #Unit: kpc comoving

    part['star']['mass'] = convert_mass( 
        get_data(snapdir_path,'PartType4','Masses') ,
        header['hubble'],
        ) #Unit: Msun
    part['star']['velocity'] = convert_velocity( 
        get_data(snapdir_path,'PartType4','Velocities') ,
        header['scalefactor'],
        ) #Unit: km/s
    part['star']['id'] = get_data(snapdir_path, 'PartType4', 'ParticleIDs')
    part['star']['sft_a'] = get_data(snapdir_path, 'PartType4', 'StellarFormationTime')

    part['gas'] = {}
    part['gas']['position'] = convert_coord_comoving(
        get_data(snapdir_path,'PartType0','Coordinates') ,
        header['hubble'],
        ) #Unit: kpc comoving
    part['gas']['mass'] = convert_mass( 
        get_data(snapdir_path,'PartType0','Masses') ,
        header['hubble'],
        ) #Unit: Msun
    part['gas']['velocity'] = convert_velocity( 
        get_data(snapdir_path,'PartType0','Velocities') ,
        header['scalefactor'],
        ) #Unit: km/s

    part['dark'] = {}
    part['dark']['position'] = convert_coord_comoving(
        get_data(snapdir_path,'PartType1','Coordinates') ,
        header['hubble'],
        ) #Unit: kpc comoving
    part['dark']['mass'] = convert_mass( 
        get_data(snapdir_path,'PartType1','Masses') ,
        header['hubble'],
        ) #Unit: Msun
    part['dark']['velocity'] = convert_velocity( 
        get_data(snapdir_path,'PartType1','Velocities') ,
        header['scalefactor'],
        ) #Unit: km/s
    return part, header
