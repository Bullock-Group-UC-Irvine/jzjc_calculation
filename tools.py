def get_fname(galname, snapdir_num, max_age):
    import paths
    import os
    from UCI_tools import staudt_tools

    df = staudt_tools.init_df()
    # ELVIS alaxies whose host_key in df is 'host.index' ('host2.index') are
    # the more (less) massive of the pair. In the
    # jzjc project, their `host_index` is 0 (1).
    host_index_dict = {'host.index': 0, 'host2.index': 1}
    host_index = host_index_dict[df.loc[galname, 'host_key']]

    if max_age is not None:
        # Max stellar age string to use in the filename:
        max_age_str = '_{0:0.0f}Myr'.format(max_age * 1.e3)
    else:
        # Don't add any max stellar age info to the filename:
        max_age_str = ''

    run_name = 'm12' + df.loc[galname, 'fsuffix'] \
               + '_res' + df.loc[galname, 'res']
    save_path = os.path.join(paths.data, run_name)
    fname = os.path.join(
                save_path, 
                'id_jzjc_jjc_' \
                    + str(snapdir_num) + '_host' + str(host_index)[0] \
                    + '_20kpc' + max_age_str \
                    + '.hdf5'
            )

    return fname

def calc_gal_fracs(galname, thin_jzjc, thick_jzjc, 
                  snapdir_num=600, max_age=None):
    import h5py
    import scipy
    import numpy as np

    fname = get_fname(galname, snapdir_num, max_age)
    
    with h5py.File(fname, 'r') as f:
        masses = f['masses'][:]
        jzjcs = f['jzjc'][:]

    M = masses.sum()
    in_thin = jzjcs >= thin_jzjc 
    thin_frac = masses[in_thin].sum() / M
    in_thick = jzjcs >= thick_jzjc 
    thick_frac = masses[in_thick].sum() / M

    return thin_frac, thick_frac

def calc_all_fracs(snapdir_num=600, max_age=None):
    import os
    import pandas as pd
    from UCI_tools import staudt_tools

    df = staudt_tools.init_df()
    for galname in df.index:
        fname = get_fname(galname, snapdir_num, max_age)
        if os.path.isfile(fname):
            fracs = calc_gal_fracs(
                        galname,
                        thin_jzjc=0.8,
                        thick_jzjc=0.2,
                        snapdir_num=snapdir_num,
                        max_age=max_age
            )
            df.loc[galname, 'thin_frac'] = fracs[0]
            df.loc[galname, 'thick_frac'] = fracs[1]

    # For now, add a comparison to Anna
    annas_fracs = {
            'Romeo': 0.45, 
            'm12b': 0.37,
            'Remus': 0.36,
            'Louise': 0.32,
            'm12f': 0.38,
            'Romulus': 0.37,
            'Juliet': 0.3,
            'm12m': 0.34,
            'm12c': 0.32,
            'm12i': 0.32,
            'Thelma': 0.27,
            'm12w': 0.24
    }
    df_anna = pd.DataFrame.from_dict(
                  annas_fracs, 
                  orient='index', 
                  columns=['anna\'s frac']
              )
    df = pd.concat([df, df_anna], axis=1)

    # Reorder the columns
    cols = ['thick_frac', 'thin_frac', 'anna\'s frac']
    last_cols = [col for col in df.columns if col not in cols]
    cols.extend(last_cols)
    df = df[cols]

    df.sort_values('thin_frac', inplace=True) # Sort by thin frac.

    return df
