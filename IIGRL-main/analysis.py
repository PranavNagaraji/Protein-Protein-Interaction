import atom3.database as db
import numpy as np
import logging
import pickle
import os
import pdb

def get_residue_group(df):
    res_group = df[['pdb_name', 'model', 'chain', 'residue']]

    res_group = res_group.reset_index(drop=True)

    res_group = res_group.drop_duplicates(subset=['chain', 'residue'],keep='first')

    res_group = res_group.reset_index(drop=True)
    
    return res_group

# the max value is 15
# different from alphafold2(which is 14)
def max_atom_num_in_residue():
    print("reading pkl dataset")
    work_filenames = db.get_structures_filenames("/opt/data/private/protein/DIPS-Plus/project/datasets/DB5/interim/parsed", extension='.pkl')
    max_atom_num = -1
    num = 1
    for prot in work_filenames:
        if '_b_' in prot:
            continue
        with open(prot, 'rb') as f:
            df = pickle.load(f)
        print("reading {:} protein {:}".format(num, db.get_pdb_name(prot, with_type=False)))
        heavy = df[df['element'] != 'H']
        df_res = get_residue_group(heavy)

        df = df.reset_index().set_index(['pdb_name', 'model', 'chain', 'residue'])
        df = df.sort_index()

        for i, res_row in df_res.iterrows():
            atom_num = df.loc[tuple(res_row)].shape[0]
            if  atom_num> max_atom_num:
                max_atom_num = atom_num
                print(res_row)
        num += 1
        
        
    print("max atom's num in one residue is {:}".format(max_atom_num))

# max atom's X pos is 336.4729919433594
# min atom's X pos is -187.5989990234375
# max atom's Y pos is 271.1570129394531
# min atom's Y pos is -133.28399658203125
# max atom's Z pos is 327.885986328125
# min atom's Z pos is -217.19700622558594
def max_min_pos():
    print("reading pkl dataset")
    work_filenames = db.get_structures_filenames("/opt/data/private/protein/DIPS-Plus/project/datasets/DB5/interim/parsed", extension='.pkl')
    max_pos_X = -999999
    min_pos_X = 999999
    max_pos_Y = -999999
    min_pos_Y = 999999
    max_pos_Z = -999999
    min_pos_Z = 999999
    num = 1
    for prot in work_filenames:
        if '_b_' in prot:
            continue
        with open(prot, 'rb') as f:
            df = pickle.load(f)
        print("reading {:} protein {:}".format(num, db.get_pdb_name(prot, with_type=False)))
        all_atom_pos_X = np.array(df[df['element'] != 'H']['x'])
        all_atom_pos_Y = np.array(df[df['element'] != 'H']['y'])
        all_atom_pos_Z = np.array(df[df['element'] != 'H']['z'])
        _max_pos_X = np.max(all_atom_pos_X)
        _min_pos_X = np.min(all_atom_pos_X)
        _max_pos_Y = np.max(all_atom_pos_Y)
        _min_pos_Y = np.min(all_atom_pos_Y)
        _max_pos_Z = np.max(all_atom_pos_Z)
        _min_pos_Z = np.min(all_atom_pos_Z)
        if _max_pos_X > max_pos_X:
            max_pos_X = _max_pos_X
        if _min_pos_X < min_pos_X:
            min_pos_X = _min_pos_X
        if _max_pos_Y > max_pos_Y:
            max_pos_Y = _max_pos_Y
        if _min_pos_Y < min_pos_Y:
            min_pos_Y = _min_pos_Y
        if _max_pos_Z > max_pos_Z:
            max_pos_Z = _max_pos_Z
        if _min_pos_Z < min_pos_Z:
            min_pos_Z = _min_pos_Z

        num += 1
        
        
    print("max atom's X pos is {:}".format(max_pos_X))
    print("min atom's X pos is {:}".format(min_pos_X))
    print("max atom's Y pos is {:}".format(max_pos_Y))
    print("min atom's Y pos is {:}".format(min_pos_Y))
    print("max atom's Z pos is {:}".format(max_pos_Z))
    print("min atom's Z pos is {:}".format(min_pos_Z))
if __name__ == "__main__":
    # max_atom_num_in_residue()
    max_min_pos()

        
