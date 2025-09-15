"""
NumPy compatibility patches for mpslib functions.

This module provides fixes for deprecated NumPy methods used in mpslib,
specifically replacing the deprecated np.NaN with np.nan.
"""

import numpy as np


def patch_mpslib_numpy():
    """
    Patch MPSlib functions to use np.nan instead of deprecated np.NaN.
    This function should be called before using any MPSlib functions.
    
    This patch fixes the following issues:
    - Replaces deprecated np.NaN with np.nan in mpslib.mpslib module (line 44)
    """
    try:
        import mpslib.mpslib as mps_mpslib
        
        # Store the original __init__ method
        original_init = mps_mpslib.mpslib.__init__
        
        def patched_init(self, parameter_filename='mps.txt', method='mps_genesim', debug_level=-1, n_real=1, rseed=1,
                         out_folder='.', ti_fnam='ti.dat', simulation_grid_size=np.array([40, 20, 1]),
                         mask_fnam='mask.dat',
                         origin=np.zeros(3), grid_cell_size=np.array([1, 1, 1]), hard_data_fnam='hard.dat',
                         shuffle_simulation_grid=2, entropyfactor_simulation_grid=4, shuffle_ti_grid=1,
                         hard_data_search_radius=1,
                         soft_data_categories=np.arange(2), soft_data_fnam='soft.dat', n_threads=-1, verbose_level=0,
                         template_size=np.array([8, 7, 1]), n_multiple_grids=3, n_cond=16, n_cond_soft=1, n_min_node_count=0,
                         n_max_ite=1000000, n_max_cpdf_count=1, distance_measure=1, distance_max=0, distance_pow=1,
                         max_search_radius=10000000, max_search_radius_soft=10000000,                  
                         remove_gslib_after_simulation=1, gslib_combine=1, ti=np.empty(0), 
                         colocate_dimension=0,
                         do_estimation=0, 
                         do_entropy=0,
                         distance_min=-1, #OBSOLETE
                         mpslib_exe_folder=''):
            """Patched __init__ method that uses np.nan instead of np.NaN"""
            # Next few lines to keep compatibility with old (and misleading) use of O.par['dist_min']-->O.par['dist_max']
            if (distance_min>-1):
                distance_max = distance_min
            
            self.blank_grid = None
            self.blank_val = np.nan  # FIXED: Use np.nan instead of np.NaN
            self.parameter_filename = parameter_filename.lower()  # change string to lower case
            self.method = method.lower()  # change string to lower case
            self.verbose_level = verbose_level
            
            self.remove_gslib_after_simulation = remove_gslib_after_simulation  # remove individual gslib fiels after simulation
            self.gslib_combine = gslib_combine  # combine realzations into one gslib file

            self.sim = None

            self.par = {}

            self.par['n_real'] = n_real
            self.par['rseed'] = rseed
            self.par['n_max_cpdf_count'] = n_max_cpdf_count
            self.par['out_folder'] = out_folder
            self.par['ti_fnam'] = ti_fnam.lower()  # change string to lower case
            self.par['simulation_grid_size'] = simulation_grid_size
            self.par['origin'] = origin
            self.par['grid_cell_size'] = grid_cell_size
            self.par['mask_fnam'] = mask_fnam.lower() # change string to lower case
            self.par['hard_data_fnam'] = hard_data_fnam.lower()  # change string to lower case
            self.par['shuffle_simulation_grid'] = shuffle_simulation_grid
            self.par['entropyfactor_simulation_grid'] = entropyfactor_simulation_grid
            self.par['shuffle_ti_grid'] = shuffle_ti_grid
            self.par['hard_data_search_radius'] = hard_data_search_radius
            self.par['soft_data_categories'] = soft_data_categories
            self.par['soft_data_fnam'] = soft_data_fnam.lower()  # change string to lower case
            self.par['n_threads'] = n_threads
            self.par['debug_level'] = debug_level
            self.par['do_estimation'] = do_estimation
            self.par['do_entropy'] = do_entropy

            # if the method is GENSIM, add package specific parameters
            if self.method == 'mps_genesim':
                self.par['n_cond'] = n_cond
                self.par['n_cond_soft'] = n_cond_soft
                self.par['n_max_ite'] = n_max_ite
                self.par['n_max_cpdf_count'] = n_max_cpdf_count
                self.par['distance_measure'] = distance_measure
                self.par['distance_max'] = distance_max
                self.par['distance_pow'] = distance_pow
                self.par['colocate_dimension'] = colocate_dimension
                self.par['max_search_radius'] = max_search_radius
                self.par['max_search_radius_soft'] = max_search_radius_soft

            if self.method[0:10] == 'mps_snesim':
                self.par['template_size'] = template_size
                self.par['n_multiple_grids'] = n_multiple_grids
                self.par['n_min_node_count'] = n_min_node_count
                self.par['n_cond'] = n_cond

            # Set verbose_level on eas as well
            import mpslib.eas as eas
            eas.debug_level = self.par['debug_level']
            
            # Check if on windows
            import os
            self.iswin = 0
            if (os.name == 'nt'):
                self.iswin = 1

            # Find folder with executable files
            # Fiest check the root folder of mpslib, then mslib/scikit-mps/mpslib/bin
            mpslib_py_path, fn = os.path.split(__file__)
            if len(mpslib_exe_folder)==0:
                
                mpslib_exe_folder = os.path.abspath(os.path.join(mpslib_py_path, 'bin'))
                if (self.verbose_level>0):
                    print("Testing if EXEcutables in  %s" % (mpslib_exe_folder) )
                self.mpslib_exe_folder = mpslib_exe_folder
                if self.which(method,0) is None:
                    mpslib_exe_folder = os.path.abspath(os.path.join(mpslib_py_path, '..', '..'))
                    self.mpslib_exe_folder = mpslib_exe_folder
                    if (self.verbose_level>0):
                        print("Testing if EXEcutables in  %s" % (mpslib_exe_folder) )
                    
                self.which(method,0)
                
            if self.verbose_level>-1:
                print("Using %s installed in %s (scikit-mps in %s)" % (method,self.mpslib_exe_folder,__file__))
        
        # Replace the __init__ method
        mps_mpslib.mpslib.__init__ = patched_init
        
        print("✅ Successfully patched mpslib.mpslib.__init__ to use np.nan instead of np.NaN")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import mpslib.mpslib: {e}")
        return False
    except Exception as e:
        print(f"❌ Error patching mpslib: {e}")
        return False


def patch_numpy_nan_globally():
    """
    Global patch to replace np.NaN with np.nan for all numpy operations.
    This is a more aggressive approach that affects all numpy operations.
    """
    try:
        # Replace np.NaN with np.nan globally
        np.NaN = np.nan
        print("✅ Successfully replaced np.NaN with np.nan globally")
        return True
    except Exception as e:
        print(f"❌ Error in global numpy patch: {e}")
        return False


def apply_all_numpy_patches():
    """
    Apply all NumPy compatibility patches.
    """
    print("🔧 Applying NumPy compatibility patches...")
    
    success1 = patch_mpslib_numpy()
    success2 = patch_numpy_nan_globally()
    
    if success1 and success2:
        print("✅ All NumPy patches applied successfully")
        return True
    else:
        print("❌ Some NumPy patches failed to apply")
        return False