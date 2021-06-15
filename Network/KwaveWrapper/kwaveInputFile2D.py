"""
 * @file      kwave_input_file.py
 *
 * @author    Filip Vaverka
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            ivaverka@fit.vutbr.cz
 *
 * @brief     Simple implementation of k-Wave input file generator.
 *
 * @date      21 March 2018, 12:23 (created)
 *            21 March 2018, 12:23 (revised)
 *
 * @copyright Copyright (C) 2017 Filip Vaverka, Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 *
"""

import numpy as np
from scipy.interpolate import interpn
import h5py
import math
from datetime import datetime
from copy import deepcopy
from enum import Enum

class DataSetItem(object):
    def __init__(self, size, data_type, domain_type):
        self.size = size
        self.dataType = data_type
        self.domainType = domain_type
        self.value = None
    
    def set_value(self, value):
        self.value = value
        self.size = value.shape
        
    def is_valid(self):
        if self.size is None:
            return False
        
        return not any(x is None for x in self.size)

# ============================================================================ #
# k-Wave Input DataSets
kwave_input_data_sets = {
    # ---------------------------------------------------------------------------- #
    # Simulation Flags
    'simulation_flags': {
        'ux_source_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
        'uy_source_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
        'uz_source_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
        
        'p_source_flag':          DataSetItem((1, 1, 1), 'long', 'real'),
        'p0_source_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
        'transducer_source_flag': DataSetItem((1, 1, 1), 'long', 'real'),
        
        'nonuniform_grid_flag':   DataSetItem((1, 1, 1), 'long', 'real'),  # Must be 0
        
        'nonlinear_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
        'absorbing_flag':         DataSetItem((1, 1, 1), 'long', 'real'),
        
        'elastic_flag'  :         DataSetItem((1, 1, 1), 'long', 'real'), 
        'axisymmetric_flag':      DataSetItem((1, 1, 1), 'long', 'real'), 
        
        "sxx_source_flag":        DataSetItem((1, 1, 1), 'long', 'real'),
        "sxy_source_flag":        DataSetItem((1, 1, 1), 'long', 'real'),
        "sxz_source_flag":        DataSetItem((1, 1, 1), 'long', 'real'),
        "syy_source_flag":        DataSetItem((1, 1, 1), 'long', 'real'),
        "syz_source_flag":        DataSetItem((1, 1, 1), 'long', 'real'),
        "szz_source_flag":        DataSetItem((1, 1, 1), 'long', 'real'),
    },

    # ---------------------------------------------------------------------------- #
    # Grid Properties
    'grid_properties': {
        'Nx': DataSetItem((1, 1, 1), 'long', 'real'),
        'Ny': DataSetItem((1, 1, 1), 'long', 'real'),
        'Nz': DataSetItem((1, 1, 1), 'long', 'real'),
        'Nt': DataSetItem((1, 1, 1), 'long', 'real'),
        
        'dx': DataSetItem((1, 1, 1), 'float', 'real'),
        'dy': DataSetItem((1, 1, 1), 'float', 'real'),
        # 'dz': DataSetItem((1, 1, 1), 'float', 'real'),
        'dt': DataSetItem((1, 1, 1), 'float', 'real'),
    },

    # ---------------------------------------------------------------------------- #
    # Regular Medium Properties
    'regular_medium_properties': {
        'rho0':     DataSetItem(None, 'float', 'real'),  # Size = (Nx, Ny, Nz) or (1, 1, 1)
        'rho0_sgx': DataSetItem(None, 'float', 'real'),  # Size = (Nx, Ny, Nz) or (1, 1, 1)
        'rho0_sgy': DataSetItem(None, 'float', 'real'),  # Size = (Nx, Ny, Nz) or (1, 1, 1)
        'rho0_sgz': DataSetItem(None, 'float', 'real'),  # Size = (Nx, Ny, Nz) or (1, 1, 1)
        
        'c0':       DataSetItem(None, 'float', 'real'),  # Size = (Nx, Ny, Nz) or (1, 1, 1)
        
        'c_ref':    DataSetItem((1, 1, 1), 'float', 'real'),
    },

    # ---------------------------------------------------------------------------- #
    # Nonlinear Medium Properties
    # IF nonlinear_flag == 1:
    'nonlinear_medium_properties': {
        'BonA':     DataSetItem(None, 'float', 'real'),  # Size = (Nx, Ny, Nz) or (1, 1, 1)
    },

    # ---------------------------------------------------------------------------- #
    # Absorbing Medium Properties
    # IF absorbing_flag == 1:
    'absorbing_medium_properties': {
        'alpha_coeff': DataSetItem(None, 'float', 'real'),  # Size = (Nx, Ny, Nz) or (1, 1, 1)
        'alpha_power': DataSetItem((1, 1, 1), 'float', 'real'),
    },

    # ---------------------------------------------------------------------------- #
    # Sensor Properties
    'sensor_properties': {
        'sensor_mask_type':    DataSetItem((1, 1, 1), 'long', 'real'),

        # IF sensor_mask_type == 0, Size = (Nsens, 1, 1)
        'sensor_mask_index':   DataSetItem((None, 1, 1), 'long', 'real'),

        # IF sensor_mask_type == 1, Size = (Ncubes, 6, 1)
        'sensor_mask_corners': DataSetItem((None, 6, 1), 'long', 'real'),
    },

    # ---------------------------------------------------------------------------- #
    # Velocity Source Terms
    # IF ux_source_flags == 1 OR uy_source_flag == 1 OR uz_source_flag == 1
    'velocity_source_terms': {
        'u_source_mode':   DataSetItem((1, 1, 1), 'long', 'real'),
        'u_source_many':   DataSetItem((1, 1, 1), 'long', 'real'),
        'u_source_index':  DataSetItem((None, 1, 1), 'long', 'real'),  # Size = (Nsrc, 1, 1)
        
        'ux_source_input': DataSetItem((None, None, 1), 'float', 'real'),  # Size = (Nsrc, Nt_src, 1)
        'uy_source_input': DataSetItem((None, None, 1), 'float', 'real'),  # Size = (Nsrc, Nt_src, 1)
        'uz_source_input': DataSetItem((None, None, 1), 'float', 'real'),  # Size = (Nsrc, Nt_src, 1)
    },

    # ---------------------------------------------------------------------------- #
    # Pressure Source Terms
    # IF p_source_flag == 1
    'pressure_source_terms': {
        'p_source_mode':  DataSetItem((1, 1, 1), 'long', 'real'),
        'p_source_many':  DataSetItem((1, 1, 1), 'long', 'real'),
        'p_source_index': DataSetItem((None, 1, 1), 'long', 'real'),
        
        'p_source_input': DataSetItem((None, None, 1), 'float', 'real'),  # Size = (Nsrc, Nt_src, 1)
    },

    # ---------------------------------------------------------------------------- #
    # Transducer Source Terms
    # IF transducer_source_flag == 1
    'transducer_source_terms': {
        'u_source_index':          DataSetItem((None, 1, 1), 'long', 'real'),  # Size = (Nsrc, 1, 1)
        'transducer_source_input': DataSetItem((None, 1, 1), 'float', 'real'),  # Size = (Nt_src, 1, 1)
        'delay_mask':              DataSetItem((None, 1, 1), 'float', 'real'),  # Size = (Nsrc, 1, 1)
    },

    # ---------------------------------------------------------------------------- #
    # IVP Source Terms
    # IF p0_source_flag == 1
    'ivp_source_terms': {
        'p0_source_input': DataSetItem((None, None, None), 'float', 'real'),  # Size = (Nx, Ny, Nz)
    },

    # ---------------------------------------------------------------------------- #
    # k-Space and Shift Variables
    'kspace_shift_variables': {
        'ddx_k_shift_pos_r': DataSetItem((None, 1, 1), 'float', 'complex'),  # (Nx/2 + 1, 1, 1)
        'ddx_k_shift_neg_r': DataSetItem((None, 1, 1), 'float', 'complex'),  # (Nx/2 + 1, 1, 1)
        'ddy_k_shift_pos':   DataSetItem((1, None, 1), 'float', 'complex'),  # (1, Ny, 1)
        'ddy_k_shift_neg':   DataSetItem((1, None, 1), 'float', 'complex'),  # (1, Ny, 1)
        'ddz_k_shift_pos':   DataSetItem((1, 1, None), 'float', 'complex'),  # (1, 1, Nz)
        'ddz_k_shift_neg':   DataSetItem((1, 1, None), 'float', 'complex'),  # (1, 1, Nz)
        
        'x_shift_neg_r':     DataSetItem((None, 1, 1), 'float', 'complex'),  # (Nx/2 + 1, 1, 1)
        'y_shift_neg_r':     DataSetItem((1, None, 1), 'float', 'complex'),  # (1, Ny/2 + 1, 1)
        'z_shift_neg_r':     DataSetItem((1, 1, None), 'float', 'complex'),  # (1, 1, Nz/2 + 1)
    },

    # ---------------------------------------------------------------------------- #
    # PML Variables
    'pml_variables': {
        'pml_x_size':  DataSetItem((1, 1, 1), 'long', 'real'),
        'pml_y_size':  DataSetItem((1, 1, 1), 'long', 'real'),
        
        'pml_x_alpha': DataSetItem((1, 1, 1), 'float', 'real'),
        'pml_y_alpha': DataSetItem((1, 1, 1), 'float', 'real'),
      
        'pml_x':       DataSetItem((None, 1, 1), 'float', 'real'),  # (Nx, 1, 1)
        'pml_x_sgx':   DataSetItem((None, 1, 1), 'float', 'real'),  # (Nx, 1, 1)
        'pml_y':       DataSetItem((1, None, 1), 'float', 'real'),  # (1, Ny, 1)
        'pml_y_sgy':   DataSetItem((1, None, 1), 'float', 'real'),  # (1, Ny, 1)
   },
}

# ============================================================================ #
# k-Wave Input Attributes
kwave_input_attributes = {
    # ---------------------------------------------------------------------------- #
    # Input File Header
    'file_header': {
        'created_by':       None,
        'creation_date':    None,
        'file_description': None,
        'file_type':        'input',
        'major_version':    None,
        'minor_version':    None,
    },
}


# ============================================================================ #
# k-Wave Input File
class KWaveInputFile(object):
    class SourceMode(Enum):
        DIRICHLET = 0
        DEFAULT = 1

    def __init__(self, domain_dims, nt, domain_delta, dt, c_ref=1500.0, pml_alpha=(2, 2, 2), pml_size=(20, 20, 20)):
        self.grid = KWaveGrid(domain_dims, nt, domain_delta, dt, c_ref)
        self.pmlAlpha = pml_alpha
        self.pmlSize = pml_size
        
        self.dim = 2
        
        self.input_data_sets = {}
        self.input_attribs = {}

        self.__fill_simulation_flags()
        self.__fill_grid_properties()
        # self.__fill_kspace_shift_variables()
        self.__fill_pml_variables()
        self.__fill_sensor_properties()
        
    def set_medium_properties(self, rho0, c0):
        self.input_data_sets['regular_medium_properties'] = deepcopy(kwave_input_data_sets['regular_medium_properties'])

        if isinstance(rho0, (np.ndarray,)):
            self.input_data_sets['regular_medium_properties']['rho0'].set_value(rho0)

            self.input_data_sets['regular_medium_properties']['rho0_sgx'].set_value(self.grid.compute_staggered(rho0, (0.5, 0.0, 0.0)))
            self.input_data_sets['regular_medium_properties']['rho0_sgy'].set_value(self.grid.compute_staggered(rho0, (0.0, 0.5, 0.0)))
            self.input_data_sets['regular_medium_properties']['rho0_sgz'].set_value(self.grid.compute_staggered(rho0, (0.0, 0.0, 0.5)))
        else:
            self.input_data_sets['regular_medium_properties']['rho0'].value = rho0
            self.input_data_sets['regular_medium_properties']['rho0'].size = (1, 1, 1)
        
            self.input_data_sets['regular_medium_properties']['rho0_sgx'].value = rho0
            self.input_data_sets['regular_medium_properties']['rho0_sgx'].size = (1, 1, 1)
            self.input_data_sets['regular_medium_properties']['rho0_sgy'].value = rho0
            self.input_data_sets['regular_medium_properties']['rho0_sgy'].size = (1, 1, 1)
            self.input_data_sets['regular_medium_properties']['rho0_sgz'].value = rho0
            self.input_data_sets['regular_medium_properties']['rho0_sgz'].size = (1, 1, 1)

        if isinstance(c0, (np.ndarray,)):
            self.input_data_sets['regular_medium_properties']['c0'].set_value(c0)
        else:
            self.input_data_sets['regular_medium_properties']['c0'].value = c0
            self.input_data_sets['regular_medium_properties']['c0'].size = (1, 1, 1)
        
        self.input_data_sets['regular_medium_properties']['c_ref'].value = self.grid.cRef
        self.input_data_sets['regular_medium_properties']['c_ref'].size = (1, 1, 1)
        
    def set_non_lin_medium(self, coeff):
        self.input_data_sets['simulation_flags']['nonlinear_flag'].value = 0 if coeff is None else 1
        
        if coeff is None:
            self.input_data_sets.pop('nonlinear_medium_properties', None)
            return
        
        self.input_data_sets['nonlinear_medium_properties'] = deepcopy(kwave_input_data_sets['nonlinear_medium_properties'])
        
        self.input_data_sets['nonlinear_medium_properties']['BonA'].set_value(coeff)
    
    def set_absorb_medium(self, alpha_coeff, alpha_power=1):
        self.input_data_sets['simulation_flags']['absorbing_flag'] = 0 if alpha_coeff is None else 1
        
        if alpha_coeff is None:
            self.input_data_sets.pop('absorbing_medium_properties', None)
            return
        
        self.input_data_sets['absorbing_medium_properties'] = deepcopy(kwave_input_data_sets['absorbing_medium_properties'])
        
        self.input_data_sets['absorbing_medium_properties']['alpha_coeff'].set_value(alpha_coeff)
        self.input_data_sets['absorbing_medium_properties']['alpha_power'].value = alpha_power
    
    def set_p0_source_input(self, p0):
        self.input_data_sets['simulation_flags']['p0_source_flag'].value = 0 if p0 is None else 1
        
        if p0 is None:
            self.input_data_sets.pop('ivp_source_terms', None)
            return
        
        self.input_data_sets['ivp_source_terms'] = deepcopy(kwave_input_data_sets['ivp_source_terms'])
        self.input_data_sets['ivp_source_terms']['p0_source_input'].set_value(p0)

    def set_p_source(self, source_index, p_source_series, mode=SourceMode.DEFAULT):
        self.input_data_sets['simulation_flags']['p_source_flag'].value = 0 if source_index is None else p_source_series.shape[1]
        self.input_data_sets['pressure_source_terms'] = deepcopy(kwave_input_data_sets['pressure_source_terms'])

        self.input_data_sets['pressure_source_terms']['p_source_mode'].value = mode.value

        if source_index.shape[1] > 1 or source_index.shape[2] > 1:
            raise ValueError('Invalid source index shape')

        self.input_data_sets['pressure_source_terms']['p_source_index'].set_value(source_index)

        if p_source_series.shape[0] > 1 and p_source_series.shape[0] != source_index.shape[0]:
            raise ValueError('Number of signal series and source indexes mismatch')

        self.input_data_sets['pressure_source_terms']['p_source_many'].value = 1 if p_source_series.shape[0] > 1 else 0

        c0 = self.input_data_sets['regular_medium_properties']['c0'].value
        if mode == self.SourceMode.DIRICHLET:
            if np.isscalar(c0):
                p_norm_coeff = 1.0 / (3.0 * c0 * c0)
            else:
                p_norm_coeff = 1.0 / (3.0 * c0.flat[source_index])
        else:
            if np.isscalar(c0):
                p_norm_coeff = (2.0 * self.grid.dt) / (3.0 * c0 * self.grid.domainDelta[0])
            else:
                p_norm_coeff = (2.0 * self.grid.dt) / (3.0 * c0.flat[source_index] * self.grid.domainDelta[0])

        # TODO: This needs to be modified for heterogeneous media (using sound speed at given point)
        # hom_p_norm = 2.0 * self.grid.dt / (3.0 * self.grid.cRef * self.grid.domainDelta[0])

        self.input_data_sets['pressure_source_terms']['p_source_input'].set_value(p_source_series * p_norm_coeff)

    def set_u_source(self, source_index, ux_source_series, uy_source_series, uz_source_series, mode=SourceMode.DEFAULT):
        self.input_data_sets['simulation_flags']['ux_source_flag'].value = 0 if ux_source_series is None else ux_source_series.shape[1]
        self.input_data_sets['simulation_flags']['uy_source_flag'].value = 0 if uy_source_series is None else uy_source_series.shape[1]
        self.input_data_sets['simulation_flags']['uz_source_flag'].value = 0 if uz_source_series is None else uz_source_series.shape[1]

        self.input_data_sets['velocity_source_terms'] = deepcopy(kwave_input_data_sets['velocity_source_terms'])
        self.input_data_sets['velocity_source_terms']['u_source_mode'].value = mode.value

        if source_index.shape[1] > 1 or source_index.shape[2] > 1:
            raise ValueError('Invalid source index shape')
        self.input_data_sets['velocity_source_terms']['u_source_index'].set_value(source_index)

        uxyz_source_series = (ux_source_series, uy_source_series, uz_source_series)
        uxyz_source_series_names = ('ux_source_input', 'uy_source_input', 'uz_source_input')
        u_source_series_shape = None

        c0 = self.input_data_sets['regular_medium_properties']['c0'].value
        for u_source_series, u_source_series_name, grid_delta in zip(uxyz_source_series, uxyz_source_series_names, self.grid.domainDelta):
            if u_source_series is None:
                continue

            if u_source_series_shape is None:
                u_source_series_shape = u_source_series.shape

            if u_source_series.shape != u_source_series_shape:
                raise ValueError('Shapes of velocity source series mismatch')

            if u_source_series.shape[0] > 1 and u_source_series.shape[0] != source_index.shape[0]:
                raise ValueError('number of signal series and source indexes mismatch')

            if mode == self.SourceMode.DEFAULT:
                if np.isscalar(c0):
                    u_norm_coeff = 2.0 * c0 * self.grid.dt / grid_delta
                else:
                    u_norm_coeff = 2.0 * c0.flat[source_index] * self.grid.dt / grid_delta

            self.input_data_sets['velocity_source_terms'][u_source_series_name].set_value(u_source_series * u_norm_coeff)

    def set_transducer_source(self, source_index, source_series, delay_mask=None):
        self.input_data_sets['simulation_flags']['transducer_source_flag'] = 0 if source_series is None else source_series.shape[0]

        self.input_data_sets['transducer_source_terms'] = deepcopy(kwave_input_data_sets['transducer_source_terms'])

        if source_index.shape[1] > 1 or source_index.shape[2] > 1:
            raise ValueError('Invalid source index shape')
        self.input_data_sets['transducer_source_terms']['u_source_index'].set_value(source_index)

        if delay_mask is None:
            delay_mask = np.zeros(source_index.shape)

        if delay_mask.shape != source_index.shape:
            raise ValueError('Delay mask shape doesn\'t match source index shape')

        c0 = self.input_data_sets['regular_medium_properties']['c0'].value
        if np.isscalar(c0):
            u_norm_coeff = 2.0 * c0 * self.grid.dt / self.grid.domainDelta[0]
        else:
            u_norm_coeff = 2.0 * np.mean(c0.flat[source_index]) * self.grid.dt / self.grid.domainDelta[0]

        self.input_data_sets['transducer_source_terms']['transducer_source_input'].set_value(source_series * u_norm_coeff)

    def set_sensor_mask(self, mask_type, mask_data):
        self.input_data_sets['sensor_properties'] = deepcopy(kwave_input_data_sets['sensor_properties'])

        self.input_data_sets['sensor_properties']['sensor_mask_type'].value = mask_type

        if mask_type == 0:
            if mask_data.shape[1] != 1 or mask_data.shape[2] != 1:
                raise ValueError('Invalid dimensionality of sensor mask data (expected: N,1,1)')
            self.input_data_sets['sensor_properties']['sensor_mask_index'].set_value(mask_data)
        else:
            if mask_data.shape[1] != 6 or mask_data.shape[2] != 1:
                raise ValueError('Invalid dimensionality of sensor mask data (expected: N,6,1)')
            self.input_data_sets['sensor_properties']['sensor_mask_index'].set_value(mask_data)
        
    def __fill_simulation_flags(self):
        self.input_data_sets['simulation_flags'] = deepcopy(kwave_input_data_sets['simulation_flags'])
        self.input_data_sets['simulation_flags']['ux_source_flag'].value = 0
        self.input_data_sets['simulation_flags']['uy_source_flag'].value = 0
        self.input_data_sets['simulation_flags']['uz_source_flag'].value = 0
        
        self.input_data_sets['simulation_flags']['p_source_flag'].value = 0
        self.input_data_sets['simulation_flags']['p0_source_flag'].value = 0
        self.input_data_sets['simulation_flags']['transducer_source_flag'].value = 0
        
        self.input_data_sets['simulation_flags']['nonuniform_grid_flag'].value = 0
        
        self.input_data_sets['simulation_flags']['nonlinear_flag'].value = 0
        self.input_data_sets['simulation_flags']['absorbing_flag'].value = 0
        
        self.input_data_sets['simulation_flags']['elastic_flag'].value = 0
        self.input_data_sets['simulation_flags']['axisymmetric_flag'].value = 0
        
        self.input_data_sets['simulation_flags']["sxx_source_flag"].value = 0
        self.input_data_sets['simulation_flags']["sxy_source_flag"].value = 0
        self.input_data_sets['simulation_flags']["sxz_source_flag"].value = 0
        self.input_data_sets['simulation_flags']["syy_source_flag"].value = 0
        self.input_data_sets['simulation_flags']["syz_source_flag"].value = 0
        self.input_data_sets['simulation_flags']["szz_source_flag"].value = 0
        
    def __fill_grid_properties(self):
        self.input_data_sets['grid_properties'] = deepcopy(kwave_input_data_sets['grid_properties'])
        
        self.input_data_sets['grid_properties']['Nx'].value = self.grid.domainDims[0]
        self.input_data_sets['grid_properties']['Ny'].value = self.grid.domainDims[1]
        self.input_data_sets['grid_properties']['Nz'].value = self.grid.domainDims[2]
        self.input_data_sets['grid_properties']['Nt'].value = self.grid.Nt
        
        self.input_data_sets['grid_properties']['dx'].value = self.grid.domainDelta[0]
        self.input_data_sets['grid_properties']['dy'].value = self.grid.domainDelta[1]
        # self.input_data_sets['grid_properties']['dz'].value = self.grid.domainDelta[2]
        self.input_data_sets['grid_properties']['dt'].value = self.grid.dt
    
    def __fill_kspace_shift_variables(self):
        self.input_data_sets['kspace_shift_variables'] = deepcopy(kwave_input_data_sets['kspace_shift_variables'])
        k_space_shift_variables = self.input_data_sets['kspace_shift_variables']
        
        ddx_k_shift_pos_r = self.grid.compute_ddk_shift(self.grid.kx_vec, self.grid.domainDelta[0], 1.0)
        ddx_k_shift_pos_r = ddx_k_shift_pos_r[0:self.grid.domainDimsR[0], :, :].flatten().view(np.float64)
        k_space_shift_variables['ddx_k_shift_pos_r'].set_value(ddx_k_shift_pos_r.reshape(2*self.grid.domainDimsR[0], 1, 1))
        
        ddx_k_shift_neg_r = self.grid.compute_ddk_shift(self.grid.kx_vec, self.grid.domainDelta[0], -1.0)
        ddx_k_shift_neg_r = ddx_k_shift_neg_r[0:self.grid.domainDimsR[0], :, :].flatten().view(np.float64)
        k_space_shift_variables['ddx_k_shift_neg_r'].set_value(ddx_k_shift_neg_r.reshape(2*self.grid.domainDimsR[0], 1, 1))
        
        ddy_k_shift_pos = self.grid.compute_ddk_shift(self.grid.ky_vec, self.grid.domainDelta[1], 1.0)
        ddy_k_shift_pos = ddy_k_shift_pos.flatten().view(np.float64).reshape(1, 2*self.grid.domainDims[1], 1)
        k_space_shift_variables['ddy_k_shift_pos'].set_value(ddy_k_shift_pos)
        ddy_k_shift_neg = self.grid.compute_ddk_shift(self.grid.ky_vec, self.grid.domainDelta[1], -1.0)
        ddy_k_shift_neg = ddy_k_shift_neg.flatten().view(np.float64).reshape(1, 2*self.grid.domainDims[1], 1)
        k_space_shift_variables['ddy_k_shift_neg'].set_value(ddy_k_shift_neg)
        
        ddz_k_shift_pos = self.grid.compute_ddk_shift(self.grid.kz_vec, self.grid.domainDelta[2], 1.0)
        ddz_k_shift_pos = ddz_k_shift_pos.flatten().view(np.float64).reshape(1, 1, 2*self.grid.domainDims[2])
        k_space_shift_variables['ddz_k_shift_pos'].set_value(ddz_k_shift_pos)
        ddz_k_shift_neg = self.grid.compute_ddk_shift(self.grid.kz_vec, self.grid.domainDelta[2], -1.0)
        ddz_k_shift_neg = ddz_k_shift_neg.flatten().view(np.float64).reshape(1, 1, 2*self.grid.domainDims[2])
        k_space_shift_variables['ddz_k_shift_neg'].set_value(ddz_k_shift_neg)
        
        x_shift_neg_r = self.grid.compute_shift(self.grid.kx_vec, self.grid.domainDelta[0], -1.0)
        x_shift_neg_r = x_shift_neg_r[0:self.grid.domainDimsR[0], :, :].flatten().view(np.float64)
        k_space_shift_variables['x_shift_neg_r'].set_value(x_shift_neg_r.reshape(2*self.grid.domainDimsR[0], 1, 1))
        
        y_shift_neg_r = self.grid.compute_shift(self.grid.ky_vec, self.grid.domainDelta[1], -1.0)
        y_shift_neg_r = y_shift_neg_r[:, 0:self.grid.domainDimsR[1], :].flatten().view(np.float64)
        k_space_shift_variables['y_shift_neg_r'].set_value(y_shift_neg_r.reshape(1, 2*self.grid.domainDimsR[1], 1))
        
        z_shift_neg_r = self.grid.compute_shift(self.grid.kz_vec, self.grid.domainDelta[2], -1.0)
        z_shift_neg_r = z_shift_neg_r[:, :, 0:self.grid.domainDimsR[2]].flatten().view(np.float64)
        k_space_shift_variables['z_shift_neg_r'].set_value(z_shift_neg_r.reshape(1, 1, 2*self.grid.domainDimsR[2]))
    
    def __fill_pml_variables(self):
        self.input_data_sets['pml_variables'] = deepcopy(kwave_input_data_sets['pml_variables'])
        pml_vars = self.input_data_sets['pml_variables']
        
        pml_vars['pml_x_size'].value = self.pmlSize[0]
        pml_vars['pml_y_size'].value = self.pmlSize[1]
        if self.dim == 3:
            pml_vars['pml_z_size'].value = self.pmlSize[2]
        
        pml_vars['pml_x_alpha'].value = self.pmlAlpha[0]
        pml_vars['pml_y_alpha'].value = self.pmlAlpha[1]
        if self.dim == 3:
            pml_vars['pml_z_alpha'].value = self.pmlAlpha[2]
        
        pml_x = self.grid.compute_pml(self.grid.domainDims[0], self.grid.domainDelta[0], self.grid.dt, self.pmlSize[0],
                                      self.pmlAlpha[0], 0.0)
        pml_vars['pml_x'].set_value(pml_x.reshape(self.grid.domainDims[0], 1, 1))
        pml_x_s_g_x = self.grid.compute_pml(self.grid.domainDims[0], self.grid.domainDelta[0], self.grid.dt,
                                            self.pmlSize[0], self.pmlAlpha[0], 0.5)
        # pml_vars['pml_x_sgx'].set_value(pml_x_s_g_x.reshape(self.grid.domainDims[0], 1, 1))
        
        pml_y = self.grid.compute_pml(self.grid.domainDims[1], self.grid.domainDelta[1],
                                      self.grid.dt, self.pmlSize[1], self.pmlAlpha[1], 0.0)
        pml_vars['pml_y'].set_value(pml_y.reshape(1, self.grid.domainDims[1], 1))
        pml_y_s_g_y = self.grid.compute_pml(self.grid.domainDims[1], self.grid.domainDelta[1],
                                            self.grid.dt, self.pmlSize[1], self.pmlAlpha[1], 0.5)
        # pml_vars['pml_y_sgy'].set_value(pml_y_s_g_y.reshape(1, self.grid.domainDims[1], 1))
        if self.dim == 3:
            pml_z = self.grid.compute_pml(self.grid.domainDims[2], self.grid.domainDelta[2],
                                      self.grid.dt, self.pmlSize[2], self.pmlAlpha[2], 0.0)

            pml_vars['pml_z'].set_value(pml_z.reshape(1, 1, self.grid.domainDims[2]))
            pml_z_s_g_z = self.grid.compute_pml(self.grid.domainDims[2], self.grid.domainDelta[2],
                                                self.grid.dt, self.pmlSize[2], self.pmlAlpha[2], 0.5)
            pml_vars['pml_z_sgz'].set_value(pml_z_s_g_z.reshape(1, 1, self.grid.domainDims[2]))
            
    def __fill_file_header(self):
        self.input_attribs['file_header'] = deepcopy(kwave_input_attributes['file_header'])
        
        self.input_attribs['file_header']['created_by'] = "k-Wave Input Generator 1.0"
        self.input_attribs['file_header']['creation_date'] = str(datetime.now().isoformat())
        self.input_attribs['file_header']['file_description'] = ""
        self.input_attribs['file_header']['file_type'] = "input"
        self.input_attribs['file_header']['major_version'] = "1"
        self.input_attribs['file_header']['minor_version'] = "1"
        
    def __fill_sensor_properties(self):
        self.input_data_sets['sensor_properties'] = deepcopy(kwave_input_data_sets['sensor_properties'])
        
        self.input_data_sets['sensor_properties']['sensor_mask_type'].value = 0
        self.input_data_sets['sensor_properties']['sensor_mask_index'].set_value(np.ones((1, 1, 1)))

    @staticmethod
    def __create_string_attrib(data_set, attr_name, string):
        np_string = np.string_(string)
        tid = h5py.h5t.C_S1.copy()
        tid.set_size(len(string) + 1)
        data_set.attrs.create(attr_name, np_string, dtype=h5py.Datatype(tid))
    
    def write_to_file(self, out_file):
        self.__fill_file_header()
        
        for groupName, group in self.input_attribs.items():
            for name, item in group.items():
                self.__create_string_attrib(out_file, name, item)
        
        for groupName, group in self.input_data_sets.items():

            for name, item in group.items():
                if not item.is_valid():
                    continue
                
                new_size = (item.size[2], item.size[1], item.size[0])
                
                data_set = out_file.create_dataset(name, new_size, dtype=('f' if item.dataType == 'float' else 'u8'))

                if item.size != new_size:
                    data_set[:] = item.value.transpose((2, 1, 0))
                else:
                    data_set[:] = item.value
                    
                self.__create_string_attrib(data_set, 'data_type', item.dataType)
                self.__create_string_attrib(data_set, 'domain_type', item.domainType)

#===================== ADDED BY JOEP =======================
    def clear_source(self):
        self.input_data_sets['simulation_flags']['p0_source_flag'].value = 0
        self.input_data_sets['simulation_flags']['p_source_flag'].value = 0
#===================== ADDED BY JOEP =======================

class KWaveGrid(object):
    def __init__(self, domain_dims, nt, domain_delta, dt, c_ref):
        self.domainDims = domain_dims
        self.domainDelta = domain_delta
        self.Nt = nt
        self.dt = dt
        self.cRef = c_ref
        
        self.domainDimsR = (int(self.domainDims[0] / 2) + 1, 
                            int(self.domainDims[1] / 2) + 1, 
                            int(self.domainDims[2] / 2) + 1)
        
        self.kx_vec = self.__compute_wave_numbers(self.domainDims[0], self.domainDelta[0])
        self.kx_vec = self.kx_vec.reshape((self.kx_vec.shape[0], 1, 1))
        
        self.ky_vec = self.__compute_wave_numbers(self.domainDims[1], self.domainDelta[1])
        self.ky_vec = self.ky_vec.reshape((1, self.ky_vec.shape[0], 1))
        
        self.kz_vec = self.__compute_wave_numbers(self.domainDims[2], self.domainDelta[2])
        self.kz_vec = self.kz_vec.reshape((1, 1, self.kz_vec.shape[0]))

    @staticmethod
    def __compute_wave_numbers(nx, dx):
        if nx % 2 == 0:
            k = np.arange(-nx/2, nx/2) / nx
        else:
            nn = nx - 1
            k = np.arange(-nn/2, nn/2+1) / nx
        
        k[int(nx/2)] = 0
        
        k = (2.0*math.pi/dx) * k
            
        return k

    @staticmethod
    def compute_ddk_shift(kx_vec, dx, sign):
        return np.fft.ifftshift(1j * kx_vec * np.exp(sign * 1j * kx_vec * dx / 2.0))

    @staticmethod
    def compute_shift(kx_vec, dx, sign):
        return np.fft.ifftshift(np.exp(sign * 1j * kx_vec * dx / 2.0))
    
    def compute_pml(self, nx, dx, dt, pml_size, pml_alpha, offset):
        x = np.arange(1, pml_size + 1) + offset
        
        pml_left = pml_alpha * (self.cRef / dx) * pow((x - pml_size - 1) / (0 - pml_size), 4.0)
        pml_left = np.exp(-pml_left * dt / 2.0)
        pml_right = pml_alpha * (self.cRef / dx) * pow(x / pml_size, 4.0)
        pml_right = np.exp(-pml_right * dt / 2.0)
        
        pml = np.ones(nx)
        pml[0:pml_size] = pml_left
        pml[-pml_size:] = pml_right
        
        return pml

    def compute_staggered(self, values, offset=(0.5, 0.0, 0.0)):
        x = np.arange(self.domainDims[0])
        y = np.arange(self.domainDims[1])
        z = np.arange(self.domainDims[2])
        mesh = np.array(np.meshgrid(x + offset[0], y + offset[1], z + offset[2], indexing='ij'))
        points = np.rollaxis(mesh, 0, 4).reshape(self.domainDims + (3,))

        values_staggered = interpn((x, y, z), values, points, bounds_error=False, fill_value=float('nan'))

        if offset[0] > 0.0:
            values_staggered[-1:, :, :] = values[-1:, :, :]
        elif offset[0] < 0.0:
            values_staggered[0:1, :, :] = values[0:1, :, :]

        if offset[1] > 0.0:
            values_staggered[:, -1:, :] = values[:, -1:, :]
        elif offset[1] < 0.0:
            values_staggered[:, 0:1, :] = values[:, 0:1, :]

        if offset[2] > 0.0:
            values_staggered[:, :, -1:] = values[:, :, -1:]
        elif offset[2] < 0.0:
            values_staggered[:, :, 0:1] = values[:, :, 0:1]

        return values_staggered
