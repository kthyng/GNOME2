#!/usr/bin/env python

"""
tracpy_mover.py

This calls TRACMASS to step drifters on a structured grid.


"""

import copy

import numpy as np
from numpy import random

from gnome import basic_types
from gnome.movers import Mover
from gnome.utilities.projections import FlatEarthProjection as proj
from gnome.utilities import serializable
from tracpy.tracpy_class import Tracpy
import tracpy


class TracpyMover(Mover):

    """
    tracpy_mover
    """

    def __init__(
        self,
        currents_filename,
        grid_filename=None,
        nsteps=1, ndays=1, ff=1, tseas=3600.,
        ah=0., av=0., z0='s', zpar=1, do3d=0, doturb=0, name='test', dostream=0, N=1, 
        time_units='seconds since 1970-01-01', dtFromTracmass=None, zparuv=None, tseas_use=None
        **kwargs
        ):
        """
        tracpy_mover (velocity)

        create a tracpy_mover instance

        :param velocity: a (u, v, w) triple -- in meters per second
        
        Remaining kwargs are passed onto Mover's __init__ using super. 
        See Mover documentation for remaining valid kwargs.
        """

        # self.velocity = np.asarray(velocity,
        #                            dtype=basic_types.mover_type).reshape((3,
        #         ))  # use this, to be compatible with whatever we are using for location
        # self.uncertainty_scale = uncertainty_scale

        # initializes Tracpy class
        self.tp = Tracpy(currents_filename, grid_filename=grid_filename, nsteps=nsteps,
                        ndays=ndays, ff=ff, tseas=tseas, ah=ah, av=av, z0=z0, zpar=zpar,
                        do3d=do3d, doturb=doturb, name=name, dostream=dostream, N=N,
                        time_units=time_units, dtFromTracmass=dtFromTracmass,
                        zparuv=zparuv, tseas_use=tseas_use)

        # calls Mover class
        super(TracpyMover, self).__init__(**kwargs)

    def __repr__(self):
        return 'TracpyMover(<%s>)' % self.id

    def prepare_for_model_run(self, date, lon0, lat0):

        tinds, nc, t0save, xend, yend, zend, zp, ttend, t, flag = self.tp.prepare_for_model_run(date, lon0, lat0)

        return tinds, nc, t0save, xend, yend, zend, zp, ttend, t, flag

    def prepare_for_model_step2(self, tind, nc, flag, xend, yend, zend, j):

        xstart, ystart, zstart = self.tp.prepare_for_model_step(tind, nc, flag, xend, yend, zend, j)
        # xstart, ystart, zstart = self.tp.prepare_for_model_step(tinds[j+1], nc, flag, xend, yend, zend, j)

    def get_move(
        self,
        spill,
        time_step,
        model_time,
        ):
        """
        moves the particles defined in the spill object
        
        :param spill: spill is an instance of the gnome.spill.Spill class
        :param time_step: time_step in seconds
        :param model_time: current model time as a datetime object
        In this case, it uses the:
            positions
            status_code
        data arrays.
        
        :returns delta: Nx3 numpy array of movement -- in (long, lat, meters) units
        
        """

        # Get the data:
        try:
            positions = spill['positions'] # Nx3 with lon,lat,z
            status_codes = spill['status_codes']
        except KeyError, err:
            raise ValueError('The spill does not have the required data arrays\n'
                              + err.message)

        # which ones should we move?
        # status codes for things like off map or on beach
        in_water_mask = status_codes == basic_types.oil_status.in_water

        # compute the move

        delta = np.zeros_like(positions)

        if self.active and self.on:

            # Old:
            # delta[in_water_mask] = tracpy. #self.velocity * time_step

            # Convert from lon/lat to grid coords
            # Interpolate to get starting positions in grid space
            xstart, ystart, _ = tracpy.tools.interpolate2d(positions[:,0], positions[:,1], self.tp.grid, 'd_ll2ij')

            # Call TRACMASS
            # Since I know I am doing 2d for now, can I essentially ignore the z positions?
            # FLAG PROBABLY NEEDS TO BE UPDATED FOR STATUS CODES FOR EXITING DRIFTERS?
            xend,\
                yend,\
                zend,\
                flag,\
                ttend, U, V = self.tp.step(xstart, ystart, positions[:,2])
            # xend_temp,\
            #     yend_temp,\
            #     zend_temp,\
            #     flag[ind],\
            #     ttend_temp, U, V = self.tp.step(j, ttend[ind,j*self.tp.N], xstart, ystart, zstart)

            # Convert back to lon/lat from grid indices and calculate change in lon/lat
            lon, lat, _ = tracpy.tools.interpolate2d(xend, yend, self.tp.grid, 'm_ij2ll')
            delta[in_water_mask] = np.hstack((lon-positions[:,0], lat-positions[:,1], positions[:,2]))


        return delta

    def model_step_is_done(self):

        xend[ind,j*tp.N+1:j*tp.N+tp.N+1], \
            yend[ind,j*tp.N+1:j*tp.N+tp.N+1], \
            zend[ind,j*tp.N+1:j*tp.N+tp.N+1], \
            zp[ind,j*tp.N+1:j*tp.N+tp.N+1], \
            ttend[ind,j*tp.N+1:j*tp.N+tp.N+1] = tp.model_step_is_done(xend_temp, yend_temp, zend_temp, ttend_temp, ttend[ind,j*tp.N])
