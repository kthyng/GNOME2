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
        # (need to subtract in tracpy to get differential position)

        delta = np.zeros_like(positions)

        if self.active and self.on:

            # CALL TRACPY HERE
            # delta[in_water_mask] = tracpy. #self.velocity * time_step

            # NEED TO HAVE ALREADY CONVERTED TO LAT LON HERE FOR DELTA LON/LAT

        return delta


