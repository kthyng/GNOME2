
"""
renderer.py

module to hold all teh map rendering code.

"""

import os
import glob
import copy

import gnome    # implicitly used when loading from dict by new_from_dict
from gnome.outputters import Outputter
from gnome.utilities.map_canvas import MapCanvas
from gnome.utilities import serializable
from gnome.utilities.file_tools import haz_files


class Renderer(Outputter, MapCanvas, serializable.Serializable):

    """
    Map Renderer

    class that writes map images for GNOME results:
        writes the frames for the LE "movies", etc.

    """

    background_map_name = 'background_map.png'
    foreground_filename_format = 'foreground_%05i.png'
    foreground_filename_glob = 'foreground_?????.png'

    # todo: how should images_dir be saved? Absolute? Currently, it is relative
    _update = ['viewport', 'map_BB', 'images_dir', 'image_size', 'draw_ontop']
    _create = ['image_size', 'projection_class', 'draw_ontop']

    _create.extend(_update)
    _state = copy.deepcopy(serializable.Serializable._state)
    _state.add(create=_create, update=_update)
    _state.add_field(serializable.Field('filename', isdatafile=True,
                    create=True, read=True, test_for_eq=False))

    @classmethod
    def new_from_dict(cls, dict_):
        """
        change projection_type from string to correct type
        """

        proj = eval(dict_.pop('projection_class'))
        viewport = dict_.pop('viewport')

        # filename = dict_.pop('filename')
        # images_dir = dict_.pop('images_dir')
        # obj =  cls(filename, images_dir, projection_class=proj, **dict_)

        obj = cls(projection_class=proj, **dict_)
        obj.viewport = viewport
        return obj

    def __init__(
        self,
        filename=None,
        images_dir='./',
        image_size=(800, 600),
        cache=None,
        output_timestep=None,
        output_zero_step=True,
        output_last_step=True,
        draw_ontop='forecast',
        **kwargs
        ):
        """
        Init the image renderer.

        Following args are passed to base class Outputter's init:

        :param cache: sets the cache object from which to read data. The model
            will automatically set this param

        :param output_timestep: default is None in which case everytime the
            write_output is called, output is written. If set, then output is
            written every output_timestep starting from model_start_time.
        :type output_timestep: timedelta object

        :param output_zero_step: default is True. If True then output for
            initial step (showing initial release conditions) is written
            regardless of output_timestep
        :type output_zero_step: boolean

        :param output_last_step: default is True. If True then output for
            final step is written regardless of output_timestep
        :type output_last_step: boolean

        :param draw_ontop: draw 'forecast' or 'uncertain' LEs on top. Default
            is to draw 'forecast' LEs, which are in black on top
        :type draw_ontop: str

        Remaining kwargs are passed onto baseclass's __init__ with a direct
        call: MapCanvas.__init__(..)

        Optional parameters (kwargs)

        :param projection_class: gnome.utilities.projections class to use.
            Default is gnome.utilities.projections.FlatEarthProjection
        :param map_BB:  map bounding box. Default is to use
            land_polygons.bounding_box. If land_polygons is None, then this is
            the whole world, defined by ((-180,-90),(180, 90))
        :param viewport: viewport of map -- what gets drawn and on what scale.
            Default is to set viewport = map_BB
        :param image_mode: Image mode ('P' for palette or 'L' for Black and
            White image). BW_MapCanvas inherits from MapCanvas and sets the
            mode to 'L'. Default image_mode is 'P'.
        :param id: unique identifier for a instance of this class (UUID given
            as a string). This is used when loading an object from a persisted
            model.
        """

        # set up the canvas

        self._filename = filename
        if filename is not None:
            polygons = haz_files.ReadBNA(filename, 'PolygonSet')
        else:
            polygons = None
        Outputter.__init__(self, cache, output_timestep, output_zero_step,
                           output_last_step)
        MapCanvas.__init__(self, image_size, land_polygons=polygons,
                           **kwargs)

        self.images_dir = images_dir

        self.last_filename = ''

        self.draw_ontop = draw_ontop

    filename = property(lambda self: self._filename)

    @property
    def draw_ontop(self):
        return self._draw_ontop

    @draw_ontop.setter
    def draw_ontop(self, val):
        if val not in ['forecast', 'uncertain']:
            raise ValueError("'draw_ontop' must be either 'forecast' or"
                            "'uncertain'. {0} is invalid.".format(val))
        self._draw_ontop = val

    def images_dir_to_dict(self):
        return os.path.abspath(self.images_dir)

    def prepare_for_model_run(self,
        model_start_time,
        cache=None,
        **kwargs):
        """
        prepares the renderer for a model run.

        Parameters passed to base class (use super): model_start_time, cache

        Does not take any other input arguments; however, to keep the interface
        the same for all outputters, define **kwargs and pass into base class

        In this case, it draws the background image and clears the previous
        images. If you want to save the previous images, a new output dir
        should be set.

        """

        super(Renderer, self).prepare_for_model_run(model_start_time,
                cache, **kwargs)

        self.clear_output_dir()

        self.draw_background()
        self.save_background(os.path.join(self.images_dir,
                             self.background_map_name))

    def clear_output_dir(self):

        # clear out output dir:
        # don't need to do this -- it will get written over.

        try:
            os.remove(os.path.join(self.images_dir,
                      self.background_map_name))
        except OSError:

                        # it's not there to delete..

            pass

        foreground_filenames = glob.glob(os.path.join(self.images_dir,
                self.foreground_filename_glob))
        for name in foreground_filenames:
            os.remove(name)

    def write_output(self, step_num, islast_step=False):
        """
        Render the map image, according to current parameters.

        :param step_num: the model step number you want rendered.
        :type step_num: int

        :param islast_step: default is False. Flag that indicates that step_num
            is last step. If 'output_last_step' is True then this is written
            out
        :type islast_step: bool

        :returns: A dict of info about this step number if this step
            is to be output, None otherwise.
                   'step_num': step_num
                   'image_filename': filename
                   'time_stamp': time_stamp # as ISO string

        use super to call base class write_output method

        If this is last step, then data is written; otherwise
        prepare_for_model_step determines whether to write the output for
        this step based on output_timestep
        """

        super(Renderer, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        image_filename = os.path.join(self.images_dir,
                self.foreground_filename_format % step_num)

        self.create_foreground_image()

        # do a function call so data arrays get garbage collected after
        # function exists
        current_time_stamp = self._draw(step_num)

        # get the timestamp:
        time_stamp = current_time_stamp.isoformat()
        self.save_foreground(image_filename)

        self.last_filename = image_filename

        # update self._next_output_time if data is successfully written
        self._update_next_output_time(step_num, current_time_stamp)

        return {'step_num': step_num, 'image_filename': image_filename,
                'time_stamp': time_stamp}

    def _draw(self, step_num):
        """
        create a small function so data arrays are garbage collected from
        memory after this function exits - it returns current_time_stamp
        """

        # draw data for self.draw_ontop second so it draws on top
        scp = self.cache.load_timestep(step_num).items()
        if len(scp) == 1:
            self.draw_elements(scp[0])
        else:
            if self.draw_ontop == 'forecast':
                self.draw_elements(scp[1])
                self.draw_elements(scp[0])
            else:
                self.draw_elements(scp[0])
                self.draw_elements(scp[1])

        return scp[0].current_time_stamp

    def projection_class_to_dict(self):
        """ store projection class as a string for now since that is all that
        is required for persisting """

        return '{0}.{1}'.format(self.projection.__module__,
                                self.projection.__class__.__name__)
