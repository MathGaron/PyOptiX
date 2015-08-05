from PyOptix.objects.optix_acceleration import OptixAcceleration

__author__ = 'vizera-ubuntu'

from PyOptixCpp.Core import _OptixGeometryGroupWrapper

from PyOptix.objects.commons.optix_has_child import OptixHasChild
from PyOptix.objects.commons.optix_object import OptixObject

class OptixGeometryGroup(_OptixGeometryGroupWrapper, OptixObject, OptixHasChild):
    _acceleration = None

    def __init__(self, native, context):
        OptixObject.__init__(self, context, native)
        _OptixGeometryGroupWrapper.__init__(self, native)

        from PyOptix.objects.optix_geometry_instance import OptixGeometryInstance
        from PyOptix.objects.optix_acceleration import OptixAcceleration
        allowed_children = [OptixGeometryInstance, OptixAcceleration]

        OptixHasChild.__init__(self, allowed_children)

    def set_acceleration(self, acceleration:OptixAcceleration):
        self._set_acceleration(acceleration)
        self._acceleration = acceleration