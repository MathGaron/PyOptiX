#include "optix_geometry_wrapper.h"


OptixGeometryWrapper::OptixGeometryWrapper(optix::Geometry geometry)
{
    this->geometry = geometry;
    this->set_scoped_object(this->geometry.get());
}

OptixGeometryWrapper::~OptixGeometryWrapper()
{
    std::cout<<"~OptixContextWrapper deconstruction"<<std::endl;
    if(this->geometry.get() != 0)
        this->geometry->destroy();
}

void OptixGeometryWrapper::mark_dirty()
{
    this->geometry->markDirty();
}
bool OptixGeometryWrapper::is_dirty()
{
    return this->geometry->isDirty();
}
void OptixGeometryWrapper::set_primitive_count(unsigned int  num_primitives)
{
    this->geometry->setPrimitiveCount(num_primitives);
}
unsigned int OptixGeometryWrapper::get_primitive_count()
{
    return this->geometry->getPrimitiveCount();
}
void OptixGeometryWrapper::set_primitive_index_oOffset(unsigned int  index_offset)
{
    this->geometry->setPrimitiveIndexOffset(index_offset);
}
unsigned int OptixGeometryWrapper::get_primitive_index_offset()
{
    return this->geometry->getPrimitiveIndexOffset();
}

void OptixGeometryWrapper::set_bounding_box_program(OptixProgramWrapper* program)
{
    this->geometry->setBoundingBoxProgram(program->get_native_program());
}
void OptixGeometryWrapper::set_intersection_program(OptixProgramWrapper* program)
{
    this->geometry->setIntersectionProgram(program->get_native_program());

}

optix::Geometry OptixGeometryWrapper::get_native()
{
    return this->geometry;
}


#include "Python.h"
#include <boost/python.hpp>
void OptixGeometryWrapper::export_for_python()
{
    namespace bp = boost::python;

    bp::class_<OptixGeometryWrapper, bp::bases<OptixScopedObjectWrapper> >(
                "_OptixGeometryWrapper",
                "_OptixGeometryWrapper docstring",
                bp::init<optix::Geometry>())

            //*****************
            // DIRECT ACCESS
            //*****************

            .def("mark_dirty", &OptixGeometryWrapper::mark_dirty)
            .def("is_dirty", &OptixGeometryWrapper::is_dirty)
            .def("_set_primitive_count", &OptixGeometryWrapper::set_primitive_count)
            .def("get_primitive_count", &OptixGeometryWrapper::get_primitive_count)
            .def("_set_primitive_index_offset", &OptixGeometryWrapper::set_primitive_index_oOffset)
            .def("get_primitive_index_offset", &OptixGeometryWrapper::get_primitive_index_offset)

            .def("_set_bounding_box_program", &OptixGeometryWrapper::set_bounding_box_program)
            .def("_set_intersection_program", &OptixGeometryWrapper::set_intersection_program);
}

