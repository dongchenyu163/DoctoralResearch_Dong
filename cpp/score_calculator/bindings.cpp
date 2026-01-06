#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "ScoreCalculator.h"

namespace py = pybind11;

PYBIND11_MODULE(score_calculator, m) {
  py::class_<ScoreCalculator>(m, "ScoreCalculator")
      .def(py::init<>())
      .def("set_point_cloud", &ScoreCalculator::setPointCloud, py::arg("points"), py::arg("normals"))
      .def_property_readonly("point_count", &ScoreCalculator::pointCount)
      .def_property_readonly("points", &ScoreCalculator::points, py::return_value_policy::reference_internal)
      .def_property_readonly("normals", &ScoreCalculator::normals, py::return_value_policy::reference_internal);
}
