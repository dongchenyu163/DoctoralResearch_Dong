#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ScoreCalculator.h"

namespace py = pybind11;

PYBIND11_MODULE(score_calculator, m) {
  py::class_<ScoreCalculator>(m, "ScoreCalculator")
      .def(py::init<>())
      .def("set_point_cloud", &ScoreCalculator::setPointCloud, py::arg("points"), py::arg("normals"))
      .def(
          "configure_logging",
          &ScoreCalculator::configureLogging,
          py::arg("logger_name"),
          py::arg("enable_console"),
          py::arg("file_path"),
          py::arg("level"))
      .def("set_max_candidates", &ScoreCalculator::setMaxCandidates, py::arg("max_candidates"))
      .def("set_geo_weights", &ScoreCalculator::setGeoWeights, py::arg("w_fin"), py::arg("w_knf"), py::arg("w_tbl"))
      .def("set_geo_filter_ratio", &ScoreCalculator::setGeoFilterRatio, py::arg("ratio"))
      .def(
          "filter_by_geo_score",
          &ScoreCalculator::filterByGeoScore,
          py::arg("candidate_indices"),
          py::arg("knife_position"),
          py::arg("knife_normal"),
          py::arg("table_z"),
          R"pbdoc(Returns the filtered candidate matrix.)pbdoc")
      .def("last_geo_order", &ScoreCalculator::lastGeoOrder)
      .def(
          "calc_positional_scores",
          &ScoreCalculator::calcPositionalScores,
          py::arg("candidate_indices"),
          py::arg("knife_position"),
          py::arg("knife_normal"))
      .def(
          "calc_positional_distances",
          &ScoreCalculator::calcPositionalDistances,
          py::arg("candidate_indices"),
          py::arg("knife_position"),
          py::arg("knife_normal"))
      .def(
          "calc_dynamics_scores",
          &ScoreCalculator::calcDynamicsScores,
          py::arg("candidate_indices"),
          py::arg("wrench"),
          py::arg("friction_coef"),
          py::arg("friction_angle_deg"),
          py::arg("max_attempts"),
          py::arg("balance_threshold"))
      .def("last_dynamics_attempts", &ScoreCalculator::lastDynamicsAttempts, py::return_value_policy::reference_internal)
      .def_property_readonly("point_count", &ScoreCalculator::pointCount);
}
