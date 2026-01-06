cmake_minimum_required(VERSION 3.18)
project(ScoreCalculator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_subdirectory(pybind11)

pybind11_add_module(
  score_calculator
  cpp/score_calculator/bindings.cpp
  cpp/score_calculator/ScoreCalculator.cpp
)

target_link_libraries(score_calculator PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

target_include_directories(score_calculator PRIVATE cpp/score_calculator)
