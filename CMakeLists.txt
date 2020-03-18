# NOTE: Update `install_dependencies.sh` if updating the minimum required version.
cmake_minimum_required(VERSION 3.2)

project(smarties)

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# List of options

option(COMPILE_PY_SO "Compile Python bindings" ON)

# Smarties library

set(root_folder ${CMAKE_CURRENT_LIST_DIR})
set(source_dir ${root_folder}/source)

set(smarties_sources
    ${source_dir}/Learners/PPO.cpp
    ${source_dir}/Learners/MixedPG.cpp
    ${source_dir}/Learners/NAF.cpp
    ${source_dir}/Learners/DPG.cpp
    ${source_dir}/Learners/DQN.cpp
    ${source_dir}/Learners/ACER.cpp
    ${source_dir}/Learners/RACER.cpp
    ${source_dir}/Learners/CMALearner.cpp
    ${source_dir}/Learners/Learner_pytorch.cpp
    ${source_dir}/Learners/Learner_approximator.cpp
    ${source_dir}/Learners/AlgoFactory.cpp
    ${source_dir}/Learners/Learner.cpp

    ${source_dir}/ReplayMemory/Episode.cpp
    ${source_dir}/ReplayMemory/Sampling.cpp
    ${source_dir}/ReplayMemory/Collector.cpp
    ${source_dir}/ReplayMemory/MemoryBuffer.cpp
    ${source_dir}/ReplayMemory/DataCoordinator.cpp
    ${source_dir}/ReplayMemory/MemoryProcessing.cpp

    ${source_dir}/Math/Math_tests.cpp
    ${source_dir}/Utils/Warnings.cpp
    ${source_dir}/Utils/Profiler.cpp
    ${source_dir}/Utils/StatsTracker.cpp

    ${source_dir}/Core/Launcher.cpp
    ${source_dir}/Core/Master.cpp
    ${source_dir}/Core/Worker.cpp
    ${source_dir}/Core/StateAction.cpp

    ${source_dir}/Communicator.cpp
    ${source_dir}/Settings.cpp
    ${source_dir}/Engine.cpp

    ${source_dir}/Network/Approximator.cpp
    ${source_dir}/Network/Builder.cpp
    ${source_dir}/Network/Network.cpp
    ${source_dir}/Network/Optimizer.cpp
    ${source_dir}/Network/CMA_Optimizer.cpp
)

add_library(smarties SHARED ${smarties_sources})

# compilation flags
target_compile_features(smarties PUBLIC cxx_std_14)

set(cxx_private_flags -Wall -Wextra -pedantic -Wno-unused-parameter -Wno-maybe-uninitialized -fmax-errors=3)
target_compile_options(smarties PRIVATE ${cxx_private_flags})

target_compile_options(smarties PRIVATE
  $<$<CONFIG:Debug>:-O0 -g>
  $<$<CONFIG:Release>:-DNDEBUG -O3 -march=native -fstrict-aliasing -ffast-math -falign-functions=32>
  )

# include flags
# here interface since no internal source include the following directory
target_include_directories(smarties INTERFACE ${root_folder}/include)

# -fPIC
set_property(TARGET smarties PROPERTY POSITION_INDEPENDENT_CODE ON)

# External libraries

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# For supporting CMake < 3.9:

if(NOT TARGET MPI::MPI_CXX)
  add_library(MPI::MPI_CXX IMPORTED INTERFACE)
  
  set_property(TARGET MPI::MPI_CXX
    PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
  set_property(TARGET MPI::MPI_CXX
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
  set_property(TARGET MPI::MPI_CXX
    PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()

if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(Threads REQUIRED)
  add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
  set_property(TARGET OpenMP::OpenMP_CXX
    PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
  # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
  set_property(TARGET OpenMP::OpenMP_CXX
    PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

target_link_libraries(smarties PUBLIC
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  ${BLAS_LIBRARIES})


# Python bindings

if (COMPILE_PY_SO)
  add_subdirectory(${source_dir}/extern/pybind11)
  set(PYBIND11_CPP_STANDARD -std=c++14)

  add_library(pysmarties MODULE ${source_dir}/smarties_pybind11.cpp)
  target_link_libraries(pysmarties PRIVATE smarties pybind11::module)

  set_target_properties(${LIB_MIR} PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

endif()



