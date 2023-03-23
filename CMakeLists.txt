###############################################################################
# Copyright (c) 2020 CEA
# This program and the accompanying materials are made available under the 
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# SPDX-License-Identifier: EPL-2.0
# Contributors: Francois Letierce
###############################################################################

cmake_minimum_required(VERSION 3.10)
set(CMAKE_VERBOSE_MAKEFILE FALSE)

CMAKE_POLICY(SET CMP0054 NEW)
CMAKE_POLICY(SET CMP0057 NEW)

CMAKE_POLICY(SET CMP0074 NEW)

project(mw_kokkos_cuda)

# Kokkos build: 
#cmake ../kokkos-master/ -DCMAKE_CXX_COMPILER=/home/letiercef/kokkos/kokkos-master/bin/nvcc_wrapper -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=~/kokkos/kokkos-install -DKokkos_ENABLE_OPENMP=On -DKokkos_ENABLE_HWLOC=On -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ARCH_INTEL_GEN=On -DKokkos_ARCH_PASCAL61=On -DBUILD_SHARED_LIBS=On -DCMAKE_POSITION_INDEPENDENT_CODE=On

find_package(Kokkos)
if(TARGET Kokkos::kokkos)
  message(STATUS "Kokkos::kokkos found")
  add_executable(mw_kokkos_cuda MiniWeatherKokkos.cc ChronoTimer.cc)
  target_link_libraries(mw_kokkos_cuda PUBLIC Kokkos::kokkos)
  target_compile_options(mw_kokkos_cuda PUBLIC -g --expt-relaxed-constexpr)
else()
  message(STATUS "Kokkos::kokkos not found")
endif()

# cmake:
# mkdir build; cd build; cmake .. -DCMAKE_CXX_COMPILER=/home/letiercef/kokkos/kokkos-master/bin/nvcc_wrapper -DCMAKE_INSTALL_PREFIX=~/kokkos/kokkos-install
