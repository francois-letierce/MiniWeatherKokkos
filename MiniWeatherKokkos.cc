// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//////////////////////////////////////////////////////////////////////////////////////////

/*
** Copyright (c) 2018, National Center for Computational Sciences, Oak Ridge National Laboratory. All rights reserved.
**
** Portions Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
*/

// Copyright (c) 2018, National Center for Computational Sciences, Oak Ridge National Laboratory
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.

// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "Kokkos_Core.hpp"

#include "ChronoTimer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * NOTE: Version de MiniWeatherArray.cc d'Arcane, recodé en Kokkos pour
 * besoin de comparaison.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// namespace MiniWeatherArray
// {

// constexpr double pi = 3.14159265358979323846264338327;       //Pi
constexpr double grav = 9.8;                             //Gravitational acceleration (m / s^2)
constexpr double cp = 1004.;                             //Specific heat of dry air at constant pressure
constexpr double rd = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
constexpr double p0 = 1.e5;                              //Standard pressure at the surface in Pascals
constexpr double C0 = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
constexpr double gamm = 1.40027894002789400278940027894; //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
constexpr double xlen = 2.e4;     //Length of the domain in the x-direction (meters)
constexpr double zlen = 1.e4;     //Length of the domain in the z-direction (meters)
constexpr double hv_beta = 0.25;  //How strong to diffuse the solution: hv_beta \in [0:1]
constexpr double cfl = 1.50;      //"Courant, Friedrichs, Lewy" number (for numerical stability)
constexpr double max_speed = 450; //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
constexpr int hs = 2;             //"Halo" size: number of cells needed for a full "stencil" of information for reconstruction
constexpr int sten_size = 4;      //Size of the stencil used for interpolation

//Parameters for indexing and flags
constexpr int NUM_VARS = 4; //Number of fluid state variables
constexpr int ID_DENS = 0;  //index for density ("rho")
constexpr int ID_UMOM = 1;  //index for momentum in the x-direction ("rho * u")
constexpr int ID_WMOM = 2;  //index for momentum in the z-direction ("rho * w")
constexpr int ID_RHOT = 3;  //index for density * potential temperature ("rho * theta")
constexpr int DIR_X = 1;    //Integer constant to express that this operation is in the x-direction
constexpr int DIR_Z = 2;    //Integer constant to express that this operation is in the z-direction

//How is this not in the standard?!
inline double
dmin(double a, double b)
{
  if (a < b)
    return a;
  else
    return b;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MiniWeatherArray
{
 public:
  // using TargetMem = Kokkos::CudaSpace;
  // using TargetMem = Kokkos::CudaHostPinnedSpace
  using TargetMem = Kokkos::CudaUVMSpace;
  // using TargetMem = Kokkos::HostSpace;

  using TargetExec = Kokkos::Cuda;
  // using TargetExec = Kokkos::HPX;  // not possible ATM
  // using TargetExec = Kokkos::OpenMP;
  // using TargetExec = Kokkos::Serial;
  // using TargetExec = Kokkos::Threads;

  MiniWeatherArray(int nb_cell_x, int nb_cell_z, double final_time);
  
 public:
  
  int doOneIteration();
  void doExit(Kokkos::View<double[NUM_VARS], TargetMem> reduced_values);

 public:

  void init();
  KOKKOS_INLINE_FUNCTION
  void injection(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht) const;
  KOKKOS_INLINE_FUNCTION
  void hydro_const_theta(double z, double &r, double &t) const;
  void output(Kokkos::View<double***, TargetMem> state, double etime);
  void perform_timestep(Kokkos::View<double***, TargetMem> state, Kokkos::View<double***, TargetMem> state_tmp,
                        Kokkos::View<double***, TargetMem> flux, Kokkos::View<double***, TargetMem> tend, double dt);
  void semi_discrete_step(Kokkos::View<double***, TargetMem> nstate_init, Kokkos::View<double***, TargetMem> nstate_forcing,
                          Kokkos::View<double***, TargetMem> nstate_out, double dt, int dir,
                          Kokkos::View<double***, TargetMem> flux, Kokkos::View<double***, TargetMem> tend);
  void compute_tendencies_x(Kokkos::View<double***, TargetMem> nstate, Kokkos::View<double***, TargetMem> flux, Kokkos::View<double***, TargetMem> tend);
  void compute_tendencies_z(Kokkos::View<double***, TargetMem> nstate, Kokkos::View<double***, TargetMem> flux, Kokkos::View<double***, TargetMem> tend);
  void set_halo_values_x(Kokkos::View<double***, TargetMem> nstate);
  void set_halo_values_z(Kokkos::View<double***, TargetMem> nstate);

 private:

  class ConstValues
  {
    ///////////////////////////////////////////////////////////////////////////////////////
    // Variables that are initialized but remain static over the course of the simulation
    ///////////////////////////////////////////////////////////////////////////////////////
   public:
    double sim_time;            //total simulation time in seconds
    double output_freq;         //frequency to perform output in seconds
    double dt;                  //Model time step (seconds)
    int nx, nz;                 //Number of local grid cells in the x- and z- dimensions
    int i_beg, k_beg;           // beginning index in the x- and z-directions
    int nranks, myrank;         // my rank id
    int left_rank, right_rank;  // Rank IDs that exist to my left and right in the global domain
    int nx_glob, nz_glob;       // Number of total grid cells in the x- and z- dimensions
    double dx, dz;              // Grid space length in x- and z-dimension (meters)
  };
  ConstValues m_const;
  double sim_time() const { return m_const.sim_time; }
  double output_freq() const { return m_const.output_freq; }
  double dt() const { return m_const.dt; }
  int nx() const { return m_const.nx; }
  int nz() const { return m_const.nz; }
  int i_beg() const { return m_const.i_beg; }
  int k_beg() const { return m_const.k_beg; }
  double dx() const { return m_const.dx; }
  double dz() const { return m_const.dz; }

  Kokkos::View<double*, TargetMem> hy_dens_cell;       // hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
  Kokkos::View<double*, TargetMem> hy_dens_theta_cell; // hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
  Kokkos::View<double*, TargetMem> hy_dens_int;        // hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
  Kokkos::View<double*, TargetMem> hy_dens_theta_int;  // hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
  Kokkos::View<double*, TargetMem> hy_pressure_int;    // hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

  ///////////////////////////////////////////////////////////////////////////////////////
  // Variables that are dynamics over the course of the simulation
  ///////////////////////////////////////////////////////////////////////////////////////
  double etime;          //Elapsed model time
  double output_counter; //Helps determine when it's time to do output
  // Runtime variable arrays
  Kokkos::View<double***, TargetMem> nstate;     // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  Kokkos::View<double***, TargetMem> nstate_tmp; // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  Kokkos::View<double***, TargetMem> nflux;      // Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
  Kokkos::View<double***, TargetMem> ntend;      // Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
  int num_out = 0;   // The number of outputs performed so far
  int direction_switch = 1;
  
  public:
   ChronoTimer m_timer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MiniWeatherArray::
MiniWeatherArray(int nb_cell_x,int nb_cell_z,double final_time)
{
  m_const.nx_glob = nb_cell_x;     //Number of total cells in the x-dirction
  m_const.nz_glob = nb_cell_z;     //Number of total cells in the z-dirction
  m_const.dx = xlen / m_const.nx_glob;
  m_const.dz = zlen / m_const.nz_glob;

  std::cout << "Using 'MiniWeather' with Kokkos" << std::endl;

  m_const.sim_time = final_time;   //How many seconds to run the simulation
  m_const.output_freq = 100; //How frequently to output data to file (in seconds)
  //Set the cell grid size
  init();
  //Output the initial state
  output(nstate, etime);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////

int MiniWeatherArray::
doOneIteration()
{
  ///////////////////////////////////////////////////////////////////////////////////////
  // BEGIN USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////
  //The x-direction length is twice as long as the z-direction length
  //So, you'll want to have nx_glob be twice as large as nz_glob
  //nx_glob = nb_cell_x;     //Number of total cells in the x-dirction
  //nz_glob = nb_cell_z;     //Number of total cells in the z-dirction
  //sim_time = 1500;   //How many seconds to run the simulation
  //sim_time = final_time;   //How many seconds to run the simulation
  //output_freq = 100; //How frequently to output data to file (in seconds)
  //output_freq = 20; //How frequently to output data to file (in seconds)
  ///////////////////////////////////////////////////////////////////////////////////////
  // END USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////

  //Output the initial state
  //output(state, etime);
  
  m_timer.start();

  while (etime < m_const.sim_time)
  {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt() > m_const.sim_time) {
      m_const.dt = m_const.sim_time - etime;
    }

    //Perform a single time step
    perform_timestep(nstate, nstate_tmp, nflux, ntend, dt());

    //Inform the user

    std::cout << "Elapsed Time: " << etime << " / " << sim_time() << std::endl;

    //Update the elapsed time and output counter
    etime = etime + dt();
    output_counter = output_counter + dt();
    //If it's time for output, reset the counter, and do output

    if (output_counter >= output_freq()){
      output_counter = output_counter - output_freq();
      output(nstate, etime);
    }
    m_timer.stop();
    return 0;
  }
  m_timer.stop();
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void MiniWeatherArray::
perform_timestep(Kokkos::View<double***, TargetMem> state, Kokkos::View<double***, TargetMem> state_tmp,
                 Kokkos::View<double***, TargetMem> flux, Kokkos::View<double***, TargetMem> tend, double dt)
{
  if (direction_switch==1){
    //x-direction first
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_X, flux, tend);
    //z-direction second
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_Z, flux, tend);
  }
  else{
    //z-direction second
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_Z, flux, tend);
    //x-direction first
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_X, flux, tend);
  }
  if (direction_switch) {
    direction_switch = 0;
  }
  else
  {
    direction_switch = 1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void MiniWeatherArray::
semi_discrete_step(Kokkos::View<double***, TargetMem> nstate_init, Kokkos::View<double***, TargetMem> nstate_forcing,
                   Kokkos::View<double***, TargetMem> nstate_out, double dt, int dir,
                   Kokkos::View<double***, TargetMem> flux, Kokkos::View<double***, TargetMem> tend)
{
  if (dir == DIR_X) {
    // Set the halo values  in the x-direction
    set_halo_values_x(nstate_forcing);
    // Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(nstate_forcing, flux, tend);
  }
  else if (dir == DIR_Z){
    // Set the halo values  in the z-direction
    set_halo_values_z(nstate_forcing);
    // Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(nstate_forcing, flux, tend);
  }

  Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<3>> mdrange({0,0,0},{NUM_VARS,nz(),nx()});
  Kokkos::parallel_for(mdrange, KOKKOS_CLASS_LAMBDA(int ll, int k, int i)
  {
    nstate_out(ll,k+hs,i+hs) = nstate_init(ll,k+hs,i+hs) + dt * tend(ll, k, i);
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Compute the time tendencies of the fluid state using forcing in the x-direction

//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void MiniWeatherArray::
compute_tendencies_x(Kokkos::View<double***, TargetMem> nstate, Kokkos::View<double***, TargetMem> flux,
                     Kokkos::View<double***, TargetMem> tend)
{
  const auto dx = this->dx();
  const auto nx = this->nx();
  const auto nz = this->nz();
  
  // Pour une raison que je ne comprends pas Kokkos ne capture pas les constantes suivantes sur le device
  // Alors que NUM_VARS, c'est bon :o
  constexpr int ID_DENS_ = ID_DENS;
  constexpr int ID_UMOM_ = ID_UMOM;
  constexpr int ID_WMOM_ = ID_WMOM;
  constexpr int ID_RHOT_ = ID_RHOT;

  const double hv_coef = -hv_beta * dx / (16 * dt());

  //Compute fluxes in the x-direction for each cell
  Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<2>> mdrange2d({0, 0},{nz, nx + 1});
  Kokkos::parallel_for(mdrange2d, KOKKOS_CLASS_LAMBDA(int k, int i)
  {
    double r, u, w, t, p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];
    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for ( int ll = 0; ll < NUM_VARS; ll++){
      for ( int s = 0; s < sten_size; s++)
        stencil[s] = nstate(ll,k+hs,i+s);

      //Fourth-order-accurate interpolation of the state
      vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
      //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
      d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    r = vals[ID_DENS_] + hy_dens_cell(k + hs);
    u = vals[ID_UMOM_] / r;
    w = vals[ID_WMOM_] / r;
    t = (vals[ID_RHOT_] + hy_dens_theta_cell(k + hs)) / r;
    p = C0 * pow((r * t), gamm);

    // Compute the flux vector
    flux(ID_DENS_,k,i) = r * u - hv_coef * d3_vals[ID_DENS_];
    flux(ID_UMOM_,k,i) = r * u * u + p - hv_coef * d3_vals[ID_UMOM_];
    flux(ID_WMOM_,k,i) = r * u * w - hv_coef * d3_vals[ID_WMOM_];
    flux(ID_RHOT_,k,i) = r * u * t - hv_coef * d3_vals[ID_RHOT_];
  });

  // Use the fluxes to compute tendencies for each cell
  Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<3>> mdrange3d({0,0,0},{NUM_VARS,nz,nx});
  Kokkos::parallel_for(mdrange3d, KOKKOS_CLASS_LAMBDA(int ll, int k, int i)
  {
    tend(ll,k,i) = -(flux(ll,k,i+1) - flux(ll, k, i)) / dx;
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Compute the time tendencies of the fluid state using forcing in the z-direction

//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void MiniWeatherArray::
compute_tendencies_z(Kokkos::View<double***, TargetMem> nstate, Kokkos::View<double***, TargetMem> flux,
                     Kokkos::View<double***, TargetMem> tend)
{
  const auto dx = this->dx();
  const auto dz = this->dz();
  const auto nx = this->nx();
  const auto nz = this->nz();
  
  constexpr int ID_DENS_ = ID_DENS;
  constexpr int ID_UMOM_ = ID_UMOM;
  constexpr int ID_WMOM_ = ID_WMOM;
  constexpr int ID_RHOT_ = ID_RHOT;

  //Compute the hyperviscosity coeficient
  const double hv_coef = -hv_beta * dx / (16 * dt());
  //Compute fluxes in the x-direction for each cell
  Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<2>> mdrange2d({0, 0},{nz + 1, nx});
  Kokkos::parallel_for(mdrange2d, KOKKOS_CLASS_LAMBDA(int k, int i)
  {
    double r, u, w, t, p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];
    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (int ll = 0; ll < NUM_VARS; ll++){
      for (int s = 0; s < sten_size; s++)
        stencil[s] = nstate(ll,k+s,i+hs);

      //Fourth-order-accurate interpolation of the state
      vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
      //First-order-accurate interpolation of the third spatial derivative of the state
      d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    r = vals[ID_DENS_] + hy_dens_int(k);
    u = vals[ID_UMOM_] / r;
    w = vals[ID_WMOM_] / r;
    t = (vals[ID_RHOT_] + hy_dens_theta_int(k)) / r;
    p = C0 * pow((r * t), gamm) - hy_pressure_int(k);

    //Compute the flux vector with hyperviscosity
    flux(ID_DENS_,k,i) = r * w - hv_coef * d3_vals[ID_DENS_];
    flux(ID_UMOM_,k,i) = r * w * u - hv_coef * d3_vals[ID_UMOM_];
    flux(ID_WMOM_,k,i) = r * w * w + p - hv_coef * d3_vals[ID_WMOM_];
    flux(ID_RHOT_,k,i) = r * w * t - hv_coef * d3_vals[ID_RHOT_];
  });

  // Use the fluxes to compute tendencies for each cell
  Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<3>> mdrange3d({0,0,0},{NUM_VARS,nz,nx});
  Kokkos::parallel_for(mdrange3d, KOKKOS_CLASS_LAMBDA(int ll, int k, int i)
  {
    double t = -(flux(ll,k+1,i) - flux(ll, k, i)) / dz;
    if (ll == ID_WMOM_)
      t  = t - nstate(ID_DENS_,k+hs,i+hs) * grav;
    tend(ll, k, i) = t;
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MiniWeatherArray::
set_halo_values_x(Kokkos::View<double***, TargetMem> nstate)
{
  const auto nx = this->nx();
  const auto nz = this->nz();
  const auto dz = this->dz();
  const auto k_beg = this->k_beg();

  constexpr int ID_DENS_ = ID_DENS;
  constexpr int ID_UMOM_ = ID_UMOM;
  constexpr int ID_RHOT_ = ID_RHOT;
  constexpr int hs_ = hs;

  Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<2>> mdrange2d({0, 0},{NUM_VARS, nz});
  Kokkos::parallel_for(mdrange2d, KOKKOS_CLASS_LAMBDA(int ll, int k)
  {
    nstate(ll,k+hs_,0) = nstate(ll,k+hs_,nx+hs_-2);
    nstate(ll,k+hs_,1) = nstate(ll,k+hs_,nx+hs_-1);
    nstate(ll,k+hs_,nx+hs_) = nstate(ll,k+hs_,hs_);
    nstate(ll,k+hs_,nx+hs_+1) = nstate(ll,k+hs_,hs_+1);
  });

  if (m_const.myrank == 0) {
    Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<2>> mdrange({0, 0},{nz, hs_});
    Kokkos::parallel_for(mdrange, KOKKOS_CLASS_LAMBDA(int k, int i)
    {
      double z = ((double)(k_beg + k) + 0.5) * dz;
      if (abs(z - 3 * zlen / 4) <= zlen / 16){
        nstate(ID_UMOM_,k+hs_,i) = (nstate(ID_DENS_,k+hs_,i) + hy_dens_cell(k + hs_)) * 50.;
        nstate(ID_RHOT_,k+hs_,i) = (nstate(ID_DENS_,k+hs_,i) + hy_dens_cell(k + hs_)) * 298. - hy_dens_theta_cell(k + hs_);
      }
    });
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Set this task's halo values in the z-direction.
//decomposition in the vertical direction.
void MiniWeatherArray::
set_halo_values_z(Kokkos::View<double***, TargetMem> nstate)
{
  const auto nx = this->nx();
  const auto nz = this->nz();

  constexpr int hs_ = hs;

  Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<2>> mdrange2d({0, 0},{NUM_VARS, nx+2*hs_});
  Kokkos::parallel_for(mdrange2d, KOKKOS_CLASS_LAMBDA(int ll, int i)
  {
    if (ll == ID_WMOM){
      nstate(ll,0,i) = 0.0;
      nstate(ll,1,i) = 0.0;
      nstate(ll,nz+hs_,i) = 0.0;
      nstate(ll,nz+hs_+1,i) = 0.0;
    }
    else {
      nstate(ll,0,i) = nstate(ll,hs_,i);
      nstate(ll,1,i) = nstate(ll,hs_,i); // GG: bizarre que ce soit pareil que au dessus.
      nstate(ll,nz+hs_,i) = nstate(ll,nz+hs_-1,i);
      nstate(ll,nz+hs_+1,i) = nstate(ll,nz+hs_-1,i); // Idem
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MiniWeatherArray::
init()
{
  // int k, kk;

  m_const.nranks = 1;
  m_const.myrank = 0;

  // For simpler version, replace i_beg = 0, nx = nx_glob, left_rank = 0, right_rank = 0;

  double nper = ((double)m_const.nx_glob) / m_const.nranks;
  m_const.i_beg = (int)(round(nper * (m_const.myrank)));
  int i_end = (int)(round(nper * ((m_const.myrank) + 1))) - 1;
  m_const.nx = i_end - m_const.i_beg + 1;
  m_const.left_rank = m_const.myrank - 1;
  if (m_const.left_rank == -1)
    m_const.left_rank = m_const.nranks - 1;
  m_const.right_rank = m_const.myrank + 1;
  if (m_const.right_rank == m_const.nranks)
    m_const.right_rank = 0;

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  m_const.k_beg = 0;
  m_const.nz = m_const.nz_glob;

  const auto dx = this->dx();
  const auto dz = this->dz();
  const auto nx = this->nx();
  const auto nz = this->nz();
  const auto i_beg = this->i_beg();
  const auto k_beg = this->k_beg();
  
  constexpr int ID_DENS_ = ID_DENS;
  constexpr int ID_UMOM_ = ID_UMOM;
  constexpr int ID_WMOM_ = ID_WMOM;
  constexpr int ID_RHOT_ = ID_RHOT;

  // Allocate the model data
  Kokkos::resize(nstate, NUM_VARS,(nz + 2 * hs),(nx + 2 * hs));
  Kokkos::resize(nstate_tmp, NUM_VARS,(nz + 2 * hs),(nx + 2 * hs));
  Kokkos::resize(nflux, NUM_VARS,nz+1,nx+1); 
  Kokkos::resize(ntend, NUM_VARS,nz,nx);
  Kokkos::resize(hy_dens_cell, nz + 2 * hs);
  Kokkos::resize(hy_dens_theta_cell, nz + 2 * hs);
  Kokkos::resize(hy_dens_int, nz + 1);
  Kokkos::resize(hy_dens_theta_int, nz + 1);
  Kokkos::resize(hy_pressure_int, nz + 1);

  //Define the maximum stable time step based on an assumed maximum wind speed
  m_const.dt = dmin(dx, dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  // Display grid information

  std::cout << "nx_glob, nz_glob: " << m_const.nx_glob << " " << m_const.nz_glob << std::endl;
  std::cout << "dx,dz: " << dx << " " << dz << std::endl;
  std::cout << "dt: " << dt() << std::endl;

  const int nqpoints = 3;
  const double qpoints[] =
  {
    0.112701665379258311482073460022E0,
    0.500000000000000000000000000000E0,
    0.887298334620741688517926539980E0
  };
  const double qweights[] =
  {
    0.277777777777777777777777777779E0,
    0.444444444444444444444444444444E0,
    0.277777777777777777777777777779E0
  };

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////

  Kokkos::MDRangePolicy<TargetExec, Kokkos::Rank<2>> mdrange2d({0, 0},{nz+2*hs,nx+2*hs});
  Kokkos::parallel_for(mdrange2d, KOKKOS_CLASS_LAMBDA(int k, int i)
  {
    double r, u, w, t, hr, ht;
    for (int ll = 0; ll < NUM_VARS; ll++)
      nstate(ll,k,i) = 0.0;

    // Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
    for (int kk = 0; kk < nqpoints; kk++) {
      for (int ii = 0; ii < nqpoints; ii++) {
        // Compute the x,z location within the global domain based on cell and quadrature index
        double x = ((double)(i_beg + i - hs) + 0.5) * dx + (qpoints[ii] - 0.5) * dx;
        double z = ((double)(k_beg + k - hs) + 0.5) * dz + (qpoints[kk] - 0.5) * dz;

        // Set the fluid state based on the user's specification (default is injection in this example)
        injection(x, z, r, u, w, t, hr, ht);

        // Store into the fluid state array
        nstate(ID_DENS_,k,i) += + r * qweights[ii] * qweights[kk];
        nstate(ID_UMOM_,k,i) += + (r + hr) * u * qweights[ii] * qweights[kk];
        nstate(ID_WMOM_,k,i) += + (r + hr) * w * qweights[ii] * qweights[kk];
        nstate(ID_RHOT_,k,i) += + ((r + hr) * (t + ht) - hr * ht) * qweights[ii] * qweights[kk];
      }
    }

    for (int ll = 0; ll < NUM_VARS; ll++)
      nstate_tmp(ll,k,i) = nstate(ll,k,i);
  });
  std::cout << "End init part 1" << std::endl;

  // Compute the hydrostatic background state over vertical cell averages
  Kokkos::parallel_for(nz + 2 * hs, KOKKOS_CLASS_LAMBDA(int k)
  {
    double r, u, w, t, hr, ht;
    // for (k = 0; k < nz + 2 * hs; k++){
      double dens_cell = 0.0;
      double dens_theta_cell = 0.0;
      for (int kk = 0; kk < nqpoints; kk++){
        double z = (k_beg + k - hs + 0.5) * dz;

        // Set the fluid state based on the user's specification (default is injection in this example)
        injection(0.0, z, r, u, w, t, hr, ht);

        dens_cell += hr * qweights[kk];
        dens_theta_cell += hr * ht * qweights[kk];
      }
      hy_dens_cell(k) = dens_cell;
      hy_dens_theta_cell(k) = dens_theta_cell;
    //}
  });

  Kokkos::parallel_for(nz + 1, KOKKOS_CLASS_LAMBDA(int k)
  {
    double r, u, w, t, hr, ht;
    // Compute the hydrostatic background state at vertical cell interfaces
    // for (k = 0; k < nz + 1; k++) {
      double z = (k_beg + k) * dz;

      //Set the fluid state based on the user's specification (default is injection in this example)
      injection(0.0, z, r, u, w, t, hr, ht);

      hy_dens_int(k) = hr;
      hy_dens_theta_int(k) = hr * ht;
      hy_pressure_int(k) = C0 * pow((hr * ht), gamm);
    // }
  });
  std::cout << "End init part 2" << std::endl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// This test case is initially balanced but injects fast, cold air from the left boundary near the model top
// x and z are input coordinates at which to sample
// r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
// hr and ht are output background hydrostatic density and potential temperature at that location
KOKKOS_INLINE_FUNCTION
void MiniWeatherArray::
injection(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht) const
{
  //ARCANE_UNUSED(x);
  hydro_const_theta(z, hr, ht);
  r = 0.0;
  t = 0.0;
  u = 0.0;
  w = 0.0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
KOKKOS_INLINE_FUNCTION
void MiniWeatherArray::
hydro_const_theta(double z, double &r, double &t) const
{
  const double theta0 = 300.0; //Background potential temperature
  const double exner0 = 1.0;   //Surface-level Exner pressure
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                //Potential Temperature at z
  double exner = exner0 - grav * z / (cp * theta0); //Exner pressure at z
  double p = p0 * pow(exner, (cp / rd));            //Pressure at z
  double rt = pow((p / C0), (1. / gamm));           //rho*theta at z
  r = rt / t;                                //Density at z
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
//The file I/O uses netcdf, the only external library required for this mini-app.
//If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
void MiniWeatherArray::
output(Kokkos::View<double***, TargetMem> state, double etime)
{
  //ARCANE_UNUSED(state);
  //ARCANE_UNUSED(etime);
  // Ne fait rien car on n'est pas branché avec 'NetCDF'.
}

// Affiche la somme sur les mailles des variables.
// Cela est utile pour la validation
void MiniWeatherArray::
doExit(Kokkos::View<double[NUM_VARS], TargetMem> reduced_values)
{
  int k, i, ll;
  double sum_v[NUM_VARS];

  //auto ns = nstate.constSpan();

  for (ll = 0; ll < NUM_VARS; ll++)
    sum_v[ll] = 0.0;
  for (k = 0; k < nz(); k++){
    for (i = 0; i < nx() + 1; i++){
      for (ll = 0; ll < NUM_VARS; ll++){
        sum_v[ll] += nstate(ll,k+hs,i);
      }
    }
  }
  for ( int ll = 0; ll < NUM_VARS; ll++)
    reduced_values(ll) = sum_v[ll];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
  // from testMiniWeatherArray.arc
  int nb_cell_x = 400;
  int nb_cell_z = 200;
  double final_time = 2.0;
  
  if (argc == 1)
	{
    std::cout << "Usage:    no arg = default values, 3 args = nb_cell_x nb_cell_z final_time" << std::endl;
    std::cout << "##########################################################################" << std::endl;
		std::cout << "### Using default values: nb_cell_x=400, nb_cell_z=200, final_time=2.0 ###" << std::endl;
    std::cout << "##########################################################################" << std::endl;
	}
	else if (argc == 4)
	{
    nb_cell_x = atoi(argv[1]);
    nb_cell_z = atoi(argv[2]);
    final_time = atof(argv[3]);
    std::cout << "##########################################################################" << std::endl;
		std::cout << "### Using default values: nb_cell_x=" << nb_cell_x << ", nb_cell_z=" << nb_cell_z << ", final_time=" << final_time << " ###" << std::endl;
    std::cout << "##########################################################################" << std::endl;
	} else {
    std::cerr << "[ERROR], wrong number of arguments." << std::endl;
    std::cout << "Usage: no arg = default values, 3 args = nb_cell_x nb_cell_z final_time" << std::endl;
  }
  
  Kokkos::View<double[NUM_VARS], MiniWeatherArray::TargetMem> reduced_values("reduced_values");

  MiniWeatherArray* mw = new MiniWeatherArray(nb_cell_x, nb_cell_z, final_time);
  
  mw->doExit(reduced_values);
  std::cout << "---------- Init -----------" << std::endl;
  for (int ll = 0; ll < NUM_VARS; ll++)
    std::cout << "SUM var" << ll << " sum_v=" << reduced_values[ll] << std::endl;
  std::cout << std::endl;

  while (!(mw->doOneIteration())) {}
  mw->doExit(reduced_values);
  
  std::cout << "---------- Final -----------" << std::endl;
  for (int ll = 0; ll < NUM_VARS; ll++)
    std::cout << "SUM var" << ll << " sum_v=" << reduced_values[ll] << std::endl;
  std::cout << std::endl;
  
  std::cout << "---------- Check diff -----------" << std::endl;
  std::cout << "- only legit for default values -" << std::endl;
  double ref_v[NUM_VARS] = {26.6243096397231, 2631.23267576729, -259.490171322721, 7897.73654775889};
  for (int ll = 0; ll < NUM_VARS; ll++)
    std::cout << "rdiff" << ll << " = " << reduced_values[ll] - ref_v[ll] << std::endl;
  std::cout << std::endl;
  
  std::cout << "##### Timer stats for computeloop: #####" << std::endl;
  std::cout << mw->m_timer.summary() << std::endl;

  delete mw;
  } 
  Kokkos::finalize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// } // End namespace MiniWeatherArrayKokkos

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
