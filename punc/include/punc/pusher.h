// Copyright (C) 2018, Diako Darian and Sigvald Marholm
//
// This file is part of PUNC++.
//
// PUNC++ is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// PUNC++ is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// PUNC++. If not, see <http://www.gnu.org/licenses/>.

/**
 * @file		pusher.h
 * @brief		Particle pusher
 *
 * Functions for pushing (accelerating and moving) particles.
 */

#ifndef PUSHER_H
#define PUSHER_H

#include "population.h"

namespace punc
{

namespace df = dolfin;

/**
 * @brief Accelerates particles in absence of a magnetic field
 * @param[in,out]   pop     Population
 * @param           E       Electric field
 * @param           dt      Time-step
 * @return                  Kinetic energy at mid-step
 * @see boris()
 *
 * Advances particle velocities according to
 * \f[
 *      \frac{\mathbf{v}^\mathrm{new}-\mathbf{v}^\mathrm{old}}{\Delta t}
 *      \approx \dot{\mathbf{v}} = \frac{q}{m}\mathbf{E}
 * \f]
 *
 * If the velocities are time-staggered about the electric field (i.e. particle
 * positions), this is a centered difference. Otherwise, it is a forward
 * difference.
 * 
 * To initialize from time-step \f$ n=0 \f$ a time-staggered grid where
 * velocities are at half-integer time-steps, the particles can be accelerated
 * only half a time-step the first time.
 */
double accel(Population &pop, const df::Function &E, double dt);

double accel_cg1(Population &pop, const df::Function &E, double dt);

double accel_cg_new(Population &pop, const df::Function &E, double dt); 

double accel_cg_2d(Population &pop, const df::Function &E, double dt);

double accel_cg(Population &pop, const df::Function &E, double dt);

/**
 * @brief Accelerates particles in a homogeneous magnetic field
 * @param[in,out]   pop     Population
 * @param           E       Electric field
 * @param           B       Magnetic flux density
 * @param           dt      Time-step
 * @return                  Kinetic energy at mid-step
 * @see accel()
 *
 * Advances particle velocities according to the Boris scheme:
 * \f[
 *      \frac{\mathbf{v}^\mathrm{new}-\mathbf{v}^\mathrm{old}}{\Delta t}
 *      \approx \dot{\mathbf{v}} =
 *      \frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})
 *      \approx \frac{q}{m}\left(\mathbf{E}+
 *      \frac{\mathbf{v}^\mathrm{new}+\mathbf{v}^\mathrm{old}}{2}
 *      \times\mathbf{B}\right)
 * \f]
 * 
 * To initialize from time-step \f$ n=0 \f$ a time-staggered grid where
 * velocities are at half-integer time-steps, the particles must be accelerated
 * only half a time-step the first time.
 */
double boris(Population &pop, const df::Function &E, 
             const std::vector<double> &B, double dt);

/**
 * @brief Accelerates particles in a inhomogeneous magnetic field
 * @param[in,out]   pop     Population
 * @param           E       Electric field
 * @param           B       Magnetic flux density
 * @param           dt      Time-step
 * @return                  Kinetic energy at mid-step
 * @see boris()
 *
 * Same as punc::boris but for inhomogeneous magnetic field.
 */
double boris(Population &pop, const df::Function &E,
             const df::Function &B, double dt);

/**
 * @brief Move particles
 * @param[in,out]   pop     Population
 * @param           dt      Time-step
 * 
 * Move particles according to:
 * \f[
 *      \frac{\mathbf{x}^\mathrm{new}-\mathbf{x}^\mathrm{old}}{\Delta t} 
 *      \approx \dot\mathbf{x} = \mathbf{v}
 * \f]
 */
void move(Population &pop, const double dt);

void move_new(Population &pop, const double dt);

// FIXME: Make a separate function for imposing periodic BCs *after* move
/* void move_periodic(Population &pop, double dt, const std::vector<double> &Ld); */
void move_periodic(Population &pop, const double dt, const std::vector<double> &Ld);

void move_periodic_new(Population &pop, const double dt, const std::vector<double> &Ld);
}

#endif // PUSHER_H
