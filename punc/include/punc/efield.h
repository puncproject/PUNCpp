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
 * @file		poisson.h
 * @brief		Solvers for the electric field
 */

#ifndef EFIELD_H
#define EFIELD_H

#include <dolfin.h>
#include <petscvec.h>

namespace punc
{

namespace df = dolfin;

/**
 * @brief Solver for the electric field (in CG1)
 */
class ESolver
{
private:
    df::PETScKrylovSolver solver;         /// < Linear algebra solver
    // std::shared_ptr<df::FunctionSpace> W; /// < Function space
    std::shared_ptr<df::Form> a, L;       /// < Bilinear and linear forms
    df::PETScMatrix A;                    /// < Stiffness matrix
    df::PETScVector b;                    /// < Load vector

public:
     /**
     * @brief Constructor 
     * @param W                 The vector function space of the electric field
     * @param method            Method of linear algebra solver
     * @param preconditioner    Preconditioner for matrix equation
     */
    ESolver(const df::FunctionSpace &W,
            std::string method = "gmres",
            std::string preconditioner = "hypre_amg");

    /**
     * @brief Solver
     * @param  E[in, out]            The electric field
     * @param  phi[in]               The electric potential            
     */
    void solve(df::Function &E, const df::Function &phi);
};

/**
 * @brief Solver for the electric field (in DG0)
 */
class EFieldDG0
{
  private:
    std::shared_ptr<df::Form> M; /// < Mass matrix

  public:
	std::shared_ptr<df::FunctionSpace> Q; /// < DG0 function space
    
    /**
     * @brief Constructor 
     * @param mesh      The mesh
     */
	EFieldDG0(std::shared_ptr<const df::Mesh> mesh);

    /**
     * @brief Solver
     * @param  phi[in]               The electric potential
     * @return The electric field            
     */
    df::Function solve(const df::Function &phi);
};

/**
 * @brief Solver for the electric field 
 * 
 * The electric field is solved in DG0, and than projected to CG1 by either using
 * arithmetic mean or Clement interpolation
 */
class EFieldMean
{
  private:
    std::shared_ptr<df::FunctionSpace> Q, W; /// < Function spaces (DG0 and CG1)
    std::shared_ptr<df::Form> a, b, c, d;    /// < Forms
    df::PETScMatrix A;                       /// < Transformation matrix (DG0 -> CG1)
    df::PETScVector ones, Av, e_dg0;         /// < Vectors

  public:
    std::shared_ptr<df::FunctionSpace> V; /// < Function space (CG1)

    /**
     * @brief Constructor 
     * @param mesh[in]      The mesh
     * @param aritheticmean[in]  true for arithmetic mean method, false for Clement interpolation
     */
    EFieldMean(std::shared_ptr<const df::Mesh> mesh, bool arithmetic_mean = false);

    /**
     * @brief Calculates either the mean or Clement interpolation of the electric field
     * @param  phi[in]               The electric potential
     * @return The electric field            
     */
    df::Function mean(const df::Function &phi);
};

// TBD: This is actually more generic.
// Perhaps we should have some place to put such generic functions acting
// on FEniCS functions?
/**
 * @brief Clement interpolant
 * 
 * Given a non-smooth function (in DG0), projects the function into CG1 by 
 * using Clement interpolation.
 */
class ClementInterpolant
{
  private:
    std::shared_ptr<df::Form> a, b; /// < Forms
    df::PETScMatrix A;              /// < Matrix representing the Clement interpolation
    df::PETScVector ones, Av;       /// < Vectors

  public:
    std::shared_ptr<df::FunctionSpace> V; /// < Function spaces (CG1)
    std::shared_ptr<df::FunctionSpace> Q; /// < Function spaces (DG0)

    /**
     * @brief Constructor 
     * @param mesh[in]      The mesh
     */
    ClementInterpolant(std::shared_ptr<const df::Mesh> mesh);

    /**
     * @brief Projects the function from DG0 to CG1
     * @param  u[in]      The function in DG0
     * return  The function in CG1
     */
    df::Function interpolate(const df::Function &u);

};

} // namespace punc

#endif // POISSON_H
