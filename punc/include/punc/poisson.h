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
 * @brief		Solvers for Poisson's equation
 */

#ifndef POISSON_H
#define POISSON_H

// These should eventually be removed.
// Poisson should only use the interface.
#include "object_BC.h"
#include "object_CM.h"

#include "object.h"
#include <dolfin.h>
#include <boost/optional.hpp>

namespace punc
{

namespace df = dolfin;

// TBD: Move to Mesh? Or Object?
/**
 * @brief Calculates the surface area of an object
 * @param mesh[in] - df::Mesh 
 * @param bnd[in] - df::MeshFunction 
 * @return the surface area
 */
double surface_area(std::shared_ptr<const df::Mesh> &mesh,
                    df::MeshFunction<std::size_t> &bnd);

/**
 * @brief Boundary condition for the electric potential
 */
class PhiBoundary : public df::Expression
{
public:
    const std::vector<double> &B; ///< A uniform magnetic field
    const std::vector<double> &vd;///< Drift velocity
    std::vector<double> E;        ///< Electric field

    /**
    * @brief Constructor
    * @param B[in] -  the external uniform magnetic field
    * @param vd[in] - the drift velocity
    */
    PhiBoundary(const std::vector<double> &B, const std::vector<double> &vd);

private:
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
};

/**
 * @brief Boundary condition for non-periodic problems
 */
class NonPeriodicBoundary : public df::SubDomain
{
public:
    const std::vector<double> &Ld;
    const std::vector<bool> &periodic;
    NonPeriodicBoundary(const std::vector<double> &Ld,
                        const std::vector<bool> &periodic);

private:
    bool inside(const df::Array<double> &x, bool on_boundary) const;
};

/**
 * @brief Boundary condition for periodic problems
 */
class PeriodicBoundary : public df::SubDomain
{
public:
    const std::vector<double> &Ld;
    const std::vector<bool> &periodic;
    PeriodicBoundary(const std::vector<double> &Ld,
                     const std::vector<bool> &periodic);

    bool inside(const df::Array<double> &x, bool on_boundary) const;

    void map(const df::Array<double> &x, df::Array<double> &y) const;
};

/**
 * @brief Creates a function space in CG1
 * @param mesh[in] - df::Mesh 
 * @param constr[in] Constraint to be imposed for periodic problems
 * @return CG1 function space
 * 
 * @see DG0_space
 */
df::FunctionSpace function_space(std::shared_ptr<const df::Mesh> &mesh,
                                 boost::optional<std::shared_ptr<PeriodicBoundary>> constr = boost::none);

/**
 * @brief Creates a function space in DG0
 * @param mesh[in] - df::Mesh 
 * @return CG1 function space
 * 
 * @see function_space
 */
df::FunctionSpace DG0_space(std::shared_ptr<const df::Mesh> &mesh);

/**
 * @brief Solver for Poisson's equation
 */
class PoissonSolver {
private:
    boost::optional<std::vector<df::DirichletBC> &> ext_bc; /// < Exterior boundaries
    bool remove_null_space;                                 /// < Whether or not to remove null space
    std::shared_ptr<df::PETScKrylovSolver> solver;          /// < Linear algebra solver
    std::shared_ptr<df::Form> a;                            /// < Bilinear form
    std::shared_ptr<df::Form> L;                            /// < Linear form
    df::PETScMatrix A;                                      /// < Stiffness matrix
    df::PETScVector b;                                      /// < Load vector
    std::shared_ptr<df::VectorSpaceBasis> null_space;
    std::size_t num_bcs = 0;                                ///< Number of boundaries

public:
    /**
     * @brief Constructor 
     * @param V                 The function space of rho and phi
     * @param ext_bc            Exterior Dirichlet boundary conditions
     * @param circuit           Circuits between objects
     * @param eps0              Vacuum permittivity in simulation parameters
     *                          (depends on normalization scheme)
     * @param method            Method of linear algebra solver
     * @param preconditioner    Preconditioner for matrix equation
     */
    PoissonSolver(const df::FunctionSpace &V, 
                  boost::optional<std::vector<df::DirichletBC>& > ext_bc = boost::none,
                  // boost::optional<Circuit& > circuit=boost::none,
                  std::shared_ptr<Circuit> circuit = nullptr,
                  double eps0 = 1,
                  bool remove_null_space = false,
                  std::string method = "",
                  std::string preconditioner = "");

    /**
     * @brief Solves Poisson's equation without any internal objects
     * @param    rho               Total charge density
     * @return   The electric potential
     */
    df::Function solve(const df::Function &rho);

    /**
     * @brief Solves Poisson's equation in the domain contaning objects
     * @param    rho               Total charge density
     * @param    objects           A vector of objects
     * @return   The electric potential
     */
    df::Function solve(const df::Function &rho,
                       const std::vector<ObjectCM> &objects);

    /**
     * @brief Solves Poisson's equation in the domain contaning objects
     * @param    rho               Total charge density
     * @param    objects           A vector of objects
     * @param    V                 The function space
     * @return   The electric potential
     */
    df::Function solve(const df::Function &rho,
                       std::vector<ObjectBC> &objects,
                       const df::FunctionSpace &V);

    /**
     * @brief Solves Poisson's equation in the domain contaning objects and circuits
     * @param    rho               Total charge density
     * @param    objects           A vector of objects
     * @param    circuit           The circuitry
     * @param    V                 The function space
     * @return   The electric potential
     */
    df::Function solve(const df::Function &rho,
                       std::vector<ObjectBC> &objects,
                       std::shared_ptr<Circuit> circuit,
                       const df::FunctionSpace &V);

    /**
     * @brief Calculates the residual of the Poisson solution
     * @param    phi     - The solution of Poisson's equation
     * @return   The residual
     * 
     * The residual is given by 
     * \f[
     * \mathbf{r} =\lVert A\mathbf{\phi} - \mathbf{b} \rVert_{L_2}. 
     * /f]
     *
     */
    double residual(const df::Function &phi);

};

/**
     * @brief Calculates the L2 error-norm 
     * @param    phi           Numerical solution
     * @param    phi_e         The exact solution
     * @return   L2 error-norm
     * 
     * The error is measured by using \f[L_2\f] norm, which is defined by 
     * \f[
     * \lVert\mathbf{e}\rVert_{L_2} =\left(\int_{\Omega}\mathbf{e}\cdot\mathbf{e} \,\dd\mathbf{x}\right)^{\frac{1}{2}}. 
    * /f]
    *
    */
double errornorm(const df::Function &phi, const df::Function &phi_e);

} // namespace punc

#endif // POISSON_H
