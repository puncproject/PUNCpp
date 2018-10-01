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

#ifndef POISSON_H
#define POISSON_H

#include <dolfin.h>
#include "object.h"
#include <boost/optional.hpp>
#include <petscvec.h>

namespace punc
{

namespace df = dolfin;

std::shared_ptr<const df::Mesh> load_mesh(std::string fname);

df::MeshFunction<std::size_t> load_boundaries(std::shared_ptr<const df::Mesh> mesh, std::string fname);

std::shared_ptr<const df::Mesh> load_h5_mesh(std::string fname);

df::MeshFunction<std::size_t> load_h5_boundaries(std::shared_ptr<const df::Mesh> &mesh, std::string fname);

std::vector<std::size_t> get_mesh_ids(df::MeshFunction<std::size_t> &boundaries);

std::vector<double> get_mesh_size(std::shared_ptr<const df::Mesh> &mesh);

double volume(std::shared_ptr<const df::Mesh> &mesh);

double surface_area(std::shared_ptr<const df::Mesh> &mesh,
                    df::MeshFunction<std::size_t> &bnd);

class PhiBoundary : public df::Expression
{
public:
    const std::vector<double> &B;
    const std::vector<double> &vd;
    std::vector<double> E;

    PhiBoundary(const std::vector<double> &B, const std::vector<double> &vd);

private:
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
};

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

df::FunctionSpace function_space(std::shared_ptr<const df::Mesh> &mesh,
                                 boost::optional<std::shared_ptr<PeriodicBoundary>> constr = boost::none);

df::FunctionSpace DG0_space(std::shared_ptr<const df::Mesh> &mesh);

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
                  boost::optional<Circuit& > circuit=boost::none,
                  double eps0 = 1,
                  bool remove_null_space = false,
                  std::string method = "",
                  std::string preconditioner = "");

    df::Function solve(const df::Function &rho);

    df::Function solve(const df::Function &rho,
                       const std::vector<Object> &objects);

    df::Function solve(const df::Function &rho,
                       std::vector<ObjectBC> &objects,
                       Circuit &circuit,
                       const df::FunctionSpace &V);

    double residual(const std::shared_ptr<df::Function> &phi);

};

double errornorm(const df::Function &phi, const df::Function &phi_e);

class ESolver
{
public:
  ESolver(const df::FunctionSpace &V,
          std::string method = "gmres",
          std::string preconditioner = "hypre_amg");

  df::Function solve(df::Function &phi);

private:
  df::PETScKrylovSolver solver;
  std::shared_ptr<df::FunctionSpace> W;
  std::shared_ptr<df::Form> a, L;
  df::PETScMatrix A;
  df::PETScVector b;
};

class EFieldDG0
{
public:
	std::shared_ptr<df::FunctionSpace> Q;
	EFieldDG0(std::shared_ptr<const df::Mesh> mesh);
	df::Function solve(const df::Function &phi);

private:
	std::shared_ptr<df::Form> M;
};

class EFieldMean
{
  public:
    std::shared_ptr<df::FunctionSpace> V;
    EFieldMean(std::shared_ptr<const df::Mesh> mesh, bool arithmetic_mean = false);
    df::Function mean(const df::Function &phi);

  private:
    std::shared_ptr<df::FunctionSpace> Q, W;
    std::shared_ptr<df::Form> a, b, c, d;
    df::PETScMatrix A;
    df::PETScVector ones, Av, e_dg0;
};

class ClementInterpolant
{
  public:
    std::shared_ptr<df::FunctionSpace> V;
    std::shared_ptr<df::FunctionSpace> Q;
    ClementInterpolant(std::shared_ptr<const df::Mesh> mesh);
    df::Function interpolate(const df::Function &u);

  private:
    std::shared_ptr<df::Form> a, b;
    df::PETScMatrix A;
    df::PETScVector ones, Av;
};

}

#endif // POISSON_H
