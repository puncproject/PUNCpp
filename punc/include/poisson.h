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
#include "../ufl/Potential1D.h"
#include "../ufl/Potential2D.h"
#include "../ufl/Potential3D.h"
#include "../ufl/PotentialDG1D.h"
#include "../ufl/PotentialDG2D.h"
#include "../ufl/PotentialDG3D.h"
#include "../ufl/EField1D.h"
#include "../ufl/EField2D.h"
#include "../ufl/EField3D.h"
#include "../ufl/Clement1D.h"
#include "../ufl/Clement2D.h"
#include "../ufl/Clement3D.h"
#include "../ufl/Mean1D.h"
#include "../ufl/Mean2D.h"
#include "../ufl/Mean3D.h"
#include "../ufl/EFieldDG01D.h"
#include "../ufl/EFieldDG02D.h"
#include "../ufl/EFieldDG03D.h"
#include "../ufl/ErrorNorm1D.h"
#include "../ufl/ErrorNorm2D.h"
#include "../ufl/ErrorNorm3D.h"
#include "../ufl/ErrorNormVec1D.h"
#include "../ufl/ErrorNormVec2D.h"
#include "../ufl/ErrorNormVec3D.h"
#include "../ufl/Surface.h"
#include "../ufl/Volume.h"
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

class PoissonSolver
{
public:
    PoissonSolver(const df::FunctionSpace &V, 
                  boost::optional<std::vector<df::DirichletBC>& > ext_bc = boost::none,
                  boost::optional<Circuit& > circuit=boost::none,
                  bool remove_null_space = false,
                  std::string method = "gmres",
                  std::string preconditioner = "hypre_amg");

    df::Function solve(const df::Function &rho);

    df::Function solve(const df::Function &rho,
                       const std::vector<Object> &objects);

    df::Function solve(const df::Function &rho,
                       std::vector<ObjectBC> &objects,
                       Circuit &circuit,
                       const df::FunctionSpace &V);

    double residual(const std::shared_ptr<df::Function> &phi);

private:
    boost::optional<std::vector<df::DirichletBC> &> ext_bc;
    bool remove_null_space;
    df::PETScKrylovSolver solver;
    std::shared_ptr<df::Form> a, L;
    df::PETScMatrix A;
    df::PETScVector b;
    std::shared_ptr<df::VectorSpaceBasis> null_space;
    std::size_t num_bcs = 0;
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

#endif
