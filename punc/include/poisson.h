#ifndef POISSON_H
#define POISSON_H

#include <dolfin.h>
#include "object.h"
#include "../ufl/Potential1D.h"
#include "../ufl/Potential2D.h"
#include "../ufl/Potential3D.h"
#include "../ufl/VarPotential1D.h"
#include "../ufl/VarPotential2D.h"
#include "../ufl/VarPotential3D.h"
#include "../ufl/EField1D.h"
#include "../ufl/EField2D.h"
#include "../ufl/EField3D.h"
#include "../ufl/ErrorNorm1D.h"
#include "../ufl/ErrorNorm2D.h"
#include "../ufl/ErrorNorm3D.h"
#include "../ufl/ErrorNormVec1D.h"
#include "../ufl/ErrorNormVec2D.h"
#include "../ufl/ErrorNormVec3D.h"
#include "../ufl/Surface.h"
#include "../ufl/Volume.h"
#include <boost/optional.hpp>

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

df::FunctionSpace var_function_space(std::shared_ptr<const df::Mesh> &mesh);

class PoissonSolver
{
public:
    PoissonSolver(const df::FunctionSpace &V,
                  boost::optional<std::vector<df::DirichletBC>& > ext_bc = boost::none,
                  bool remove_null_space = false,
                  std::string method = "cg",
                  std::string preconditioner = "hypre_amg");

    df::Function solve(const df::Function &rho);

    df::Function solve(const df::Function &rho,
                       const std::vector<Object> &objects);

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
          std::string method = "cg",
          std::string preconditioner = "hypre_amg");

  df::Function solve(df::Function &phi);

private:
  df::PETScKrylovSolver solver;
  std::shared_ptr<df::FunctionSpace> W;
  std::shared_ptr<df::Form> a, L;
  df::PETScMatrix A;
  df::PETScVector b;
};

class VarPoissonSolver
{
  public:
    std::shared_ptr<df::FunctionSpace> V;
    std::vector<df::DirichletBC> ext_bc;
    df::PETScKrylovSolver solver;
    std::shared_ptr<df::Form> a, L;
    df::PETScMatrix A;
    df::PETScVector b;
    std::shared_ptr<df::Constant> S;
    VarPoissonSolver(df::FunctionSpace &W,
                     std::vector<df::DirichletBC> &ext_bc,
                     VObject &vobject,
                     std::string method = "tfqmr",
                     std::string preconditioner = "none");

    df::Function solve(const df::Function &rho, double Q, VObject &int_bc);
};

}

#endif
