#ifndef POISSON_H
#define POISSON_H

#include <dolfin.h>
#include "object.h"
#include "Potential1D.h"
#include "Potential2D.h"
#include "Potential3D.h"
#include "VarPotential1D.h"
#include "VarPotential2D.h"
#include "VarPotential3D.h"
#include "EField1D.h"
#include "EField2D.h"
#include "EField3D.h"
#include "ErrorNorm1D.h"
#include "ErrorNorm2D.h"
#include "ErrorNorm3D.h"
#include "Surface.h"
#include "Volume.h"
#include <boost/optional.hpp>

namespace punc
{

namespace df = dolfin;


std::shared_ptr<const df::Mesh> load_mesh(std::string fname);

std::shared_ptr<df::MeshFunction<std::size_t>> load_boundaries(std::shared_ptr<const df::Mesh> mesh, std::string fname);

df::Mesh load_h5_mesh(std::string fname);

df::MeshFunction<std::size_t> load_h5_boundaries(std::shared_ptr<const df::Mesh> &mesh, std::string fname);

std::vector<std::size_t> get_mesh_ids(std::shared_ptr<df::MeshFunction<std::size_t>> boundaries);

std::vector<double> get_mesh_size(std::shared_ptr<const df::UnitSquareMesh> &mesh);

std::vector<double> get_mesh_size(std::shared_ptr<const df::Mesh> &mesh);

double volume(std::shared_ptr<const df::Mesh> &mesh);

double surface_area(std::shared_ptr<const df::Mesh> &mesh,
                    std::shared_ptr<df::MeshFunction<std::size_t>> &bnd);

std::shared_ptr<df::FunctionSpace> function_space(std::shared_ptr<const df::Mesh> &mesh);

std::shared_ptr<df::FunctionSpace> var_function_space(std::shared_ptr<const df::Mesh> &mesh);

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

class PoissonSolver
{
public:
    PoissonSolver(std::shared_ptr<df::FunctionSpace> &V,
                  boost::optional<std::vector<df::DirichletBC>& > bc = boost::none,
                  bool remove_null_space = false,
                  std::string method = "cg",
                  std::string preconditioner = "petsc_amg");

    std::shared_ptr<df::Function> solve(std::shared_ptr<df::Function> &rho);

    std::shared_ptr<df::Function> solve(std::shared_ptr<df::Function> &rho,
                                        const std::vector<Object> &objects);

    double residual(std::shared_ptr<df::Function> &phi);

private:
    boost::optional<std::vector<df::DirichletBC>& > ext_bc;
    bool remove_null_space;
    df::PETScKrylovSolver solver;
    std::shared_ptr<df::Form> a, L;
    df::PETScMatrix A;
    df::PETScVector b;
    std::shared_ptr<df::VectorSpaceBasis> null_space;
    std::size_t num_bcs = 0;
};

double errornorm(std::shared_ptr<df::Function> &phi,
                 std::shared_ptr<df::Function> &phi_h);

class ESolver
{
public:
  ESolver(std::shared_ptr<df::FunctionSpace> &V,
          std::string method = "cg",
          std::string preconditioner = "hypre_amg");

  std::shared_ptr<df::Function> solve(std::shared_ptr<df::Function> &phi);

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
    std::shared_ptr<df::FunctionSpace> &V;
    std::vector<df::DirichletBC> ext_bc;
    df::PETScKrylovSolver solver;
    std::shared_ptr<df::Form> a, L;
    df::PETScMatrix A;
    df::PETScVector b;
    std::shared_ptr<df::Constant> S;
    VarPoissonSolver(std::shared_ptr<df::FunctionSpace> &V,
                     std::vector<df::DirichletBC> &ext_bc,
                     VObject &vobject,
                     std::string method = "tfqmr",
                     std::string preconditioner = "none");

    df::Function solve(std::shared_ptr<df::Function> &rho, double Q,
                                        VObject &int_bc);
};

}

#endif
