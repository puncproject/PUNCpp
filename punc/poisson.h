#ifndef POISSON_H
#define POISSON_H

#include <iostream>
#include <dolfin.h>
#include "Potential.h"
#include "EField.h"
#include "object.h"
#include "Error.h"
#include "Errorv2.h"
#include <algorithm>
#include <boost/optional.hpp>

namespace punc
{

namespace df = dolfin;

std::vector<double> get_mesh_size(std::shared_ptr<const df::UnitSquareMesh> &mesh);

std::vector<double> get_mesh_size(std::shared_ptr<const df::Mesh> &mesh);

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
    const std::shared_ptr<Potential::FunctionSpace> &V;
    std::vector<std::shared_ptr<df::DirichletBC> > bc;
    bool remove_null_space;
    df::PETScKrylovSolver solver;
    std::string method;
    std::string preconditioner;
    Potential::BilinearForm a;
    Potential::LinearForm L;
    df::PETScMatrix A;
    df::PETScVector b;
    std::shared_ptr<df::VectorSpaceBasis> null_space;

    PoissonSolver(const std::shared_ptr<Potential::FunctionSpace> &V,
                  bool remove_null_space = false,
                  std::string method = "gmres",
                  std::string preconditioner = "hypre_amg");

    PoissonSolver(const std::shared_ptr<Potential::FunctionSpace> &V,
                  std::shared_ptr<df::DirichletBC> dbc,
                  bool remove_null_space = false,
                  std::string method = "gmres",
                  std::string preconditioner = "hypre_amg");

    PoissonSolver(const std::shared_ptr<Potential::FunctionSpace> &V,
                  std::vector<std::shared_ptr<df::DirichletBC> > bc,
                  bool remove_null_space = false,
                  std::string method = "gmres",
                  std::string preconditioner = "hypre_amg");

    void initialize();

    void solve(std::shared_ptr<df::Function> &phi,
               const std::shared_ptr<df::Function> &rho);

    void solve(std::shared_ptr<df::Function> &phi,
               const std::shared_ptr<df::Function> &rho,
               const Object &bcs);

    void solve(std::shared_ptr<df::Function> &phi,
               const std::shared_ptr<df::Function> &rho,
               const std::vector<Object> &bcs);

    void solve(std::shared_ptr<df::Function> &phi,
               const std::shared_ptr<df::Function> &rho,
               const std::shared_ptr<Object> &bcs);

    void solve(std::shared_ptr<df::Function> &phi,
               const std::shared_ptr<df::Function> &rho,
               const std::vector<std::shared_ptr<Object>> &bcs);

    double residual(const std::shared_ptr<df::Function> &phi);

    double errornormv1(const std::shared_ptr<df::Function> &phi,
                       const std::shared_ptr<df::Function> &phi_h);

    double errornormv2(const std::shared_ptr<df::Function> &phi,
                       const std::shared_ptr<df::Function> &phi_h);
};

std::shared_ptr<df::Function> electric_field(const std::shared_ptr<df::Function> &phi);

}

#endif
