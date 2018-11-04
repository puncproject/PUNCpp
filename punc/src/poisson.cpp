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

#include "../include/punc/poisson.h"

#include <dolfin/la/Vector.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/Constant.h>
#include <dolfin/math/basic.h>

#include "../ufl/Potential1D.h"
#include "../ufl/Potential2D.h"
#include "../ufl/Potential3D.h"
#include "../ufl/PotentialDG1D.h"
#include "../ufl/PotentialDG2D.h"
#include "../ufl/PotentialDG3D.h"
#include "../ufl/EField1D.h"
#include "../ufl/EField2D.h"
#include "../ufl/EField3D.h"
#include "../ufl/EFieldDG01D.h"
#include "../ufl/EFieldDG02D.h"
#include "../ufl/EFieldDG03D.h"
#include "../ufl/ErrorNorm1D.h"
#include "../ufl/ErrorNorm2D.h"
#include "../ufl/ErrorNorm3D.h"
#include "../ufl/ErrorNormVec1D.h"
#include "../ufl/ErrorNormVec2D.h"
#include "../ufl/ErrorNormVec3D.h"


namespace punc
{

df::FunctionSpace CG1_space(const Mesh &mesh,
                            boost::optional<std::shared_ptr<PeriodicBoundary>> constr)
{

    std::size_t dim = mesh.dim;

    if(dim<1 || dim>3)
        df::error("PUNC is programmed for dimensions up to 3D only.");

    if (dim == 1)
    {
        if(constr)
        {
            Potential1D::FunctionSpace V(mesh.mesh, constr.get());
            return V;
        }
        else
        {
            Potential1D::FunctionSpace V(mesh.mesh);
            return V;
        }
    }
    else if (dim == 2)
    {
        if (constr)
        {
            Potential2D::FunctionSpace V(mesh.mesh, constr.get());
            return V;
        }
        else
        {
            Potential2D::FunctionSpace V(mesh.mesh);
            return V;
        }
    }
    else
    {
        if (constr)
        {
            Potential3D::FunctionSpace V(mesh.mesh, constr.get());
            return V;
        }
        else
        {
            Potential3D::FunctionSpace V(mesh.mesh);
            return V;
        }
    }
}

df::FunctionSpace CG1_vector_space(const Mesh &mesh)
{

    std::size_t dim = mesh.dim;

    if (dim < 1 || dim > 3)
        df::error("PUNC is programmed for dimensions up to 3D only.");

    if (dim == 1)
    {
        EField1D::FunctionSpace W(mesh.mesh);
        return W;
    }
    else if (dim == 2)
    {
        EField2D::FunctionSpace W(mesh.mesh);
        return W;
    }
    else
    {
        EField3D::FunctionSpace W(mesh.mesh);
        return W;
    }
}

df::FunctionSpace DG0_space(const Mesh &mesh)
{
    std::size_t dim = mesh.dim;

    if(dim<1 || dim>3)
        df::error("PUNC is programmed for dimensions up to 3D only.");

    if (dim == 1)
    {
        PotentialDG1D::CoefficientSpace_rho Q(mesh.mesh);
        return Q;
    }
    else if (dim == 2)
    {
        PotentialDG2D::CoefficientSpace_rho Q(mesh.mesh);
        return Q;
    }
    else
    {
        PotentialDG3D::Form_L::CoefficientSpace_rho Q(mesh.mesh);
        return Q;
    }
}

df::FunctionSpace DG0_vector_space(const Mesh &mesh)
{
    std::size_t dim = mesh.dim;

    if (dim < 1 || dim > 3)
        df::error("PUNC is programmed for dimensions up to 3D only.");

    if (dim == 1)
    {
        EFieldDG01D::FunctionSpace P(mesh.mesh);
        return P;
    }
    else if (dim == 2)
    {
        EFieldDG02D::FunctionSpace P(mesh.mesh);
        return P;
    }
    else
    {
        EFieldDG03D::FunctionSpace P(mesh.mesh);
        return P;
    }
}

PhiBoundary::PhiBoundary(const std::vector<double> &B,
                         const std::vector<double> &vd)
                         :B(B), vd(vd)
{
    std::vector<double> E(vd.size());
    E[0] = -(vd[1] * B[2] - vd[2] * B[1]);
    E[1] = -(vd[2] * B[0] - vd[0] * B[2]);
    E[2] = -(vd[0] * B[1] - vd[1] * B[0]);
    this->E = E;
}

void PhiBoundary::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    values[0] = 0.0;
    for (std::size_t i = 0; i < E.size(); ++i)
    {
        values[0] += x[i] * E[i];
    }
}

std::vector<df::DirichletBC> exterior_bc(const df::FunctionSpace &V, 
                                         const Mesh &mesh,
                                         const std::vector<double> &vd,
                                         const std::vector<double> &B)
{
    auto V_shared = std::make_shared<df::FunctionSpace>(V);
    
    auto phi_bc = std::make_shared<PhiBoundary>(B, vd);

    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V), phi_bc,
                       std::make_shared<df::MeshFunction<size_t>>(mesh.bnd), 
                       mesh.ext_bnd_id);

    std::vector<df::DirichletBC> ext_bc = {bc};
    return ext_bc;
}

NonPeriodicBoundary::NonPeriodicBoundary(const std::vector<double> &Ld,
                                         const std::vector<bool> &periodic)
                                         :df::SubDomain(), Ld(Ld), periodic(periodic){}

bool NonPeriodicBoundary::inside(const df::Array<double> &x, bool on_boundary) const
{
    bool on_bnd = false;
    for (std::size_t i = 0; i < Ld.size(); ++i)
    {
        if (!periodic[i])
        {
            on_bnd = on_bnd or df::near(x[i], 0.0) or df::near(x[i], Ld[i]);
        }
    }
    return on_bnd and on_boundary;
}

PeriodicBoundary::PeriodicBoundary(const std::vector<double> &Ld,
                                   const std::vector<bool> &periodic)
                                   :df::SubDomain(), Ld(Ld), periodic(periodic){}


bool PeriodicBoundary::inside(const df::Array<double> &x, bool on_boundary) const
{
    std::vector<bool> tmp1(Ld.size());
    std::vector<bool> tmp2(Ld.size());
    for (std::size_t i = 0; i < Ld.size(); ++i)
    {
        tmp1[i] = df::near(x[i], 0.0) && periodic[i];
        tmp2[i] = df::near(x[i], Ld[i]);
    }
    bool on_bnd = true;
    bool lower_bnd = std::find(std::begin(tmp1), std::end(tmp1), on_bnd) != std::end(tmp1);
    bool upper_bnd = std::find(std::begin(tmp2), std::end(tmp2), on_bnd) != std::end(tmp2);
    return on_boundary && lower_bnd && !upper_bnd;
}

void PeriodicBoundary::map(const df::Array<double> &x, df::Array<double> &y) const
{
    for (std::size_t i = 0; i < Ld.size(); ++i)
    {
        if (df::near(x[i], Ld[i]) && periodic[i])
        {
            y[i] = x[i] - Ld[i];
        }
        else
        {
            y[i] = x[i];
        }
    }
}

PoissonSolver::PoissonSolver(const df::FunctionSpace &V, 
                             boost::optional<std::vector<df::DirichletBC>& > ext_bc,
                             std::shared_ptr<Circuit> circuit,
                             double eps0,
                             bool remove_null_space,
                             std::string method,
                             std::string preconditioner) : 
                             ext_bc(ext_bc),
                             remove_null_space(remove_null_space)
{
    auto dim = V.mesh()->geometry().dim();
    auto eps0_ = std::make_shared<df::Constant>(eps0);
    auto V_shared = std::make_shared<df::FunctionSpace>(V);
    if (dim == 1)
    {
        a = std::make_shared<Potential1D::BilinearForm>(V_shared, V_shared, eps0_);
        L = std::make_shared<Potential1D::LinearForm>(V_shared);
    }
    else if (dim == 2)
    {
        a = std::make_shared<Potential2D::BilinearForm>(V_shared, V_shared, eps0_);
        L = std::make_shared<Potential2D::LinearForm>(V_shared);
    }
    else if (dim == 3)
    {
        a = std::make_shared<Potential3D::BilinearForm>(V_shared, V_shared, eps0_);
        L = std::make_shared<Potential3D::LinearForm>(V_shared);
    }

    if(circuit){
        // Assigns default solvers if method==preconditioner==""
        bool ok = circuit->check_solver_methods(method, preconditioner);
        if(!ok){
            std::cout << "Warning: " << method << "/" << preconditioner;
            std::cout << " solver may not converge for this circuit\n";
        }
    } else {
        if(method=="" && preconditioner==""){
            method = "gmres";
            preconditioner = "hypre_amg";
        }
    }

    std::cout << method << "/" << preconditioner << "\n";

    solver = std::make_shared<df::PETScKrylovSolver>(V.mesh()->mpi_comm(), method, preconditioner);

    if(ext_bc){
        num_bcs = ext_bc->size();
    }
    
    df::assemble(A, *a);

    if(circuit){
        circuit->apply(A);
    }

    for (std::size_t i = 0; i < num_bcs; ++i)
    {
        ext_bc.get()[i].apply(A);
    }

    solver->parameters["absolute_tolerance"] = 1e-14;
    solver->parameters["relative_tolerance"] = 1e-12;
    solver->parameters["maximum_iterations"] = 1000;
    solver->set_reuse_preconditioner(true);

    if (remove_null_space)
    {
        df::Function u(std::make_shared<df::FunctionSpace>(V));
        auto null_space_vector = u.vector()->copy();
        *null_space_vector = sqrt(1.0 / null_space_vector->size());

        std::vector<std::shared_ptr<df::GenericVector>> basis{null_space_vector};
        null_space = std::make_shared<df::VectorSpaceBasis>(basis);
        A.set_nullspace(*null_space);
    }
}

void PoissonSolver::solve(df::Function &phi, const df::Function &rho,
                          ObjectVector &objects,
                          std::shared_ptr<Circuit> circuit)
{
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    df::assemble(b, *L);

    for(std::size_t i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }
    for(auto& bc: objects)
    {
        bc->apply(A);
        bc->apply(b);
    }
    if(circuit) circuit->apply(b);
    if(remove_null_space) null_space->orthogonalize(b);

    solver->solve(A, *phi.vector(), b);
}

void PoissonSolver::solve_circuit(df::Function &phi, const df::Function &rho,
                                  Mesh &mesh,
                                  ObjectVector &objects,
                                  std::shared_ptr<Circuit> circuit)
{
    circuit->pre_solve();
    solve(phi, rho, objects, circuit);
    circuit->post_solve(phi, mesh);

    if(circuit->correction_required){
        solve(phi, rho, objects, circuit);
    }
}

double PoissonSolver::residual(const df::Function &phi)
{
    auto phi_vec = phi.vector();
    df::Vector residual(*phi_vec);
    A.mult(*phi_vec, residual);
    residual.axpy(-1.0, b);
    return residual.norm("l2");
}

double errornorm(const df::Function &phi, const df::Function &phi_e)
{
    auto mesh = phi.function_space()->mesh();
    auto dim = mesh->geometry().dim();
    std::shared_ptr<df::FunctionSpace> W;
    std::shared_ptr<df::Form> error;
    if (dim == 1)
    {
        if (phi.value_rank()==0)
        {
            W = std::make_shared<ErrorNorm1D::CoefficientSpace_e>(mesh);
            error = std::make_shared<ErrorNorm1D::Form_a>(mesh);
        }
        else if (phi.value_rank() == 1)
        {
            W = std::make_shared<ErrorNormVec1D::CoefficientSpace_e>(mesh);
            error = std::make_shared<ErrorNormVec1D::Form_a>(mesh);
        }
    }
    else if (dim == 2)
    {
        if (phi.value_rank() == 0)
        {
            W = std::make_shared<ErrorNorm2D::CoefficientSpace_e>(mesh);
            error = std::make_shared<ErrorNorm2D::Form_a>(mesh);
        }
        else if (phi.value_rank() == 1)
        {
            W = std::make_shared<ErrorNormVec2D::CoefficientSpace_e>(mesh);
            error = std::make_shared<ErrorNormVec2D::Form_a>(mesh);
        }
    }
    else if (dim == 3)
    {
        if (phi.value_rank() == 0)
        {
            W = std::make_shared<ErrorNorm3D::CoefficientSpace_e>(mesh);
            error = std::make_shared<ErrorNorm3D::Form_a>(mesh);
        }
        else if (phi.value_rank() == 1)
        {
            W = std::make_shared<ErrorNormVec3D::CoefficientSpace_e>(mesh);
            error = std::make_shared<ErrorNormVec3D::Form_a>(mesh);
        }
    }
    auto u = std::make_shared<df::Function>(W);
    auto uh = std::make_shared<df::Function>(W);
    auto e = std::make_shared<df::Function>(W);

    u->interpolate(phi);
    uh->interpolate(phi_e);

    *e->vector() += *u->vector();
    *e->vector() -= *uh->vector();

    error->set_coefficient("e", e);

    return std::sqrt(df::assemble(*error));
}


} // namespace punc
