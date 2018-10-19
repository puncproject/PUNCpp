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

namespace punc
{

std::vector<std::size_t> get_mesh_ids(df::MeshFunction<std::size_t> &boundaries)
{
    auto comm = boundaries.mesh()->mpi_comm();
    auto values = boundaries.values();
    auto length = boundaries.size();
    std::vector<std::size_t> tags(length);

    for (std::size_t i = 0; i < length; ++i)
    {
        tags[i] = values[i];
    }
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    if(df::MPI::size(comm)==1)
    {
        return tags;
    }else{
        std::vector<std::vector<std::size_t>> all_ids;
        df::MPI::all_gather(comm, tags, all_ids);
        std::vector<std::size_t> ids;
        for (const auto &id : all_ids)
        {
            ids.insert(ids.end(), id.begin(), id.end());
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
    }
}

std::vector<double> get_mesh_size(std::shared_ptr<const df::Mesh> &mesh)
{
    auto dim = mesh->geometry().dim();
    auto count = mesh->num_vertices();

    std::vector<double> Ld(dim);
    double X, max;
    for (std::size_t i = 0; i < dim; ++i)
    {
        max = 0.0;
        for (std::size_t j = 0; j < count; ++j)
        {
            X = mesh->geometry().point(j)[i];
            if (X >= max)
            {
                max = X;
            }
        }
        Ld[i] = max;
    }
    return Ld;
}

double volume(std::shared_ptr<const df::Mesh> &mesh)
{
    auto dim = mesh->geometry().dim();
    auto one = std::make_shared<df::Constant>(1.0);
    std::shared_ptr<df::Form> volume_form;
    if (dim == 1)
    {
        volume_form = std::make_shared<Volume::Form_0>(mesh, one);
    }
    else if (dim == 2)
    {
        volume_form = std::make_shared<Volume::Form_1>(mesh, one);
    }
    else if (dim == 3)
    {
        volume_form = std::make_shared<Volume::Form_2>(mesh, one);
    }
    return df::assemble(*volume_form);
}

double surface_area(std::shared_ptr<const df::Mesh> &mesh,
                    df::MeshFunction<std::size_t> &bnd)
{
    auto dim = mesh->geometry().dim();
    auto one = std::make_shared<df::Constant>(1.0);
    std::shared_ptr<df::Form> area;
    if (dim == 1)
    {
        area = std::make_shared<Surface::Form_0>(mesh, one);
    }
    if (dim == 2)
    {
        area = std::make_shared<Surface::Form_1>(mesh, one);
    }
    else if (dim == 3)
    {
        area = std::make_shared<Surface::Form_2>(mesh, one);
    }
    area->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(bnd));
    return df::assemble(*area);
}

df::FunctionSpace function_space(std::shared_ptr<const df::Mesh> &mesh,
                                 boost::optional<std::shared_ptr<PeriodicBoundary>> constr)
{

    std::size_t dim = mesh->geometry().dim();

    if(dim<1 || dim>3)
        df::error("PUNC is programmed for dimensions up to 3D only.");

    if (dim == 1)
    {
        if(constr)
        {
            Potential1D::FunctionSpace V(mesh, constr.get());
            return V;
        }
        else
        {
            Potential1D::FunctionSpace V(mesh);
            return V;
        }
    }
    else if (dim == 2)
    {
        if (constr)
        {
            Potential2D::FunctionSpace V(mesh, constr.get());
            return V;
        }
        else
        {
            Potential2D::FunctionSpace V(mesh);
            return V;
        }
    }
    else
    {
        if (constr)
        {
            Potential3D::FunctionSpace V(mesh, constr.get());
            return V;
        }
        else
        {
            Potential3D::FunctionSpace V(mesh);
            return V;
        }
    }
}

df::FunctionSpace DG0_space(std::shared_ptr<const df::Mesh> &mesh)
{
    std::size_t dim = mesh->geometry().dim();

    if(dim<1 || dim>3)
        df::error("PUNC is programmed for dimensions up to 3D only.");

    if (dim == 1)
    {
        PotentialDG1D::CoefficientSpace_rho Q(mesh);
        return Q;
    }
    else if (dim == 2)
    {
        PotentialDG2D::CoefficientSpace_rho Q(mesh);
        return Q;
    }
    else
    {
        PotentialDG3D::Form_L::CoefficientSpace_rho Q(mesh);
        return Q;
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
                             boost::optional<Circuit& > circuit,
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

    bool has_charge_constraints = circuit && circuit.get().has_charge_constraints();
    if(has_charge_constraints){
        if(method=="" && preconditioner==""){
            method = "bicgstab";
            preconditioner = "ilu";
        } else {
            // FIXME: Write proper status/warning/error message system
            std::cerr << "Some linear algebra solvers/preconditioners may not work for circuits with charge constraints.\n";
        }
    } else {
        if(method=="" && preconditioner==""){
            method = "gmres";
            preconditioner = "hypre_amg";
        }
    }

    solver = std::make_shared<df::PETScKrylovSolver>(V.mesh()->mpi_comm(), method, preconditioner);

    if(ext_bc)
    {
        num_bcs = ext_bc->size();
    }
    
    if(circuit)
    {
        df::PETScMatrix A0;
        df::assemble(A0, *a);
        circuit.get().apply(A0, A);
    }else{
        df::assemble(A, *a);
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

df::Function PoissonSolver::solve(const df::Function &rho)
{
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    df::assemble(b, *L);

    for(std::size_t i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }

    if (remove_null_space)
    {
        null_space->orthogonalize(b);
    }
    df::Function phi(rho.function_space());
    solver->solve(A, *phi.vector(), b);
    return phi;
 }

df::Function PoissonSolver::solve(const df::Function &rho,
                          const std::vector<Object> &objects)
{
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    df::assemble(b, *L);
    for(std::size_t i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }
    for(auto& bc: objects)
    {
        bc.apply(A, b);
    }
    df::Function phi(rho.function_space());
    solver->solve(A, *phi.vector(), b);
    return phi;
}

df::Function PoissonSolver::solve(const df::Function &rho,
                                  std::vector<ObjectBC> &objects,
                                  const df::FunctionSpace &V)
{
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    df::assemble(b, *L);
    for (std::size_t i = 0; i < num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }
    for (auto &bc : objects)
    {
        bc.apply(A);
        bc.apply(b);
    }

    auto V_shared = std::make_shared<df::FunctionSpace>(V);
    df::Function phi(V_shared);
    solver->solve(A, *phi.vector(), b);
    return phi;
}

df::Function PoissonSolver::solve(const df::Function &rho,
                                  std::vector<ObjectBC> &objects,
                                  Circuit &circuit,
                                  const df::FunctionSpace &V)
{
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    df::assemble(b, *L);
    for(std::size_t i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }
    for(auto& bc: objects)
    {
        bc.apply(A);
        bc.apply(b);
    }

    circuit.apply(b);
    auto V_shared = std::make_shared<df::FunctionSpace>(V);
    df::Function phi(V_shared);
    solver->solve(A, *phi.vector(), b);
    return phi;
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


ESolver::ESolver(const df::FunctionSpace &V,
                 std::string method, std::string preconditioner):
                 solver(V.mesh()->mpi_comm(), method, preconditioner)
{
    auto dim = V.mesh()->geometry().dim();
    if (dim == 1)
    {
        W = std::make_shared<EField1D::FunctionSpace>(V.mesh());
        a = std::make_shared<EField1D::BilinearForm>(W, W);
        L = std::make_shared<EField1D::LinearForm>(W);
    }
    else if (dim == 2)
    {
        W = std::make_shared<EField2D::FunctionSpace>(V.mesh());
        a = std::make_shared<EField2D::BilinearForm>(W, W);
        L = std::make_shared<EField2D::LinearForm>(W);
    }
    else if (dim == 3)
    {
        W = std::make_shared<EField3D::FunctionSpace>(V.mesh());
        a = std::make_shared<EField3D::BilinearForm>(W, W);
        L = std::make_shared<EField3D::LinearForm>(W);
    }
    solver.parameters["absolute_tolerance"] = 1e-14;
    solver.parameters["relative_tolerance"] = 1e-12;
    solver.parameters["maximum_iterations"] = 1000;
    solver.set_reuse_preconditioner(true);

    df::assemble(A, *a);
}

df::Function ESolver::solve(df::Function &phi)
{
    L->set_coefficient("phi", std::make_shared<df::Function>(phi));
    df::assemble(b, *L);
    df::Function E(W);
    solver.solve(A, *E.vector(), b);
    return E;
}

EFieldDG0::EFieldDG0(std::shared_ptr<const df::Mesh> mesh)
{
	auto gdim = mesh->geometry().dim();
	if (gdim == 1)
	{
		Q = std::make_shared<EFieldDG01D::FunctionSpace>(mesh);
		M = std::make_shared<EFieldDG01D::LinearForm>(Q);
	}
	else if (gdim == 2)
	{
		Q = std::make_shared<EFieldDG02D::FunctionSpace>(mesh);
		M = std::make_shared<EFieldDG02D::LinearForm>(Q);
	}
	else if (gdim == 3)
	{
		Q = std::make_shared<EFieldDG03D::FunctionSpace>(mesh);
		M = std::make_shared<EFieldDG03D::LinearForm>(Q);
	}
}

df::Function EFieldDG0::solve(const df::Function &phi)
{
	M->set_coefficient("phi", std::make_shared<df::Function>(phi));
	
	df::Function E(Q);
	df::assemble(*E.vector(), *M);
	return E;
}

EFieldMean::EFieldMean(std::shared_ptr<const df::Mesh> mesh, bool arithmetic_mean)
{
    auto gdim = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    std::vector<double> one_vec(gdim, 1.0);
    if (gdim == 1)
    {
        V = std::make_shared<Mean1D::Form_a_FunctionSpace_0>(mesh);
        Q = std::make_shared<Mean1D::Form_a_FunctionSpace_1>(mesh);
        W = std::make_shared<Mean1D::Form_d_FunctionSpace_0>(mesh);
        a = std::make_shared<Mean1D::Form_a>(Q, V);
        c = std::make_shared<Mean1D::Form_c>(Q, V);
        b = std::make_shared<Mean1D::Form_b>(Q);
        d = std::make_shared<Mean1D::Form_d>(W);
    }
    else if (gdim == 2)
    {
        V = std::make_shared<Mean2D::Form_a_FunctionSpace_0>(mesh);
        Q = std::make_shared<Mean2D::Form_a_FunctionSpace_1>(mesh);
        W = std::make_shared<Mean2D::Form_d_FunctionSpace_0>(mesh);
        a = std::make_shared<Mean2D::Form_a>(Q, V);
        c = std::make_shared<Mean2D::Form_c>(Q, V);
        b = std::make_shared<Mean2D::Form_b>(Q);
        d = std::make_shared<Mean2D::Form_d>(W);
    }
    else if (gdim == 3)
    {
        V = std::make_shared<Mean3D::Form_a_FunctionSpace_0>(mesh);
        Q = std::make_shared<Mean3D::Form_a_FunctionSpace_1>(mesh);
        W = std::make_shared<Mean3D::Form_d_FunctionSpace_0>(mesh);
        a = std::make_shared<Mean3D::Form_a>(Q, V);
        c = std::make_shared<Mean3D::Form_c>(Q, V);
        b = std::make_shared<Mean3D::Form_b>(Q);
        d = std::make_shared<Mean3D::Form_d>(W);
    }
    a->set_coefficient("c1", std::make_shared<df::Constant>(tdim + 1.0));
    b->set_coefficient("c2", std::make_shared<df::Constant>(one_vec));

    if (arithmetic_mean)
    {
        df::assemble(A, *c);
    }
    else
    {
        df::assemble(A, *a);
    }

    df::assemble(ones, *b);
    A.mult(ones, Av);
    auto A_vec = Av.vec();
    auto A_mat = A.mat();

    VecReciprocal(A_vec);
    MatDiagonalScale(A_mat, A_vec, NULL);
}

df::Function EFieldMean::mean(const df::Function &phi)
{
    // df::parameters["linear_algebra_backend"] = "PETSc";
    d->set_coefficient("phi", std::make_shared<df::Function>(phi));
    df::assemble(e_dg0, *d);

    df::Function E(V);
    A.mult(e_dg0, *E.vector());
    // E.vector()->update_ghost_values();
    return E;
}

ClementInterpolant::ClementInterpolant(std::shared_ptr<const df::Mesh> mesh)
{
	auto gdim = mesh->geometry().dim();
	auto tdim = mesh->topology().dim();
	if (gdim == 1){
		V = std::make_shared<Clement1D::Form_a_FunctionSpace_0>(mesh); 
		Q = std::make_shared<Clement1D::Form_a_FunctionSpace_1>(mesh);
		a = std::make_shared<Clement1D::BilinearForm>(Q, V);
		b = std::make_shared<Clement1D::LinearForm>(Q);
	}else if (gdim==2){
		V = std::make_shared<Clement2D::Form_a_FunctionSpace_0>(mesh);
		Q = std::make_shared<Clement2D::Form_a_FunctionSpace_1>(mesh);
		a = std::make_shared<Clement2D::BilinearForm>(Q, V);
		b = std::make_shared<Clement2D::LinearForm>(Q);
	} else if (gdim==3){
		V = std::make_shared<Clement3D::Form_a_FunctionSpace_0>(mesh); 
		Q = std::make_shared<Clement3D::Form_a_FunctionSpace_1>(mesh);
		a = std::make_shared<Clement3D::BilinearForm>(Q, V);
		b = std::make_shared<Clement3D::LinearForm>(Q);
	}
	a->set_coefficient("c1", std::make_shared<df::Constant>(tdim+1.0));
	b->set_coefficient("c2", std::make_shared<df::Constant>(1.0));

	df::assemble(A, *a);
	df::assemble(ones, *b);
	A.mult(ones, Av);
	auto A_vec = Av.vec();
	auto A_mat = A.mat();

	VecReciprocal(A_vec);
	MatDiagonalScale(A_mat, A_vec, NULL);
}

df::Function ClementInterpolant::interpolate(const df::Function &u)
{
	df::Function ui(V);
	auto u_vec = u.vector();
	auto ui_vec = ui.vector();
	A.mult(*u_vec, *ui_vec);
	return ui;
}

}
