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

#include "../include/punc/efield.h"
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

namespace punc
{

ESolver::ESolver(const df::FunctionSpace &W,
                 std::string method, std::string preconditioner):
                 solver(W.mesh()->mpi_comm(), method, preconditioner)
{
    auto dim = W.mesh()->geometry().dim();
    auto W_shared = std::make_shared<df::FunctionSpace>(W);
    if (dim == 1)
    {
        a = std::make_shared<EField1D::BilinearForm>(W_shared, W_shared);
        L = std::make_shared<EField1D::LinearForm>(W_shared);
    }
    else if (dim == 2)
    {
        a = std::make_shared<EField2D::BilinearForm>(W_shared, W_shared);
        L = std::make_shared<EField2D::LinearForm>(W_shared);
    }
    else if (dim == 3)
    {
        a = std::make_shared<EField3D::BilinearForm>(W_shared, W_shared);
        L = std::make_shared<EField3D::LinearForm>(W_shared);
    }
    solver.parameters["absolute_tolerance"] = 1e-14;
    solver.parameters["relative_tolerance"] = 1e-12;
    solver.parameters["maximum_iterations"] = 1000;
    solver.set_reuse_preconditioner(true);

    df::assemble(A, *a);
}

void ESolver::solve(df::Function &E, const df::Function &phi)
{
    L->set_coefficient("phi", std::make_shared<df::Function>(phi));
    df::assemble(b, *L);
    solver.solve(A, *E.vector(), b);
}

EFieldDG0::EFieldDG0(const df::FunctionSpace &P)
{
    auto dim = P.mesh()->geometry().dim();
    auto P_shared = std::make_shared<df::FunctionSpace>(P);
	if (dim == 1)
	{
        M = std::make_shared<EFieldDG01D::LinearForm>(P_shared);
    }
	else if (dim == 2)
	{
        M = std::make_shared<EFieldDG02D::LinearForm>(P_shared);
    }
	else if (dim == 3)
	{
        M = std::make_shared<EFieldDG03D::LinearForm>(P_shared);
    }
}

void EFieldDG0::solve(df::Function &E, const df::Function &phi)
{
	M->set_coefficient("phi", std::make_shared<df::Function>(phi));
	df::assemble(*E.vector(), *M);
}

EFieldMean::EFieldMean(const df::FunctionSpace &P, const df::FunctionSpace &W, bool arithmetic_mean)
{
    auto gdim = P.mesh()->geometry().dim();
    auto tdim = P.mesh()->topology().dim();

    std::vector<double> one_vec(gdim, 1.0);

    auto P_shared = std::make_shared<df::FunctionSpace>(P);
    auto W_shared = std::make_shared<df::FunctionSpace>(W);
    if (gdim == 1)
    {
        a = std::make_shared<Mean1D::Form_a>(P_shared, W_shared);
        c = std::make_shared<Mean1D::Form_c>(P_shared, W_shared);
        b = std::make_shared<Mean1D::Form_b>(P_shared);
        d = std::make_shared<Mean1D::Form_d>(P_shared);
    }
    else if (gdim == 2)
    {
        a = std::make_shared<Mean2D::Form_a>(P_shared, W_shared);
        c = std::make_shared<Mean2D::Form_c>(P_shared, W_shared);
        b = std::make_shared<Mean2D::Form_b>(P_shared);
        d = std::make_shared<Mean2D::Form_d>(P_shared);
    }
    else if (gdim == 3)
    {
        a = std::make_shared<Mean3D::Form_a>(P_shared, W_shared);
        c = std::make_shared<Mean3D::Form_c>(P_shared, W_shared);
        b = std::make_shared<Mean3D::Form_b>(P_shared);
        d = std::make_shared<Mean3D::Form_d>(P_shared);
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

void EFieldMean::mean(df::Function &E, const df::Function &phi)
{
    // df::parameters["linear_algebra_backend"] = "PETSc";
    d->set_coefficient("phi", std::make_shared<df::Function>(phi));
    df::assemble(e_dg0, *d);

    A.mult(e_dg0, *E.vector());
    // E.vector()->update_ghost_values();
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

} // namespace punc
