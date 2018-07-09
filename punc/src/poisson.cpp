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

#include "../include/poisson.h"

namespace punc
{

std::shared_ptr<const df::Mesh> load_mesh(std::string fname)
{
    auto mesh = std::make_shared<const df::Mesh>(fname + ".xml");
    return mesh;
}

df::MeshFunction<std::size_t> load_boundaries(std::shared_ptr<const df::Mesh> mesh, std::string fname)
{
    df::MeshFunction<std::size_t> boundaries(mesh, fname + "_facet_region.xml");

    return boundaries;
}

std::shared_ptr<const df::Mesh> load_h5_mesh(std::string fname)
{
    df::Mesh mesh(MPI_COMM_WORLD);
    df::HDF5File hdf(MPI_COMM_WORLD, fname + ".h5", "r");
    hdf.read(mesh, "/mesh", false);
    return std::make_shared<const df::Mesh>(mesh);
}

df::MeshFunction<std::size_t> load_h5_boundaries(std::shared_ptr<const df::Mesh> &mesh, std::string fname)
{
    auto comm = mesh->mpi_comm();
    df::HDF5File hdf(comm, fname + ".h5", "r");
    df::MeshFunction<std::size_t> boundaries(mesh);
    hdf.read(boundaries, "/boundaries");
    return boundaries;
}

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
    if (mesh->geometry().dim() == 1)
    {
        if(constr)
        {
            Potential1D::FunctionSpace V(mesh, constr.get());
            return V;
        }else{
            Potential1D::FunctionSpace V(mesh);
            return V;
        }
    }
    else if (mesh->geometry().dim() == 2)
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
    else if (mesh->geometry().dim() == 3)
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
    else
        df::error("PUNC is programmed for dimensions up to 3D only.");

}

df::FunctionSpace var_function_space(std::shared_ptr<const df::Mesh> &mesh)
{
    if (mesh->geometry().dim() == 1)
    {
        VarPotential1D::FunctionSpace V(mesh);
        return V;
    }
    else if (mesh->geometry().dim() == 2)
    {
        VarPotential2D::FunctionSpace V(mesh);
        return V;
    }
    else if (mesh->geometry().dim() == 3)
    {
        VarPotential3D::FunctionSpace V(mesh);
        return V;
    }
    else
        df::error("PUNC is programmed for dimensions up to 3D only.");
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
                             bool remove_null_space,
                             std::string method,
                             std::string preconditioner) : ext_bc(ext_bc),
                             remove_null_space(remove_null_space),
                             solver(V.mesh()->mpi_comm(), method, preconditioner)
{
    auto dim = V.mesh()->geometry().dim();
    auto V_shared = std::make_shared<df::FunctionSpace>(V);
    if (dim == 1)
    {
        a = std::make_shared<Potential1D::BilinearForm>(V_shared, V_shared);
        L = std::make_shared<Potential1D::LinearForm>(V_shared);
    }
    else if (dim == 2)
    {
        a = std::make_shared<Potential2D::BilinearForm>(V_shared, V_shared);
        L = std::make_shared<Potential2D::LinearForm>(V_shared);
    }
    else if (dim == 3)
    {
        a = std::make_shared<Potential3D::BilinearForm>(V_shared, V_shared);
        L = std::make_shared<Potential3D::LinearForm>(V_shared);
    }

    df::assemble(A, *a);
    if(ext_bc)
    {
        num_bcs = ext_bc->size();
    }
    for(auto i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(A);
    }

    solver.parameters["absolute_tolerance"] = 1e-14;
    solver.parameters["relative_tolerance"] = 1e-12;
    solver.parameters["maximum_iterations"] = 1000;
    solver.set_reuse_preconditioner(true);

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

    for(auto i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }

    if (remove_null_space)
    {
        null_space->orthogonalize(b);
    }
    df::Function phi(rho.function_space());
    solver.solve(A, *phi.vector(), b);
    return phi;
 }

df::Function PoissonSolver::solve(const df::Function &rho,
                          const std::vector<Object> &objects)
{
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    df::assemble(b, *L);
    for(auto i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }
    for(auto& bc: objects)
    {
        bc.apply(A, b);
    }
    df::Function phi(rho.function_space());
    solver.solve(A, *phi.vector(), b);
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

VarPoissonSolver::VarPoissonSolver(df::FunctionSpace &W,
                                   std::vector<df::DirichletBC> &ext_bc,
                                   VObject &vobject,
                                   std::string method,
                                   std::string preconditioner) :
                                   V(std::make_shared<df::FunctionSpace>(W)),
                                   ext_bc(ext_bc),
                                   solver(method, preconditioner)
{
    auto dim = V->mesh()->geometry().dim();
    if (dim == 1)
    {
        a = std::make_shared<VarPotential1D::BilinearForm>(V, V);
        L = std::make_shared<VarPotential1D::LinearForm>(V);
    }
    else if (dim == 2)
    {
        a = std::make_shared<VarPotential2D::BilinearForm>(V, V);
        L = std::make_shared<VarPotential2D::LinearForm>(V);
    }
    else if (dim == 3)
    {
        a = std::make_shared<VarPotential3D::BilinearForm>(V, V);
        L = std::make_shared<VarPotential3D::LinearForm>(V);
    }
    a->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(vobject.bnd));
    L->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(vobject.bnd));

    df::assemble(A, *a);
    for (auto &bc : ext_bc)
    {
        bc.apply(A);
    }
    auto mesh = V->mesh();
    auto surface = surface_area(mesh, vobject.bnd);
    S = std::make_shared<df::Constant>(surface);
    L->set_coefficient("S", S);

    solver.parameters["absolute_tolerance"] = 1e-14;
    solver.parameters["relative_tolerance"] = 1e-12;
    solver.parameters["maximum_iterations"] = 100000;
    solver.set_reuse_preconditioner(true);
}

df::Function VarPoissonSolver::solve(const df::Function &rho, double Q, VObject &int_bc)
{
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    L->set_coefficient("Q", std::make_shared<df::Constant>(Q));

    df::assemble(b, *L);
    for (auto &bc : ext_bc)
    {
        bc.apply(b);
    }

    int_bc.apply(b);
    int_bc.apply(A);

    df::Function wh(V);
    solver.solve(A, *wh.vector(), b);
    return wh[0];
}

}
