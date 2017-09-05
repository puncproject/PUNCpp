#include "poisson.h"

namespace punc
{

std::vector<double> get_mesh_size(std::shared_ptr<const df::UnitSquareMesh> &mesh)
{
    auto g_dim = mesh->geometry().dim();
    auto count = mesh->num_vertices();

    std::vector<double> Ld(g_dim);
    double X, max;
    for (std::size_t i = 0; i < g_dim; ++i)
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

std::vector<double> get_mesh_size(std::shared_ptr<const df::Mesh> &mesh)
{
    auto g_dim = mesh->geometry().dim();
    auto count = mesh->num_vertices();

    std::vector<double> Ld(g_dim);
    double X, max;
    for (std::size_t i = 0; i < g_dim; ++i)
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
    bool on_bnd = false;
    for (std::size_t i = 0; i < Ld.size(); ++i)
    {
        on_bnd = on_bnd or df::near(x[i], 0.0) and periodic[i] and not df::near(x[i], Ld[i]);
    }
    return on_bnd;
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

PoissonSolver::PoissonSolver(const std::shared_ptr<Potential::FunctionSpace> &V,
                             bool remove_null_space,
                             std::string method,
                             std::string preconditioner) : V(V), bc({}),
                             remove_null_space(remove_null_space),
                             solver(method, preconditioner), method(method),
                             preconditioner(preconditioner), a(V, V), L(V)
{
    initialize();
}

PoissonSolver::PoissonSolver(const std::shared_ptr<Potential::FunctionSpace> &V,
                             std::shared_ptr<df::DirichletBC> dbc,
                             bool remove_null_space,
                             std::string method,
                             std::string preconditioner) : V(V), bc({dbc}),
                             remove_null_space(remove_null_space),
                             solver(method, preconditioner), method(method),
                             preconditioner(preconditioner), a(V, V), L(V)
{
    initialize();
}

PoissonSolver::PoissonSolver(const std::shared_ptr<Potential::FunctionSpace> &V,
                             std::vector<std::shared_ptr<df::DirichletBC>> bc,
                             bool remove_null_space,
                             std::string method,
                             std::string preconditioner) : V(V), bc(bc),
                             remove_null_space(remove_null_space),
                             solver(method, preconditioner), method(method),
                             preconditioner(preconditioner), a(V, V), L(V)
{
    initialize();
}

void PoissonSolver::initialize()
{
    assemble(A, a);
    for(int i = 0; i<bc.size();++i)
    {
        bc[i]->apply(A);
    }

    solver.parameters["absolute_tolerance"] = 1e-14;
    solver.parameters["relative_tolerance"] = 1e-12;
    solver.parameters["maximum_iterations"] = 1000;
    solver.set_reuse_preconditioner(true);

    if (remove_null_space)
    {
        df::Function u(V);
        auto null_space_vector = u.vector()->copy();
        *null_space_vector = sqrt(1.0 / null_space_vector->size());

        std::vector<std::shared_ptr<df::GenericVector>> basis{null_space_vector};
        null_space = std::make_shared<df::VectorSpaceBasis>(basis);
        A.set_nullspace(*null_space);
    }
}

 void PoissonSolver::solve(std::shared_ptr<df::Function> &phi,
                           const std::shared_ptr<df::Function> &rho)
 {
    L.rho = rho;

    df::assemble(b, L);

    for(int i = 0; i<bc.size();++i)
    {
        bc[i]->apply(A,b);
    }

    if (remove_null_space)
    {
        null_space->orthogonalize(b);
    }

    solver.solve(A, *phi->vector(), b);
 }

void PoissonSolver::solve(std::shared_ptr<df::Function> &phi,
                          const std::shared_ptr<df::Function> &rho,
                          const std::vector<std::shared_ptr<Object>> &bcs)
{
    L.rho = rho;

    df::assemble(b, L);

    for(int i = 0; i<bc.size();++i)
    {
        bc[i]->apply(A,b);
    }

    for (std::size_t i = 0; i < bcs.size(); ++i)
    {
        bcs[i]->apply(A, b);
    }

    if (remove_null_space)
    {
        null_space->orthogonalize(b);
    }

    solver.solve(A, *phi->vector(), b);
}

double PoissonSolver::residual(const std::shared_ptr<df::Function> &phi)
{
    df::Vector residual(*phi->vector());
    A.mult(*phi->vector(), residual);
    residual.axpy(-1.0, b);
    return residual.norm("l2");
}

double PoissonSolver::errornormv1(const std::shared_ptr<df::Function> &phi,
                                  const std::shared_ptr<df::Function> &phi_e)
{
    auto mesh = V->mesh();
    Error::Functional error_form(mesh);
    error_form.phi = phi;
    error_form.phi_e = phi_e;
    return sqrt(df::assemble(error_form));
}

double PoissonSolver::errornormv2(const std::shared_ptr<df::Function> &phi,
                                  const std::shared_ptr<df::Function> &phi_e)
{
    auto mesh = V->mesh();
    auto W = std::make_shared<Errorv2::CoefficientSpace_e>(mesh);
    auto u = std::make_shared<df::Function>(W);
    auto uh = std::make_shared<df::Function>(W);
    auto err = std::make_shared<df::Function>(W);

    u->interpolate(*phi);
    uh->interpolate(*phi_e);

    *err->vector() += *u->vector();
    *err->vector() -= *uh->vector();

    Errorv2::Functional error_form(mesh);
    error_form.e = err;
    return sqrt(df::assemble(error_form));
}

std::shared_ptr<df::Function> electric_field(const std::shared_ptr<df::Function> &phi)
{
    auto mesh = phi->function_space()->mesh();
    auto W = std::make_shared<EField::FunctionSpace>(mesh);
    auto E = std::make_shared<df::Function>(W);

    EField::BilinearForm a(W, W);
    EField::LinearForm L(W);

    L.phi = phi;
    solve(a == L, *E);
    return E;
}

}