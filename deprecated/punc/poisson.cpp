#include "poisson.h"

namespace punc
{

std::shared_ptr<const df::Mesh> load_mesh(std::string fname)
{
    auto mesh = std::make_shared<const df::Mesh>(fname + ".xml");
    return mesh;
}

std::shared_ptr<df::MeshFunction<std::size_t>> load_boundaries(std::shared_ptr<const df::Mesh> mesh, std::string fname)
{
    auto boundaries = std::make_shared<df::MeshFunction<std::size_t>>(mesh, fname + "_facet_region.xml");
    return boundaries;
}

df::Mesh load_h5_mesh(std::string fname)
{
    MPI_Comm comm;
    df::Mesh mesh(comm);
    df::HDF5File hdf(comm, fname + ".h5", "r");
    hdf.read(mesh, "/mesh", false);
    return mesh;
}

df::MeshFunction<std::size_t> load_h5_boundaries(std::shared_ptr<const df::Mesh> &mesh, std::string fname)
{
    auto comm = mesh->mpi_comm();
    df::HDF5File hdf(comm, fname + ".h5", "r");
    df::MeshFunction<std::size_t> boundaries(mesh);
    hdf.read(boundaries, "/boundaries");
    return boundaries;
}

std::vector<std::size_t> get_mesh_ids(std::shared_ptr<df::MeshFunction<std::size_t>> boundaries)
{
    auto values = boundaries->values();
    auto length = boundaries->size();
    std::vector<std::size_t> tags(length);

    for (std::size_t i = 0; i < length; ++i)
    {
        tags[i] = values[i];
    }
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    return tags;
}

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
                    std::shared_ptr<df::MeshFunction<std::size_t>> &bnd)
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
    area->set_exterior_facet_domains(bnd);
    return df::assemble(*area);
}

std::shared_ptr<df::FunctionSpace> function_space(std::shared_ptr<const df::Mesh> &mesh)
{
    std::shared_ptr<df::FunctionSpace> V;
    if (mesh->geometry().dim() == 1)
    {
        V = std::make_shared<Potential1D::FunctionSpace>(mesh);
    }
    else if (mesh->geometry().dim() == 2)
    {
        V = std::make_shared<Potential2D::FunctionSpace>(mesh);
    }
    else if (mesh->geometry().dim() == 3)
    {
        V = std::make_shared<Potential3D::FunctionSpace>(mesh);
    }
    else
        df::error("PUNC is programmed for dimensions up to 3D only.");

    return V;
}

std::shared_ptr<df::FunctionSpace> var_function_space(std::shared_ptr<const df::Mesh> &mesh)
{
    std::shared_ptr<df::FunctionSpace> V;
    if (mesh->geometry().dim() == 1)
    {
        V = std::make_shared<VarPotential1D::FunctionSpace>(mesh);
    }
    else if (mesh->geometry().dim() == 2)
    {
        V = std::make_shared<VarPotential2D::FunctionSpace>(mesh);
    }
    else if (mesh->geometry().dim() == 3)
    {
        V = std::make_shared<VarPotential3D::FunctionSpace>(mesh);
    }
    else
        df::error("PUNC is programmed for dimensions up to 3D only.");

    return V;
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

PoissonSolver::PoissonSolver(std::shared_ptr<df::FunctionSpace> &V,
                             boost::optional<std::vector<df::DirichletBC>& > bc,
                             bool remove_null_space,
                             std::string method,
                             std::string preconditioner) : ext_bc(bc),
                             remove_null_space(remove_null_space),
                             solver(method, preconditioner)
{
    auto dim = V->mesh()->geometry().dim();
    if (dim == 1)
    {
        a = std::make_shared<Potential1D::BilinearForm>(V, V);
        L = std::make_shared<Potential1D::LinearForm>(V);
    }
    else if (dim == 2)
    {
        a = std::make_shared<Potential2D::BilinearForm>(V, V);
        L = std::make_shared<Potential2D::LinearForm>(V);
    }
    else if (dim == 3)
    {
        a = std::make_shared<Potential3D::BilinearForm>(V, V);
        L = std::make_shared<Potential3D::LinearForm>(V);
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
        df::Function u(V);
        auto null_space_vector = u.vector()->copy();
        *null_space_vector = sqrt(1.0 / null_space_vector->size());

        std::vector<std::shared_ptr<df::GenericVector>> basis{null_space_vector};
        null_space = std::make_shared<df::VectorSpaceBasis>(basis);
        A.set_nullspace(*null_space);
    }
}

std::shared_ptr<df::Function> PoissonSolver::solve(std::shared_ptr<df::Function> &rho)
 {
    L->set_coefficient("rho", rho);
    df::assemble(b, *L);

    for(auto i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }

    if (remove_null_space)
    {
        null_space->orthogonalize(b);
    }
    auto phi = std::make_shared<df::Function>(rho->function_space());
    solver.solve(A, *phi->vector(), b);
    return phi;
 }

std::shared_ptr<df::Function> PoissonSolver::solve(std::shared_ptr<df::Function> &rho,
                          const std::vector<Object> &objects)
{
    L->set_coefficient("rho", rho);
    df::assemble(b, *L);

    for(auto i = 0; i<num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }

    for(auto& bc: objects)
    {
        bc.apply(A, b);
    }
    auto phi = std::make_shared<df::Function>(rho->function_space());
    solver.solve(A, *phi->vector(), b);
    return phi;
}

double PoissonSolver::residual(std::shared_ptr<df::Function> &phi)
{
    df::Vector residual(*phi->vector());
    A.mult(*phi->vector(), residual);
    residual.axpy(-1.0, b);
    return residual.norm("l2");
}

double errornorm(std::shared_ptr<df::Function> &phi,
                 std::shared_ptr<df::Function> &phi_e)
{
    auto mesh = phi->function_space()->mesh();
    auto dim = mesh->geometry().dim();
    std::shared_ptr<df::FunctionSpace> W;
    std::shared_ptr<df::Form> error;
    if (dim == 1)
    {
        W = std::make_shared<ErrorNorm1D::CoefficientSpace_e>(mesh);
        error = std::make_shared<ErrorNorm1D::Form_a>(mesh);
    }
    else if (dim == 2)
    {
        W = std::make_shared<ErrorNorm2D::CoefficientSpace_e>(mesh);
        error = std::make_shared<ErrorNorm2D::Form_a>(mesh);
    }
    else if (dim == 3)
    {
        W = std::make_shared<ErrorNorm3D::CoefficientSpace_e>(mesh);
        error = std::make_shared<ErrorNorm3D::Form_a>(mesh);
    }
    auto u = std::make_shared<df::Function>(W);
    auto uh = std::make_shared<df::Function>(W);
    auto e = std::make_shared<df::Function>(W);

    u->interpolate(*phi);
    uh->interpolate(*phi_e);

    *e->vector() += *u->vector();
    *e->vector() -= *uh->vector();

    error->set_coefficient("e", e);
  
    return std::sqrt(df::assemble(*error));
}

ESolver::ESolver(std::shared_ptr<df::FunctionSpace> &V,
                 std::string method, std::string preconditioner)
                 : solver(method, preconditioner)
{
    auto dim = V->mesh()->geometry().dim();
    if (dim == 1)
    {
        W = std::make_shared<EField1D::FunctionSpace>(V->mesh());
        a = std::make_shared<EField1D::BilinearForm>(W, W);
        L = std::make_shared<EField1D::LinearForm>(W);
    }
    else if (dim == 2)
    {
        W = std::make_shared<EField2D::FunctionSpace>(V->mesh());
        a = std::make_shared<EField2D::BilinearForm>(W, W);
        L = std::make_shared<EField2D::LinearForm>(W);
    }
    else if (dim == 3)
    {
        W = std::make_shared<EField3D::FunctionSpace>(V->mesh());
        a = std::make_shared<EField3D::BilinearForm>(W, W);
        L = std::make_shared<EField3D::LinearForm>(W);
    }
    solver.parameters["absolute_tolerance"] = 1e-14;
    solver.parameters["relative_tolerance"] = 1e-12;
    solver.parameters["maximum_iterations"] = 1000;
    solver.set_reuse_preconditioner(true);

    df::assemble(A, *a);
}

std::shared_ptr<df::Function> ESolver::solve(std::shared_ptr<df::Function> &phi)
{
    L->set_coefficient("phi", phi);
    df::assemble(b, *L);
    auto E = std::make_shared<df::Function>(W);
    solver.solve(A, *E->vector(), b);
    return E;
}

VarPoissonSolver::VarPoissonSolver(std::shared_ptr<df::FunctionSpace> &V,
                                   std::vector<df::DirichletBC> &ext_bc,
                                   VObject &vobject,
                                   std::string method,
                                   std::string preconditioner) : 
                                   V(V), ext_bc(ext_bc), 
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
    a->set_exterior_facet_domains(vobject.bnd);
    L->set_exterior_facet_domains(vobject.bnd);

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

df::Function VarPoissonSolver::solve(std::shared_ptr<df::Function> &rho, double Q,
                                                    VObject &int_bc)
{
    L->set_coefficient("rho", rho);
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
