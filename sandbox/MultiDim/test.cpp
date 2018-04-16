#include <iostream>
#include <math.h>
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <chrono>
#include <dolfin.h>
#include "Potential.h"
#include "EField.h"
#include "ErrorNorm.h"
#include "Flux.h"
#include "Energy.h"
#include "Volume.h"
#include "Surface.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace df = dolfin;

class Object : public df::DirichletBC
{
  public:
    double potential;
    double charge;
    bool floating;
    std::size_t id;
    std::shared_ptr<df::MeshFunction<std::size_t>> bnd;

    double interpolated_charge;
    std::vector<std::size_t> dofs;
    std::size_t size_dofs;

    Object(const std::shared_ptr<df::FunctionSpace> &V,
           std::shared_ptr<df::MeshFunction<std::size_t>> &boundaries,
           std::size_t bnd_id,
           double potential = 0.0,
           double charge = 0.0,
           bool floating = true,
           std::string method = "topological");

    void get_dofs();
    void add_charge(const double &q);
    void set_potential(double voltage);
    void compute_interpolated_charge(const std::shared_ptr<df::Function> &q_rho);
};

Object::Object(const std::shared_ptr<df::FunctionSpace> &V,
               std::shared_ptr<df::MeshFunction<std::size_t>> &boundaries,
               std::size_t bnd_id,
               double potential,
               double charge,
               bool floating,
               std::string method) : df::DirichletBC(V, std::make_shared<df::Constant>(potential),
                                                     boundaries, bnd_id, method),
                                     potential(potential),
                                     charge(charge), floating(floating), id(bnd_id)
{
    auto tags = boundaries->values();
    auto size = boundaries->size();
    bnd = std::make_shared<df::MeshFunction<std::size_t>>();
    *bnd = *boundaries;
    bnd->set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
        if (tags[i] == id)
        {
            bnd->set_value(i, 9999);
        }
    }
    get_dofs();
}

void Object::get_dofs()
{
    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);

    for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
    {
        dofs.emplace_back(itr->first);
    }
    size_dofs = dofs.size();
}

void Object::add_charge(const double &q)
{
    charge += q;
}

void Object::set_potential(double voltage)
{
    this->potential = voltage;
    this->set_value(std::make_shared<df::Constant>(voltage));
}

void Object::compute_interpolated_charge(const std::shared_ptr<df::Function> &q_rho)
{
    interpolated_charge = 0.0;
    for (std::size_t i = 0; i < size_dofs; ++i)
    {
        interpolated_charge += q_rho->vector()->getitem(dofs[i]);
    }
}

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

class PoissonSolver
{
  public:
    PoissonSolver(const std::shared_ptr<df::FunctionSpace> &V,
                  boost::optional<std::vector<df::DirichletBC> &> bc = boost::none,
                  bool remove_null_space = false,
                  std::string method = "cg",
                  std::string preconditioner = "petsc_amg");

    std::shared_ptr<df::Function> solve(const std::shared_ptr<df::Function> &rho);

    std::shared_ptr<df::Function> solve(const std::shared_ptr<df::Function> &rho,
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

PoissonSolver::PoissonSolver(const std::shared_ptr<df::FunctionSpace> &V,
                             boost::optional<std::vector<df::DirichletBC> &> bc,
                             bool remove_null_space,
                             std::string method,
                             std::string preconditioner) : ext_bc(bc),
                                                           remove_null_space(remove_null_space),
                                                           solver(method, preconditioner)
{
    auto dim = V->mesh()->geometry().dim();
    if (dim == 1)
    {
        a = std::make_shared<Potential::BilinearForm>(V, V);
        L = std::make_shared<Potential::LinearForm>(V);
    }
    else if (dim == 2)
    {
        a = std::make_shared<Potential::BilinearForm>(V, V);
        L = std::make_shared<Potential::LinearForm>(V);
    }
    else if (dim == 3)
    {
        a = std::make_shared<Potential::BilinearForm>(V, V);
        L = std::make_shared<Potential::LinearForm>(V);
    }
  
    df::assemble(A, *a);

    if (ext_bc)
    {
        num_bcs = ext_bc->size();
    }
    for (auto i = 0; i < num_bcs; ++i)
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

std::shared_ptr<df::Function> PoissonSolver::solve(const std::shared_ptr<df::Function> &rho)
{
    L->set_coefficient("rho", rho);

    df::assemble(b, *L);

    for (auto i = 0; i < num_bcs; ++i)
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

std::shared_ptr<df::Function> PoissonSolver::solve(const std::shared_ptr<df::Function> &rho,
                                                   const std::vector<Object> &objects)
{
    L->set_coefficient("rho", rho);

    df::assemble(b, *L);

    for (auto i = 0; i < num_bcs; ++i)
    {
        ext_bc.get()[i].apply(b);
    }

    for (auto &bc : objects)
    {
        bc.apply(A, b);
    }
    auto phi = std::make_shared<df::Function>(rho->function_space());
    solver.solve(A, *phi->vector(), b);
    return phi;
}

double PoissonSolver::residual(const std::shared_ptr<df::Function> &phi)
{
    df::Vector residual(*phi->vector());
    A.mult(*phi->vector(), residual);
    residual.axpy(-1.0, b);
    return residual.norm("l2");
}

double errornorm(const std::shared_ptr<df::Function> &phi,
                 const std::shared_ptr<df::Function> &phi_e)
{
    auto mesh = phi->function_space()->mesh();
    auto dim = mesh->geometry().dim();
    std::shared_ptr<df::FunctionSpace> W;
    std::shared_ptr<df::Form> error;
    if (dim == 1)
    {
        W = std::make_shared<ErrorNorm::CoefficientSpace_w0>(mesh);
        error = std::make_shared<ErrorNorm::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        W = std::make_shared<ErrorNorm::CoefficientSpace_w0>(mesh);
        error = std::make_shared<ErrorNorm::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        W = std::make_shared<ErrorNorm::CoefficientSpace_w0>(mesh);
        error = std::make_shared<ErrorNorm::Form_2>(mesh);
    }
    
    auto u = std::make_shared<df::Function>(W);
    auto uh = std::make_shared<df::Function>(W);
    auto e = std::make_shared<df::Function>(W);

    u->interpolate(*phi);
    uh->interpolate(*phi_e);

    *e->vector() += *u->vector();
    *e->vector() -= *uh->vector();

    error->set_coefficient("w0", e);
    return sqrt(df::assemble(*error));
}

class ESolver
{
  public:
    ESolver(const std::shared_ptr<df::FunctionSpace> &V,
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

ESolver::ESolver(const std::shared_ptr<df::FunctionSpace> &V,
                 std::string method, std::string preconditioner)
    : solver(method, preconditioner)
{
    auto dim = V->mesh()->geometry().dim();
    if (dim == 1)
    {
        W = std::make_shared<EField::FunctionSpace>(V->mesh());
        a = std::make_shared<EField::BilinearForm>(W, W);
        L = std::make_shared<EField::LinearForm>(W);
    }
    else if (dim == 2)
    {
        W = std::make_shared<EField::FunctionSpace>(V->mesh());
        a = std::make_shared<EField::BilinearForm>(W, W);
        L = std::make_shared<EField::LinearForm>(W);
    }
    else if (dim == 3)
    {
        W = std::make_shared<EField::FunctionSpace>(V->mesh());
        a = std::make_shared<EField::BilinearForm>(W, W);
        L = std::make_shared<EField::LinearForm>(W);
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

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

template <class T>
bool inv(const boost::numeric::ublas::matrix<T> &input,
         boost::numeric::ublas::matrix<T> &inverse)
{
    typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;

    // create a working copy of the input
    boost::numeric::ublas::matrix<T> A(input);

    // create a permutation matrix for the LU-factorization
    pmatrix pm(A.size1());

    // perform LU-factorization
    int res = boost::numeric::ublas::lu_factorize(A, pm);
    if (res != 0)
        return false;

    // create identity matrix of "inverse"
    inverse.assign(boost::numeric::ublas::identity_matrix<T>(A.size1()));

    // backsubstitute to get the inverse
    boost::numeric::ublas::lu_substitute(A, pm, inverse);

    return true;
}

std::vector<std::shared_ptr<df::Function>> solve_laplace(
    const std::shared_ptr<df::FunctionSpace> &V,
    std::vector<Object> &objects,
    std::shared_ptr<df::MeshFunction<std::size_t>> &boundaries,
    std::size_t ext_bnd_id)
{
    auto phi_bnd = std::make_shared<df::Constant>(0.0);

    df::DirichletBC bnd(V, phi_bnd, boundaries, ext_bnd_id);
    std::vector<df::DirichletBC> bc = {bnd};
    PoissonSolver poisson(V, bc);
    ESolver esolver(V);
    auto num_objects = objects.size();
    std::vector<std::shared_ptr<df::Function>> object_e_field(num_objects);
    for (std::size_t i = 0; i < num_objects; ++i)
    {
        for (std::size_t j = 0; j < num_objects; ++j)
        {
            if (i == j)
            {
                objects[j].set_potential(1.0);
            }
            else
            {
                objects[j].set_potential(0.0);
            }
        }
        auto rho = std::make_shared<df::Function>(V);
        auto phi = poisson.solve(rho, objects);
        object_e_field[i] = esolver.solve(phi);
    }
    return object_e_field;
}

boost_matrix capacitance_matrix(
    const std::shared_ptr<df::FunctionSpace> &V,
    std::vector<Object> &objects,
    std::shared_ptr<df::MeshFunction<std::size_t>> &boundaries,
    std::size_t ext_bnd_id)
{
    auto mesh = V->mesh();
    auto dim = mesh->geometry().dim();
    auto num_objects = objects.size();
    Flux::Functional flux(mesh);
    // std::shared_ptr<df::Form> flux;
    // if (dim == 1)
    // {
    //     flux = std::make_shared<Flux::Form_0>(mesh);
    // }
    // else if (dim == 2)
    // {
    //     flux = std::make_shared<Flux::Form_1>(mesh);
    // }
    // else if (dim == 3)
    // {
    //     flux = std::make_shared<Flux::Form_2>(mesh);
    // }
    boost_matrix capacitance(num_objects, num_objects);
    boost_matrix inv_capacity(num_objects, num_objects);
    auto object_e_field = solve_laplace(V, objects, boundaries, ext_bnd_id);

    for (unsigned i = 0; i < num_objects; ++i)
    {
        flux.ds = objects[i].bnd;
        // flux->set_exterior_facet_domains(objects[i].bnd);
        for (unsigned j = 0; j < num_objects; ++j)
        {
            flux.e = object_e_field[j];
            // flux->set_coefficient("w0", object_e_field[j]);
            capacitance(i, j) = df::assemble(flux);
        }
    }
    std::cout << "capacity: " << capacitance << '\n';
    inv(capacitance, inv_capacity);
    std::cout << "inv_capacity: " << inv_capacity << '\n';
    // forms.append(inner(Coefficient(element), -1*n)*ds(9999))
    return inv_capacity;
}

double mesh_potential_energy(std::shared_ptr<df::Function> &phi,
                             std::shared_ptr<df::Function> &rho)
{
    auto mesh = phi->function_space()->mesh();
    auto dim = mesh->geometry().dim();
    std::shared_ptr<df::Form> energy;
    if (dim == 1)
    {
        energy = std::make_shared<Energy::Form_0>(mesh, phi, rho);
    }
    else if (dim == 2)
    {
        energy = std::make_shared<Energy::Form_1>(mesh, phi, rho);
    }
    else if (dim == 3)
    {
        energy = std::make_shared<Energy::Form_2>(mesh, phi, rho);
    }
    // energy->set_coefficient("w0", phi);
    // energy->set_coefficient("w1", rho);
    return 0.5 * df::assemble(*energy);
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

std::shared_ptr<df::FunctionSpace> create_function_space(std::shared_ptr<const df::Mesh> &mesh)
{
    std::shared_ptr<df::FunctionSpace> V;
    if (mesh->geometry().dim() == 1)
    {
        V = std::make_shared<Potential::FunctionSpace>(mesh);
    }
    else if (mesh->geometry().dim() == 2)
    {
        V = std::make_shared<Potential::FunctionSpace>(mesh);
    }
    else if (mesh->geometry().dim() == 3)
    {
        V = std::make_shared<Potential::FunctionSpace>(mesh);
    }
    else
        df::error("PUNC is programmed for dimensions up to 3D only.");
    
    return V;
}

int main()
{

    auto r = 0.02;
    auto R = 0.2;
    auto cap = 4. * M_PI * r * R / (R - r);

    //----------- Create mesh --------------------------------------------------
    std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/capacitance/mesh/main02"};
    auto mesh = load_mesh(fname);

    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    //__________________________________________________________________________

    //--------------Create the function space-----------------------------------
    // auto V = std::make_shared<Potential::FunctionSpace>(mesh);
    // auto V = create_function_space(mesh);
    auto V = std::make_shared<Potential::FunctionSpace>(mesh);
    //__________________________________________________________________________

    //-------------------Define boundary condition------------------------------
    auto u0 = std::make_shared<df::Constant>(0.0);
    df::DirichletBC bc(V, u0, boundaries, ext_bnd_id);
    std::vector<df::DirichletBC> bcs = {bc};
    
    //----------Create the object-----------------------------------------------
    Object object(V, boundaries, tags[2]);
    object.set_potential(1.0);
    std::vector<Object> obj = {object};
    //__________________________________________________________________________

    //__________________________________________________________________________

    //---------------------Capacitance------------------------------------------
    typedef boost::numeric::ublas::matrix<double> boost_matrix;

    boost_matrix inv_capacity = capacitance_matrix(V, obj, boundaries, ext_bnd_id);

    std::cout << "Analytical value: " << cap << '\n';
    printf("volume: %e\n", volume(mesh));
    //__________________________________________________________________________

    return 0;
}
