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
#include "../../punc/Potential1D.h"
#include "../../punc/Potential2D.h"
#include "../../punc/Potential3D.h"
#include "../../punc/VarPotential1D.h"
#include "../../punc/VarPotential2D.h"
#include "../../punc/VarPotential3D.h"
#include "../../punc/EField1D.h"
#include "../../punc/EField2D.h"
#include "../../punc/EField3D.h"
#include "../../punc/ErrorNorm1D.h"
#include "../../punc/ErrorNorm2D.h"
#include "../../punc/ErrorNorm3D.h"
#include "../../punc/Surface.h"
#include "../../punc/Volume.h"
#include "../../punc/Flux.h"
#include "../../punc/Charge.h"
#include "../../punc/Energy.h"
#include <boost/optional.hpp>
#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/systems/si/codata/electron_constants.hpp>
#include <boost/units/systems/si/codata/physico-chemical_constants.hpp>
#include <boost/units/systems/si/codata/universal_constants.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

namespace df = dolfin;

std::vector<double> cross(std::vector<double> &v1, std::vector<double> &v2)
{
    std::vector<double> r(v1.size());
    r[0] = v1[1] * v2[2] - v1[2] * v2[1];
    r[1] = -v1[0] * v2[2] + v1[2] * v2[0];
    r[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return r;
}

class Timer
{
  public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

  private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};

struct random_seed_seq
{
    template <typename It>
    void generate(It begin, It end)
    {
        for (; begin != end; ++begin)
        {
            *begin = device();
        }
    }

    static random_seed_seq &get_instance()
    {
        static thread_local random_seed_seq result;
        return result;
    }

  private:
    std::random_device device;
};

struct PhysicalConstants
{
    double e = boost::units::si::constants::codata::e / boost::units::si::coulomb;
    double m_e = boost::units::si::constants::codata::m_e / boost::units::si::kilograms;
    double ratio = boost::units::si::constants::codata::m_e_over_m_p / boost::units::si::dimensionless();
    double m_i = m_e / ratio;

    double k_B = boost::units::si::constants::codata::k_B * boost::units::si::kelvin / boost::units::si::joules;
    double eps0 = boost::units::si::constants::codata::epsilon_0 * boost::units::si::meter / boost::units::si::farad;
};

class Facet
{
  public:
    double area;
    std::vector<double> vertices;
    std::vector<double> normal;
    std::vector<double> basis;

    Facet(double area,
          std::vector<double> vertices,
          std::vector<double> normal,
          std::vector<double> basis) : area(area),
                                       vertices(vertices),
                                       normal(normal),
                                       basis(basis) {}
};

std::vector<Facet> exterior_boundaries(df::MeshFunction<std::size_t> &boundaries,
                                       std::size_t ext_bnd_id)
{
    auto mesh = boundaries.mesh();
    auto gdim = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    auto values = boundaries.values();
    auto length = boundaries.size();
    int num_facets = 0;
    for (std::size_t i = 0; i < length; ++i)
    {
        if (ext_bnd_id == values[i])
        {
            num_facets += 1;
        }
    }
    std::vector<Facet> ext_facets;

    double area;
    std::vector<double> normal(gdim);
    std::vector<double> vertices(gdim * gdim);
    std::vector<double> basis(gdim * gdim);
    std::vector<double> vertex(gdim);
    double norm;
    int j;
    mesh->init(tdim - 1, tdim);
    df::SubsetIterator facet_iter(boundaries, ext_bnd_id);
    for (; !facet_iter.end(); ++facet_iter)
    {
        df::Cell cell(*mesh, facet_iter->entities(tdim)[0]);
        auto cell_facet = cell.entities(tdim - 1);
        std::size_t num_facets = cell.num_entities(tdim - 1);
        for (std::size_t i = 0; i < num_facets; ++i)
        {
            if (cell_facet[i] == facet_iter->index())
            {
                area = cell.facet_area(i);
                for (std::size_t j = 0; j < gdim; ++j)
                {
                    normal[j] = -1*cell.normal(i, j);
                }
            }
        }
        j = 0;
        for (df::VertexIterator v(*facet_iter); !v.end(); ++v)
        {
            for (std::size_t i = 0; i < gdim; ++i)
            {
                vertices[j * gdim + i] = v->x(i);
            }
            j += 1;
        }
        ext_facets.push_back(Facet(area, vertices, normal, basis));
    }
    return ext_facets;
}

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

std::vector<std::size_t> get_mesh_ids(df::MeshFunction<std::size_t> &boundaries)
{
    auto values = boundaries.values();
    auto length = boundaries.size();
    std::vector<std::size_t> tags(length);

    for (std::size_t i = 0; i < length; ++i)
    {
        tags[i] = values[i];
    }
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    return tags;
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


class Object : public df::DirichletBC
{
public:
    double potential;
    double charge;
    bool floating;
    std::size_t id;
    df::MeshFunction<std::size_t> bnd;

    double interpolated_charge;
    std::vector<std::size_t> dofs;
    std::size_t size_dofs;

    Object(const df::FunctionSpace &V,
           const df::MeshFunction<std::size_t> &boundaries,
           std::size_t bnd_id,
           double potential = 0.0,
           double charge = 0.0,
           bool floating = true,
           std::string method = "topological");

    void get_dofs();
    void add_charge(const double &q);
    void set_potential(const double &voltage);
    void compute_interpolated_charge(const df::Function &q_rho);
};

Object::Object(const df::FunctionSpace &V,
               const df::MeshFunction<std::size_t> &boundaries,
               std::size_t bnd_id,
               double potential,
               double charge,
               bool floating,
               std::string method):
               df::DirichletBC(std::make_shared<df::FunctionSpace>(V),
               std::make_shared<df::Constant>(potential),
               std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
               bnd_id, method), potential(potential),
               charge(charge), floating(floating), id(bnd_id)
{
    auto tags = boundaries.values();
    auto size = boundaries.size();
    bnd = boundaries;
    bnd.set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
    	if(tags[i] == id)
    	{
    		bnd.set_value(i, 9999);
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

void Object::set_potential(const double &voltage)
{
    potential = voltage;
    set_value(std::make_shared<df::Constant>(voltage));
}

void Object::compute_interpolated_charge(const df::Function &q_rho)
{
    interpolated_charge = 0.0;
    for (std::size_t i = 0; i < size_dofs; ++i)
    {
        interpolated_charge += q_rho.vector()->getitem(dofs[i]);
    }
}

void reset_objects(std::vector<Object> &objcets)
{
    for (auto obj: objcets)
    {
        obj.set_potential(0.0);
    }
}

void compute_object_potentials(std::vector<Object> &objects,
                               const df::Function &E,
                               const boost_matrix &inv_capacity,
                               std::shared_ptr<const df::Mesh> &mesh)
{
    auto dim = mesh->geometry().dim();
    int num_objects = objects.size();
    std::vector<double> image_charge(num_objects);

    std::shared_ptr<df::Form> flux;
    if (dim == 1)
    {
        flux = std::make_shared<Flux::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        flux = std::make_shared<Flux::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        flux = std::make_shared<Flux::Form_2>(mesh);
    }
    for (unsigned j = 0; j < num_objects; ++j)
    {
        flux->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[j].bnd));
        flux->set_coefficient("w0", std::make_shared<const df::Function>(E));
        image_charge[j] = df::assemble(*flux);
        // flux.ds = std::make_shared<df::MeshFunction<std::size_t>>(objects[j].bnd);
        // flux.e = std::make_shared<const df::Function>(E);
        // image_charge[j] = df::assemble(flux);
    }

    int i, j;
    double potential;
    for (i = 0; i < num_objects; ++i)
    {
        potential = 0.0;
        for (j = 0; j < num_objects; ++j)
        {
            potential += (objects[j].charge -\
                          image_charge[j]) * inv_capacity(i, j);
        }
        objects[i].set_potential(potential);
    }
}

class VObject : public df::DirichletBC
{
  public:
    double potential;
    double charge;
    bool floating;
    std::size_t id;
    df::MeshFunction<std::size_t> bnd;
    std::vector<std::size_t> dofs;
    std::size_t num_dofs;
    std::shared_ptr<df::Form> charge_form;

    VObject(const df::FunctionSpace &V,
            df::MeshFunction<std::size_t> &boundaries,
            std::size_t bnd_id,
            double potential = 0.0,
            double charge = 0.0,
            bool floating = true,
            std::string method = "topological");
    void get_dofs();
    void add_charge(const double &q);
    double calculate_charge(df::Function &phi);
    void set_potential(double voltage);
    void apply(df::GenericVector &b);
    void apply(df::GenericMatrix &A);
};

VObject::VObject(const df::FunctionSpace &V,
                 df::MeshFunction<std::size_t> &boundaries,
                 std::size_t bnd_id,
                 double potential,
                 double charge,
                 bool floating,
                 std::string method) : df::DirichletBC(V.sub(0), std::make_shared<df::Constant>(potential),
                                                       std::make_shared<df::MeshFunction<std::size_t>>(boundaries), bnd_id, method),
                                       potential(potential), charge(charge), floating(floating),
                                       id(bnd_id)
{
    auto tags = boundaries.values();
    auto size = boundaries.size();
    bnd = boundaries;
    bnd.set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
        if (tags[i] == id)
        {
            bnd.set_value(i, 9999);
        }
    }

    auto mesh = V.mesh();
    auto dim = mesh->geometry().dim();
    if (dim == 1)
    {
        charge_form = std::make_shared<Charge::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        charge_form = std::make_shared<Charge::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        charge_form = std::make_shared<Charge::Form_2>(mesh);
    }
    charge_form->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(bnd));
    get_dofs();
}
void VObject::get_dofs()
{
    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);

    for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
    {
        dofs.emplace_back(itr->first);
    }
    num_dofs = dofs.size();
}

void VObject::add_charge(const double &q)
{
    charge += q;
}

double VObject::calculate_charge(df::Function &phi)
{
    charge_form->set_coefficient("w0", std::make_shared<df::Function>(phi));
    return df::assemble(*charge_form);
}

void VObject::set_potential(double voltage)
{
    this->potential = voltage;
    this->set_value(std::make_shared<df::Constant>(voltage));
}

void VObject::apply(df::GenericVector &b)
{
    df::DirichletBC::apply(b);
}

void VObject::apply(df::GenericMatrix &A)
{
    if (!floating)
    {
        df::DirichletBC::apply(A);
    }
    else
    {
        for (auto i = 1; i < num_dofs; ++i)
        {
            std::vector<double> neighbor_values;
            std::vector<std::size_t> neighbor_ids, surface_neighbors;
            A.getrow(dofs[i], neighbor_ids, neighbor_values);
            std::size_t num_neighbors = neighbor_ids.size();
            std::fill(neighbor_values.begin(), neighbor_values.end(), 0.0);

            std::size_t num_surface_neighbors = 0;
            std::size_t self_index;
            for (auto j = 0; j < num_neighbors; ++j)
            {
                if (std::find(dofs.begin(), dofs.end(), neighbor_ids[j]) != dofs.end())
                {
                    neighbor_values[j] = -1.0;
                    num_surface_neighbors += 1;
                    if (neighbor_ids[j] == dofs[i])
                    {
                        self_index = j;
                    }
                }
            }
            neighbor_values[self_index] = num_surface_neighbors - 1;
            A.setrow(dofs[i], neighbor_ids, neighbor_values);
            A.apply("insert");
        }
    }
}

class PoissonSolver
{
public:
    PoissonSolver(const df::FunctionSpace &V,
                  boost::optional<std::vector<df::DirichletBC>& > ext_bc = boost::none,
                  bool remove_null_space = false,
                  std::string method = "cg",
                  std::string preconditioner = "petsc_amg");

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

PoissonSolver::PoissonSolver(const df::FunctionSpace &V,
                             boost::optional<std::vector<df::DirichletBC>& > ext_bc,
                             bool remove_null_space,
                             std::string method,
                             std::string preconditioner) : ext_bc(ext_bc),
                             remove_null_space(remove_null_space),
                             solver(method, preconditioner)
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
    // L.rho = std::make_shared<df::Function>(rho);
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    df::assemble(b, *L);

    // auto num_bcs = ext_bc->size();
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
    // L.rho = std::make_shared<df::Function>(rho);
    L->set_coefficient("rho", std::make_shared<df::Function>(rho));
    df::assemble(b, *L);

    // auto num_bcs = ext_bc->size();
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

ESolver::ESolver(const df::FunctionSpace &V,
                 std::string method, std::string preconditioner):
                 solver(method, preconditioner)
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
    // L.phi = std::make_shared<df::Function>(phi);
    df::assemble(b, *L);
    df::Function E(W);
    solver.solve(A, *E.vector(), b);
    return E;
}

bool inv(const boost_matrix &input, boost_matrix &inverse)
{
    typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;
    boost::numeric::ublas::matrix<double> A(input);

    pmatrix pm(A.size1());

    int res = boost::numeric::ublas::lu_factorize(A, pm);
    if (res != 0)
        return false;

    inverse.assign(boost::numeric::ublas::identity_matrix<double>(A.size1()));

    boost::numeric::ublas::lu_substitute(A, pm, inverse);

    return true;
}

std::vector<df::Function> solve_laplace(const df::FunctionSpace &V,
                                        std::vector<Object> &objects,
                                        df::MeshFunction<std::size_t> boundaries,
                                        std::size_t ext_bnd_id)
{
    auto phi_bnd = std::make_shared<df::Constant>(0.0);
    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V),
                       phi_bnd,
                       std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
                       ext_bnd_id);
    std::vector<df::DirichletBC> ext_bc = {bc};
    PoissonSolver poisson(V, ext_bc);
    ESolver esolver(V);
    auto num_objects = objects.size();

    std::vector<df::Function> object_e_field;
    auto shared_V = std::make_shared<df::FunctionSpace>(V);
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
        df::Function rho(shared_V);
        // df::Function phi(shared_V);
        auto phi = poisson.solve(rho, objects);
        object_e_field.emplace_back(esolver.solve(phi));
    }
    return object_e_field;
}

boost_matrix capacitance_matrix(const df::FunctionSpace &V,
                                std::vector<Object> &objects,
                                const df::MeshFunction<std::size_t> &boundaries,
                                std::size_t ext_bnd_id)
{
    auto mesh = V.mesh();
    auto dim = mesh->geometry().dim();
    auto num_objects = objects.size();
    std::shared_ptr<df::Form> flux;
    if (dim == 1)
    {
        flux = std::make_shared<Flux::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        flux = std::make_shared<Flux::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        flux = std::make_shared<Flux::Form_2>(mesh);
    }
    boost_matrix capacitance(num_objects, num_objects);
    boost_matrix inv_capacity(num_objects, num_objects);
    auto object_e_field = solve_laplace(V, objects, boundaries, ext_bnd_id);
    for (unsigned i = 0; i < num_objects; ++i)
    {
        // Flux::Form_flux flux(mesh);
        // flux.ds = std::make_shared<df::MeshFunction<std::size_t>>(objects[i].bnd);
        flux->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[i].bnd));
        for (unsigned j = 0; j < num_objects; ++j)
        {
            // flux.ds = std::make_shared<df::MeshFunction<std::size_t>>(objects[i].bnd);
            // flux.e = std::make_shared<df::Function>(object_e_field[j]);
            flux->set_coefficient("w0", std::make_shared<df::Function>(object_e_field[j]));
            capacitance(i, j) = df::assemble(*flux);
        }
    }
    std::cout << "capacity: " << capacitance << '\n';
    inv(capacitance, inv_capacity);
    return inv_capacity;
}

boost_matrix bias_matrix(const boost_matrix &inv_capacity,
                         const std::map<int, std::vector<int>> &circuits_info)
{
    std::size_t num_components = inv_capacity.size1();
    std::size_t num_circuits = circuits_info.size();
    boost_matrix bias_matrix(num_components, num_components, 0.0);
    boost_matrix inv_bias(num_components, num_components);

    std::vector<int> circuit;
    int i, s = 0;
    for (auto const &c_map : circuits_info)
    {
        i = c_map.first;
        circuit = c_map.second;
        for (unsigned ii = 0; ii < circuit.size(); ++ii)
        {
            bias_matrix(num_circuits + i, circuit[ii]) = 1.0;
        }
        for (unsigned j = 1; j < circuit.size(); ++j)
        {
            for (unsigned k = 0; k < num_components; ++k)
            {
                bias_matrix(j - 1 + s, k) = inv_capacity(circuit[j], k) -
                                            inv_capacity(circuit[0], k);
            }
        }
        s += circuit.size() - 1;
    }
    inv(bias_matrix, inv_bias);
    return inv_bias;
}

std::function<double(std::vector<double> &)> maxwellian_vdf(double vth, std::vector<double> &vd)
{
    auto dim = vd.size();
    auto vth2 = vth*vth;
    auto factor =  (1.0 / (pow(sqrt(2. * M_PI * vth2), dim)));
    auto pdf = [vth2, vd, factor, dim](std::vector<double> &v) {
        double v_sqrt = 0.0;
        for (auto i = 0; i < dim; ++i)
        {
            v_sqrt += (v[i] - vd[i]) * (v[i] - vd[i]);
        }
        return factor*exp(-0.5 * v_sqrt / vth2);
    };

    return pdf;
}

std::vector<std::vector<double>> combinations(std::vector<std::vector<double>> vec, double dv)
{
    auto dim = vec.size()/2 + 1;
    auto len = pow(2,dim);
    std::vector<std::vector<double>> arr;
    arr.resize(len, std::vector<double>(dim, 0.0));

    auto plen = int(len/2);

    for (auto i = 0; i < len; ++i)
    {
        for (auto j = 0; j <dim-1; ++j)
        {
            arr[i][j] = vec[i%plen][j];
        }
    }
    for (auto i = 0; i < pow(2, dim-1); ++i)
    {
        arr[i][dim-1] = 0.0;
        arr[pow(2, dim-1)+i][dim-1] = dv;
    }

    return arr;
}

class ORS
{
public:
    std::function<double(std::vector<double> &)> vdf;
    int dim, nbins, num_edges;
    std::vector<double> dv;
    std::vector<std::vector<double>> sp;

    std::vector<double> pdf_max;
    std::vector<double> cdf;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;

    ORS(double vth, std::vector<double> &vd,
        std::function<double(std::vector<double> &)> vdf, int num_sp=60);
    std::vector<double> sample(const std::size_t N);
};

ORS::ORS(double vth, std::vector<double> &vd,
         std::function<double(std::vector<double> &)> vdf, int num_sp):vdf(vdf),
         dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
{

    dim = vd.size();
    dv.resize(dim);
    std::vector<double> nsp(3,1.0), diff(dim);

    nsp[0] = num_sp;
    for (auto i = 0; i < dim; ++i)
    {
        diff[i] = 10.0*vth;
    }
    for (auto i = 1; i < dim; ++i)
    {
        nsp[i] = nsp[i - 1] * diff[i] / diff[i - 1];
    }
    for (auto i = 0; i < dim; ++i)
    {
        dv[i] = diff[i] / nsp[i];
    }

    nbins = std::accumulate(nsp.begin(), nsp.end(), 1, std::multiplies<int>());

    std::vector<std::vector<double>> points, edges{{0.0}, {dv[0]}};
    for (auto i = 1; i < dim; ++i)
    {
        points = combinations(edges, dv[i]);
        edges = points;
    }
    num_edges = edges.size();

    int rows, cols, cells;
    rows = (int)nsp[0];
    cols = (int)nsp[1];
    cells = (int)nsp[2];
    int num_bins = rows*cols*cells;
    std::vector<int> indices(dim);

    sp.resize(num_bins*num_edges);
    for (auto i = 0; i < num_bins; ++i)
    {
        indices[0] = i/(cols*cells);
        indices[1] = (i/cells)%cols;
        indices[2] = i - indices[0] * cols*cells - indices[1] * cells;
        for (auto j = 0; j < num_edges; ++j)
        {
            for (auto k = 0; k < dim; ++k)
            {
                sp[i*num_edges+j].push_back(vd[k]-5.*vth + dv[k]*indices[k]+edges[j][k]);
            }
        }
    }

    pdf_max.resize(nbins);
    double max, value;

    for (auto j = 0; j < nbins; ++j)
    {
        max = 0.0;
        for (auto k = 0; k < num_edges; ++k)
        {
            value = vdf(sp[j * num_edges + k]);
            max = std::max(max, value);
        }
        pdf_max[j] = max;
    }

    auto normalization_factor = std::accumulate(pdf_max.begin(), pdf_max.end(), 0.0);

    std::partial_sum(pdf_max.begin(), pdf_max.end(), std::back_inserter(cdf));

    std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                   std::bind1st(std::multiplies<double>(), 1. / normalization_factor));
}

std::vector<double> ORS::sample(const std::size_t N)
{
    random_source rng(random_seed_seq::get_instance());
    std::vector<double> vs(N*dim), vs_new(dim);
    int index, n = 0;
    double p_vs, value;
    while (n < N)
    {
        index = std::distance(cdf.begin(),
                std::lower_bound(cdf.begin(), cdf.end(), dist(rng)));

        for (int i = n * dim; i < (n + 1) * dim; ++i)
        {
            vs_new[i%dim] = sp[index*num_edges][i%dim] + dv[i % dim] * dist(rng);
            vs[i] = vs_new[i%dim];
        }
        value = vdf(vs_new);
        p_vs = pdf_max[index] * dist(rng);
        n = n + (p_vs<value);
    }
    return vs;
}

enum VDFType {Generic, Maxwellian};

class InFlux
{
public:
    std::vector<double> num_particles;
    virtual std::vector<double> sample(const std::size_t N, const std::size_t f){};
};

class GenericFlux
{
public:

    int dim, nbins, num_edges;
    std::vector<double> dv;
    std::vector<std::vector<double>> sp;

    std::vector<double> pdf_max;
    std::vector<double> cdf;
    std::vector<std::function<double(std::vector<double> &)>> vdf;
    std::vector<double> num_particles;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;

    GenericFlux();
    GenericFlux(double vth, std::vector<double> &vd,
        const std::vector<std::vector<double>> &cutoffs,
        int num_sp,
        std::vector<Facet> &facets);
    std::vector<double> sample(const std::size_t N, const std::size_t f);
};

GenericFlux::GenericFlux(){}

GenericFlux::GenericFlux(double vth, std::vector<double> &vd,
         const std::vector<std::vector<double>> &cutoffs,
         int num_sp,
         std::vector<Facet> &facets) :
         dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
{
    double vth2 = vth * vth;
    double factor = (1.0 / (sqrt(2. * M_PI * vth2)));

    auto num_facets = facets.size();
    num_particles.resize(num_facets);
    vdf.resize(num_facets);

    dim = vd.size();
    dv.resize(dim);
    std::vector<double> nsp(3,1.0), diff(dim);

    nsp[0] = num_sp;
    for (auto i = 0; i < dim; ++i)
    {
        diff[i] = cutoffs[i][1] - cutoffs[i][0];
    }
    for (auto i = 1; i < dim; ++i)
    {
        nsp[i] = nsp[i - 1] * diff[i] / diff[i - 1];
    }
    for (auto i = 0; i < dim; ++i)
    {
        dv[i] = diff[i] / nsp[i];
    }

    nbins = std::accumulate(nsp.begin(), nsp.end(), 1, std::multiplies<int>());

    std::vector<std::vector<double>> points, edges{{0.0}, {dv[0]}};
    for (auto i = 1; i < dim; ++i)
    {
        points = combinations(edges, dv[i]);
        edges = points;
    }
    num_edges = edges.size();

    int rows, cols, cells;
    rows = (int)nsp[0];
    cols = (int)nsp[1];
    cells = (int)nsp[2];
    int num_bins = rows*cols*cells;
    std::vector<int> indices(dim);

    sp.resize(num_bins*num_edges);
    for (auto i = 0; i < num_bins; ++i)
    {
        indices[0] = i/(cols*cells);
        indices[1] = (i/cells)%cols;
        indices[2] = i - indices[0] * cols*cells - indices[1] * cells;
        for (auto j = 0; j < num_edges; ++j)
        {
            for (auto k = 0; k < dim; ++k)
            {
                sp[i*num_edges+j].push_back(cutoffs[k][0] + dv[k]*indices[k]+edges[j][k]);
            }
        }
    }

    auto vdf_maxwellian = maxwellian_vdf(vth, vd);

    pdf_max.resize(num_facets*nbins);
    vdf.resize(num_facets);
    double vdn, max, value;

    for(auto i=0; i<num_facets; ++i)
    {
        auto n = facets[i].normal;
        vdn = std::inner_product(n.begin(), n.end(), vd.begin(), 0.0);
        num_particles[i] = facets[i].area * (vth / (sqrt(2 * M_PI)) *
                        exp(-0.5 * (vdn / vth) * (vdn / vth)) +
                        0.5 * vdn * (1. + erf(vdn / (sqrt(2) * vth))));

        vdf[i] = [vdf_maxwellian, n](std::vector<double> &v) {
                auto vn = std::inner_product(std::begin(n), std::end(n), std::begin(v), 0.0);
                if(vn>0.0)
                {
                    return vn * vdf_maxwellian(v);
                }else{
                    return 0.0;
                }
                // return (vn > 0.0) * vn * vdf_maxwellian(v);
            };

        for (auto j = 0; j < nbins; ++j)
        {
            max = 0.0;
            for (auto k = 0; k < num_edges; ++k)
            {
                value = vdf[i](sp[j * num_edges + k]);
                max = std::max(max, value);
            }
            pdf_max[i*nbins + j] = max;
        }

        auto normalization_factor = std::accumulate(pdf_max.begin()+i*nbins, pdf_max.begin()+(i+1)*nbins, 0.0);

        std::partial_sum(pdf_max.begin()+i*nbins, pdf_max.begin()+(i+1)*nbins, std::back_inserter(cdf));

        std::transform(cdf.begin()+i*nbins, cdf.begin()+(i+1)*nbins, cdf.begin()+i*nbins,
                       std::bind1st(std::multiplies<double>(), 1. / normalization_factor));
    }
}

std::vector<double> GenericFlux::sample(const std::size_t N, const std::size_t f)
{
    random_source rng(random_seed_seq::get_instance());
    std::vector<double> vs(N*dim), vs_new(dim);
    int index, n = 0;
    double p_vs, value;
    while (n < N)
    {
        index = std::distance(cdf.begin()+f*nbins,
                std::lower_bound(cdf.begin()+f*nbins, cdf.begin()+(f+1)*nbins, dist(rng)));

        for (int i = n * dim; i < (n + 1) * dim; ++i)
        {
            vs_new[i%dim] = sp[index*num_edges][i%dim] + dv[i % dim] * dist(rng);
            vs[i] = vs_new[i%dim];
        }
        value = vdf[f](vs_new);
        p_vs = pdf_max[index+f*nbins] * dist(rng);
        n = n + (p_vs<value);
    }
    return vs;
}

class MaxwellianFlux : public InFlux
{
private:
    std::vector<Facet> facets;

    int nsp;
    int dim;
    double v0, dv;

    std::vector<double> pdf_max;
    std::vector<double> cdf;
    std::vector<std::function<double(double)>> vdf;
    std::vector<std::function<double(double, int)>> maxwellian;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    distribution dist;
    random_source rng;

public:
    MaxwellianFlux(double vth, std::vector<double> &vd, std::vector<Facet> &facets);
    std::vector<double> sample(const std::size_t N, const std::size_t f) override;
};

MaxwellianFlux::MaxwellianFlux(double vth, std::vector<double> &vd,
                               std::vector<Facet> &facets)
                               : facets(facets), dist(0.0, 1.0),
                                 rng{random_seed_seq::get_instance()}
{
    if (vth == 0.0)
    {
        vth = std::numeric_limits<double>::epsilon();
    }
    nsp = 60;
    std::vector<double> cutoffs = {*std::max_element(vd.begin(), vd.end()) - 5 * vth,
                                   *std::max_element(vd.begin(), vd.end()) + 5 * vth};
    double vth2 = vth * vth;
    double factor = (1.0 / (sqrt(2. * M_PI * vth2)));
    auto num_facets = facets.size();
    InFlux::num_particles.resize(num_facets);
    vdf.resize(num_facets);
    maxwellian.resize(num_facets);

    dim = vd.size();
    v0 = cutoffs[0];
    dv = (cutoffs[1] - cutoffs[0]) / nsp;
    std::vector<double> vdfv(nsp);

    for (auto i = 0; i < num_facets; ++i)
    {
        auto n = facets[i].normal;
        std::vector<double> vdn(dim);
        vdn[0] = std::inner_product(n.begin(), n.end(), vd.begin(), 0.0);
        for (int j = 1; j < dim; ++j)
        {
            for (int k = 0; k < dim; ++k)
            {
                vdn[j] += facets[i].basis[k * dim + j] * vd[k];
            }
        }

        InFlux::num_particles[i] = facets[i].area * (vth / (sqrt(2 * M_PI)) *
                           exp(-0.5 * (vdn[0] / vth) * (vdn[0] / vth)) +
                           0.5 * vdn[0] * (1. + erf(vdn[0] / (sqrt(2) * vth))));

        vdf[i] = [vth2, vdn, factor](double v) {
            return (v>0)*v*factor* exp(-0.5 * (v - vdn[0]) * (v - vdn[0]) / vth2);
        };

        for (auto j = 0; j < nsp; ++j)
        {
            vdfv[j] = vdf[i](v0 + j * dv);
        }

        std::transform(vdfv.begin(), vdfv.end() - 1, vdfv.begin() + 1,
                       std::back_inserter(pdf_max),
                       [](double a, double b) { return std::max(a, b); });

        auto normalization_factor = std::accumulate(pdf_max.begin() + i * (nsp - 1),
                                                    pdf_max.begin() + (i + 1) * (nsp - 1),
                                                    0.0);

        std::partial_sum(pdf_max.begin() + i * (nsp - 1),
                         pdf_max.begin() + (i + 1) * (nsp - 1),
                         std::back_inserter(cdf));

        std::transform(cdf.begin() + i * (nsp - 1),
                       cdf.begin() + (i + 1) * (nsp - 1),
                       cdf.begin() + i * (nsp - 1),
                       std::bind1st(std::multiplies<double>(),
                       1. / normalization_factor));

        maxwellian[i] = [vth, vdn](double v, int k){
            return vdn[k] - sqrt(2.0) * vth * boost::math::erfc_inv(2 * v);
        };
    }
}

std::vector<double> MaxwellianFlux::sample(const std::size_t N, const std::size_t f)
{
    random_source rng(random_seed_seq::get_instance());
    std::vector<double> vs(N * dim), vs_new(dim);
    int index, n = 0;
    double p_vs, value;
    while (n < N)
    {
        index = std::distance(cdf.begin() + f * (nsp - 1),
                std::lower_bound(cdf.begin() + f * (nsp - 1),
                                 cdf.begin() + (f + 1) * (nsp - 1),
                                 dist(rng)));

        vs_new[0] = v0 + dv * (index + dist(rng));
        value = vdf[f](vs_new[0]);
        p_vs = pdf_max[index + f * (nsp - 1)] * dist(rng);
        if (p_vs < value)
        {
            for (int k = 1; k < dim; ++k)
            {
                auto r = dist(rng);
                vs_new[k] = maxwellian[f](r, k);
            }
            for (int i = 0; i < dim; ++i)
            {
                for (int j = 0; j < dim; ++j)
                {
                    vs[n * dim + i] += facets[f].basis[i * dim + j] * vs_new[j];
                }
            }
            n += 1;
        }
    }
    return vs;
}

signed long int locate_mesh(std::shared_ptr<const df::Mesh> mesh, std::vector<double> x)
{
    auto dim = mesh->geometry().dim();
    auto tree = mesh->bounding_box_tree();
    df::Point p(dim, x.data());
    unsigned int cell_id = tree->compute_first_entity_collision(p);

    if (cell_id == UINT32_MAX)
    {
        return -1;
    }
    else
    {
        return cell_id;
    }
}

std::function<double(std::vector<double> &)> create_mesh_pdf(std::function<double(std::vector<double> &)> pdf,
                                                             std::shared_ptr<const df::Mesh> mesh)
{
    auto mesh_pdf = [mesh, pdf](std::vector<double> x) -> double {
            return (locate_mesh(mesh, x) >= 0)*pdf(x);
        };

    return mesh_pdf;
}

std::vector<double> random_domain_points(
    std::function<double(std::vector<double> &)> pdf,
    double pdf_max, int N,
    std::shared_ptr<const df::Mesh> mesh)
{
    auto mesh_pdf = create_mesh_pdf(pdf, mesh);

    auto D = mesh->geometry().dim();
    auto coordinates = mesh->coordinates();
    int num_vertices = mesh->num_vertices();
    auto Ld_min = *std::min_element(coordinates.begin(), coordinates.end());
    auto Ld_max = *std::max_element(coordinates.begin(), coordinates.end());

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    std::vector<std::uniform_real_distribution<double>> dists(D);

    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, pdf_max);

    for (int i = 0; i < D; ++i)
    {
        dists[i] = distribution(Ld_min, Ld_max);
    }

    std::vector<double> xs(N * D), v(D);
    int n = 0;
    while (n < N)
    {
        for (int i = 0; i < D; ++i)
        {
            v[i] = dists[i](rng);
        }
        if (dist(rng) < mesh_pdf(v))
        {
            for (int i = n * D; i < (n + 1) * D; ++i)
            {
                xs[i] = v[i % D];
            }
            n += 1;
        }
    }
    return xs;
}

std::vector<double> random_facet_points(const int N, std::vector<double> &facet_vertices)
{
    auto size = facet_vertices.size();
    auto dim = sqrt(size);
    std::vector<double> xs(N * dim), v(dim);

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, 1.0);

    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < dim; ++j)
        {
            xs[i*dim+j] = facet_vertices[j];
        }
        for (std::size_t k = 1; k < dim; ++k)
        {
            auto r = dist(rng);
            if(k==dim-k+1)
            {
                r = 1.0 - sqrt(r);
            }
            for (std::size_t j = 0; j < dim; ++j)
            {
                xs[i * dim + j] += r * (facet_vertices[k*dim+j] - xs[i * dim + j]);
            }

        }
    }
    return xs;
}

std::vector<double> maxwellian(double vth, std::vector<double> vd, std::size_t D,
                               const int &N)
{
    if (vth == 0.0)
    {
        vth = std::numeric_limits<double>::epsilon();
    }
    using namespace boost::math;
    typedef std::mt19937_64 random_source;
    random_source rng{random_seed_seq::get_instance()};
    typedef std::uniform_real_distribution<double> distribution;
    distribution random(0.0, 1.0);

    double r;
    auto cdf = [vth, vd](double &v, int i) { return vd[i] - sqrt(2.0) * vth * erfc_inv(2 * v); };
    int i, j;
    std::vector<double> vs(N * D);
    for (j = 0; j < D; ++j)
    {
        for (i = 0; i < N; ++i)
        {
            r = random(rng);
            vs[i * D + j] = cdf(r, j);
        }
    }
    return vs;
}

class Particle
{
  public:
    std::vector<double> x;
    std::vector<double> v;
    double q;
    double m;
    Particle(std::vector<double> &x,
             std::vector<double> &v,
             double q,
             double m) : x(x), v(v), q(q), m(m) {}
};

class Species
{
  public:
    double q;
    double m;
    double n;
    int num;
    double vth;
    std::vector<double> vd;
    std::function<double(std::vector<double> &)> pdf;
    double pdf_max;
    std::function<double(std::vector<double> &)> vdf;

    std::shared_ptr<InFlux> flux;

    Species(double q, double m, double n, int num, double vth,
            std::vector<double> &vd,
            std::function<double(std::vector<double> &)> pdf,
            double pdf_max, std::vector<Facet> &facets,
            VDFType vdf_type=VDFType::Maxwellian) : q(q), m(m), n(n), num(num),
            vth(vth), vd(vd), pdf(pdf), pdf_max(pdf_max)
    {
        if (vdf_type==VDFType::Maxwellian)
        {
            this->flux = std::make_shared<MaxwellianFlux>(vth, vd, facets);
            this->vdf = maxwellian_vdf(vth, vd);
        }
    }
};

class SpeciesList
{
  public:
    std::vector<Facet> facets;
    double X;
    int D;
    double _volume, num_cells;
    std::vector<Species> species;
    double T = std::numeric_limits<double>::quiet_NaN();
    double Q = boost::units::si::constants::codata::e / boost::units::si::coulomb;
    double M = std::numeric_limits<double>::quiet_NaN();
    double epsilon_0 = boost::units::si::constants::codata::epsilon_0*boost::units::si::meter/boost::units::si::farad;//8.854187817620389850536563031710750260608e-12;

    SpeciesList(std::shared_ptr<const df::Mesh> &mesh,
                std::vector<Facet> &facets, double X);

    void append_raw(double q, double m, double n, int npc, double vth,
                    std::vector<double> &vd,
                    std::function<double(std::vector<double> &)> pdf,
                    double pdf_max);

    void append(double q, double m, double n, int npc, double vth,
                std::vector<double> &vd,
                std::function<double(std::vector<double> &)> pdf,
                double pdf_max);
};

SpeciesList::SpeciesList(std::shared_ptr<const df::Mesh> &mesh,
                         std::vector<Facet> &facets, double X) : facets(facets), X(X)
{
    this->D = mesh->geometry().dim();
    //auto one = std::make_shared<df::Constant>(1);
    //volume::Form_A form(mesh, one);
    this->_volume = volume(mesh);//df::assemble(form);
    this->num_cells = mesh->num_cells();
}

void SpeciesList::append_raw(double q, double m, double n, int npc, double vth,
                             std::vector<double> &vd,
                             std::function<double(std::vector<double> &)> pdf,
                             double pdf_max)
{
    double num = npc * this->num_cells;
    double w = (n / num) * this->_volume;
    q *= w;
    m *= w;
    n /= w;

    Species s(q, m, n, num, vth, vd, pdf, pdf_max, facets);
    species.emplace_back(s);
}

void SpeciesList::append(double q, double m, double n, int npc, double vth,
                         std::vector<double> &vd,
                         std::function<double(std::vector<double> &)> pdf,
                         double pdf_max)
{
    if (std::isnan(this->T))
    {
        double wp = sqrt((n * q * q) / (epsilon_0 * m));
        this->T = 1.0 / wp;
    }
    if (std::isnan(this->M))
    {
        this->M = (this->T * this->T * this->Q * this->Q) /
                  (this->epsilon_0 * pow(this->X, this->D));
    }
    q /= this->Q;
    m /= this->M;
    n *= pow(this->X, this->D);
    if (vth == 0.0)
    {
        vth = std::numeric_limits<double>::epsilon();
    }
    vth /= (this->X / this->T);
    for (int i = 0; i < this->D; ++i)
    {
        vd[i] /= (this->X / this->T);
    }
    this->append_raw(q, m, n, npc, vth, vd, pdf, pdf_max);
}

class Cell
{
  public:
    std::size_t id;
    std::vector<std::size_t> neighbors;
    std::vector<signed long int> facet_adjacents;
    std::vector<double> facet_normals;
    std::vector<double> facet_mids;
    std::vector<Particle> particles;

    Cell() {};

    Cell(std::size_t id, std::vector<std::size_t> neighbors)
    : id(id), neighbors(neighbors) {};
};

class Population
{
  public:
    std::shared_ptr<const df::Mesh> mesh;
    std::size_t num_cells;
    std::vector<Cell> cells;
    std::size_t gdim, tdim;

    Population(std::shared_ptr<const df::Mesh> &mesh,
               const df::MeshFunction<std::size_t> &bnd);
    void init_localizer(const df::MeshFunction<std::size_t> &bnd);
    void add_particles(std::vector<double> &xs, std::vector<double> &vs,
                       double q, double m);
    signed long int locate(std::vector<double> &p);
    signed long int relocate(std::vector<double> &p, signed long int cell_id);
    void update(boost::optional<std::vector<Object>& > objects = boost::none);
    std::size_t num_of_particles();
    std::size_t num_of_positives();
    std::size_t num_of_negatives();
    void save_file(const std::string &fname);
    void load_file(const std::string &fname);
    void save_vel(const std::string &fname);
};


Population::Population(std::shared_ptr<const df::Mesh> &mesh,
                       const df::MeshFunction<std::size_t> &bnd)
{
    this->mesh = mesh;
    num_cells = mesh->num_cells();
    std::vector<Cell> cells(num_cells);
    this->cells = cells;
    this->tdim = mesh->topology().dim();
    this->gdim = mesh->geometry().dim();

    this->mesh->init(0, this->tdim);
    for (df::MeshEntityIterator e(*(this->mesh), this->tdim); !e.end(); ++e)
    {
        std::vector<std::size_t> neighbors;
        auto cell_id = e->index();
        auto vertices = e->entities(0);
        auto num_vertices = e->num_entities(0);
        for (std::size_t i = 0; i < num_vertices; ++i)
        {
            df::Vertex vertex(*mesh, e->entities(0)[i]);
            auto vertex_cells = vertex.entities(tdim);
            auto num_adj_cells = vertex.num_entities(tdim);
            for (std::size_t j = 0; j < num_adj_cells; ++j)
            {
                if (cell_id != vertex_cells[j])
                {
                    neighbors.push_back(vertex_cells[j]);
                }
            }
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        Cell cell(cell_id, neighbors);
        this->cells[cell_id] = cell;
    }

    this->init_localizer(bnd);
}

void Population::init_localizer(const df::MeshFunction<std::size_t> &bnd)
{
    mesh->init(tdim - 1, tdim);
    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        std::vector<signed long int> facet_adjacents;
        std::vector<double> facet_normals;
        std::vector<double> facet_mids;

        auto cell_id = e->index();
        df::Cell single_cell(*mesh, cell_id);
        auto facets = e->entities(tdim - 1);
        auto num_facets = e->num_entities(tdim - 1);
        for (std::size_t i = 0; i < num_facets; ++i)
        {
            df::Facet facet(*mesh, e->entities(tdim - 1)[i]);
            auto facet_cells = facet.entities(tdim);
            auto num_adj_cells = facet.num_entities(tdim);
            for (std::size_t j = 0; j < num_adj_cells; ++j)
            {
                if (cell_id != facet_cells[j])
                {
                    facet_adjacents.push_back(facet_cells[j]);
                }
            }
            if (num_adj_cells == 1)
            {
                facet_adjacents.push_back(-1 * bnd.values()[facets[i]]);
            }

            for (std::size_t j = 0; j < gdim; ++j)
            {
                facet_mids.push_back(facet.midpoint()[j]);
                facet_normals.push_back(single_cell.normal(i)[j]);
            }
        }

        cells[cell_id].facet_adjacents = facet_adjacents;
        cells[cell_id].facet_normals = facet_normals;
        cells[cell_id].facet_mids = facet_mids;
    }
}

void Population::add_particles(std::vector<double> &xs, std::vector<double> &vs,
                               double q, double m)
{
    std::size_t num_particles = xs.size() / gdim;
    std::vector<double> xs_tmp(gdim);
    std::vector<double> vs_tmp(gdim);
    std::size_t cell_id;
    for (std::size_t i = 0; i < num_particles; ++i)
    {
        for (std::size_t j = 0; j < gdim; ++j)
        {
            xs_tmp[j] = xs[i * gdim + j];
            vs_tmp[j] = vs[i * gdim + j];
        }
        cell_id = locate(xs_tmp);
        if (cell_id >= 0)
        {
            Particle particle(xs_tmp, vs_tmp, q, m);
            cells[cell_id].particles.push_back(particle);
        }
    }
}

signed long int Population::locate(std::vector<double> &p)
{
    return locate_mesh(mesh, p);
}

signed long int Population::relocate(std::vector<double> &p, signed long int cell_id)
{
    df::Cell _cell_(*mesh, cell_id);
    df::Point point(gdim, p.data());
    if (_cell_.contains(point))
    {
        return cell_id;
    }
    else
    {
        std::vector<double> proj(gdim + 1);
        for (std::size_t i = 0; i < gdim + 1; ++i)
        {
            proj[i] = 0.0;
            for (std::size_t j = 0; j < gdim; ++j)
            {
                proj[i] += (p[j] - cells[cell_id].facet_mids[i * gdim + j]) *
                        cells[cell_id].facet_normals[i * gdim + j];
            }
        }
        auto projarg = std::distance(proj.begin(), std::max_element(proj.begin(), proj.end()));
        auto new_cell_id = cells[cell_id].facet_adjacents[projarg];
        if (new_cell_id >= 0)
        {
            return relocate(p, new_cell_id);
        }
        else
        {
            return new_cell_id;
        }
    }
}
void Population::update(boost::optional<std::vector<Object>& > objects)
{
    std::size_t num_objects = 0;
    if(objects)
    {
        num_objects = objects->size();
    }
    signed long int new_cell_id;
    for (signed long int cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::vector<std::size_t> to_delete;
        std::size_t num_particles = cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = cells[cell_id].particles[p_id];
            new_cell_id = relocate(particle.x, cell_id);
            if (new_cell_id != cell_id)
            {
                to_delete.push_back(p_id);
                if (new_cell_id >= 0)
                {
                    cells[new_cell_id].particles.push_back(particle);
                }else{
                    for (auto i = 0; i < num_objects; ++i)
                    {
                        if(-new_cell_id == objects.get()[i].id)
                        {
                            objects.get()[i].charge += particle.q;
                        }
                    }
                }
            }
        }
        std::size_t size_to_delete = to_delete.size();
        for (std::size_t it = size_to_delete; it-- > 0;)
        {
            auto p_id = to_delete[it];
            if (p_id == num_particles - 1)
            {
                cells[cell_id].particles.pop_back();
            }
            else
            {
                std::swap(cells[cell_id].particles[p_id], cells[cell_id].particles.back());
                cells[cell_id].particles.pop_back();
            }
        }
    }
}

std::size_t Population::num_of_particles()
{
    std::size_t num_particles = 0;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        num_particles += cells[cell_id].particles.size();
    }
    return num_particles;
}

std::size_t Population::num_of_positives()
{
    std::size_t num_positives = 0;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = cells[cell_id].particles[p_id];
            if (particle.q > 0)
            {
                num_positives++;
            }
        }
    }
    return num_positives;
}

std::size_t Population::num_of_negatives()
{
    std::size_t num_negatives = 0;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = cells[cell_id].particles[p_id];
            if (particle.q < 0)
            {
                num_negatives++;
            }
        }
    }
    return num_negatives;
}

void Population::save_vel(const std::string &fname)
{
    FILE *fout = fopen(fname.c_str(), "w");
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = cells[cell_id].particles.size();
        if (num_particles > 0)
        {
            for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
            {
                auto particle = cells[cell_id].particles[p_id];
                for (std::size_t i = 0; i < gdim; ++i)
                {
                    fprintf(fout, "%.17g\t", particle.v[i]);
                }
                fprintf(fout, "\n");
            }
        }
    }
    fclose(fout);
}

void Population::save_file(const std::string &fname)
{
    FILE *fout = fopen(fname.c_str(), "w");
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = cells[cell_id].particles.size();
        if (num_particles > 0)
        {
            for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
            {
                auto particle = cells[cell_id].particles[p_id];
                for (std::size_t i = 0; i < gdim; ++i)
                {
                    fprintf(fout, "%.17g\t", particle.x[i]);
                }
                for (std::size_t i = 0; i < gdim; ++i)
                {
                    fprintf(fout, "%.17g\t", particle.v[i]);
                }
                fprintf(fout, "%.17g\t %.17g\t", particle.q, particle.m);
                fprintf(fout, "\n");
            }
        }
    }
    fclose(fout);
}

void Population::load_file(const std::string &fname)
{
    std::fstream in(fname);
    std::string line;
    std::vector<double> x(gdim);
    std::vector<double> v(gdim);
    double q, m;
    int i;
    while (std::getline(in, line))
    {
        double value;
        std::stringstream ss(line);
        i = 0;
        while (ss >> value)
        {
            if (i < gdim)
            {
                x[i] = value;
            }
            else if (i >= gdim && i < 2 * gdim)
            {
                v[i % gdim] = value;
            }
            else if (i == 2 * gdim)
            {
                q = value;
            }
            else if (i == 2 * gdim + 1)
            {
                m = value;
            }
            ++i;
            add_particles(x,v,q,m);
        }
    }
}

void inject_particles(Population &pop, SpeciesList &listOfSpecies,
                      std::vector<Facet> &facets, const double dt)
{
    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, 1.0);

    auto dim = pop.gdim;
    std::size_t num_species = listOfSpecies.species.size();
    auto num_facets = facets.size();
    std::vector<double> xs_tmp(dim);

    for (std::size_t i = 0; i < num_species; ++i)
    {
        std::vector<double> xs, vs;
        int len=0;
        for(std::size_t j = 0; j < num_facets; ++j)
        {
            auto normal_i = facets[j].normal;
            int N = int(listOfSpecies.species[i].n*dt*listOfSpecies.species[i].flux->num_particles[j]);
            if (dist(rng) < (listOfSpecies.species[i].n * dt * listOfSpecies.species[i].flux->num_particles[j]-N))
            {
                N += 1;
            }
            auto count = 0;
            auto outside = 0;
            while (count <N)
            {
                auto n = N - count;
                auto xs_new = random_facet_points(n, facets[j].vertices);
                auto vs_new = listOfSpecies.species[i].flux->sample(n, j);

                for(auto k=0; k<n; ++k)
                {
                    auto r = dist(rng);
                    for (auto l = 0; l <dim; ++l)
                    {
                        xs_tmp[l] = xs_new[k*dim + l] + dt*r*vs_new[k*dim + l];
                    }
                    if (pop.locate(xs_tmp) >= 0)
                    {
                        for (auto l = 0; l < dim; ++l)
                        {
                            xs.push_back(xs_tmp[l]);
                            vs.push_back(vs_new[k * dim + l]);
                        }
                    }else{
                        outside +=1;
                    }
                    count += 1;
                }
            }
        }
        pop.add_particles(xs, vs, listOfSpecies.species[i].q, listOfSpecies.species[i].m);
    }
}

void load_particles(Population &pop, SpeciesList &listOfSpecies)
{
    auto gdim = listOfSpecies.D;
    std::size_t num_species = listOfSpecies.species.size();
    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto s = listOfSpecies.species[i];
        auto xs = random_domain_points(s.pdf, s.pdf_max, s.num, pop.mesh);
        auto vs = maxwellian(s.vth, s.vd, gdim, s.num);
        pop.add_particles(xs, vs, s.q, s.m);
    }
}

std::vector<double> voronoi_volume_approx(const df::FunctionSpace &V)
{
    auto num_dofs = V.dim();
    auto dof_indices = df::vertex_to_dof_map(V);
    std::vector<double> volumes(num_dofs, 0.0);

    auto mesh = V.mesh();
    auto tdim = mesh->topology().dim();
    auto gdim = mesh->geometry().dim();
    int j = 0;
    mesh->init(0, tdim);
    for (df::MeshEntityIterator e(*mesh, 0); !e.end(); ++e)
    {
        auto cells = e->entities(tdim);
        auto num_cells = e->num_entities(tdim);
        for (std::size_t i = 0; i < num_cells; ++i)
        {
            df::Cell cell(*mesh, e->entities(tdim)[i]);
            volumes[dof_indices[j]] += cell.volume();
        }
        j++;
    }
    for (std::size_t i = 0; i < num_dofs; ++i)
    {
        volumes[i] = (gdim + 1.0) / volumes[i];
    }
    return volumes;
}

df::Function distribute(const df::FunctionSpace &V,
                        Population &pop,
                        const std::vector<double> &dv_inv)
{
    auto mesh = V.mesh();
    auto tdim = mesh->topology().dim();
    df::Function rho(std::make_shared<const df::FunctionSpace>(V));
    auto rho_vec = rho.vector();
    std::size_t len_rho = rho_vec->size();
    std::vector<double> rho0(len_rho);
    rho_vec->get_local(rho0);

    auto element = V.element();
    auto s_dim = element->space_dimension();

    std::vector<double> basis_matrix(s_dim);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();
        auto dof_id = V.dofmap()->cell_dofs(cell_id);

        std::vector<double> basis(1);
        std::vector<double> accum(s_dim, 0.0);

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];

            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x.data(),
                                        vertex_coordinates.data(),
                                        cell_orientation);
                basis_matrix[i] = basis[0];
                accum[i] += particle.q * basis_matrix[i];
            }

        }
        for (std::size_t i = 0; i < s_dim; ++i)
        {
            rho0[dof_id[i]] += accum[i];
        }
    }
    for (std::size_t i = 0; i < len_rho; ++i)
    {
        rho0[i] *= dv_inv[i];
    }
    rho.vector()->set_local(rho0);
    return rho;
}

double kinetic_energy(Population &pop)
{
    double KE = 0.0;
    for (df::MeshEntityIterator e(*pop.mesh, pop.tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            auto m = particle.m;
            auto v = particle.v;
            for (std::size_t i = 0; i < pop.gdim; ++i)
            {
                KE += 0.5 * m * v[i] * v[i];
            }
        }
    }
}

double mesh_potential_energy(df::Function &phi, df::Function &rho)
{
    auto mesh = phi.function_space()->mesh();
    auto dim = mesh->geometry().dim();
    auto phis = std::make_shared<df::Function>(phi);
    auto rhos = std::make_shared<df::Function>(rho);
    std::shared_ptr<df::Form> energy;
    if (dim == 1)
    {
        energy = std::make_shared<Energy::Form_0>(mesh, phis, rhos);
    }
    else if (dim == 2)
    {
        energy = std::make_shared<Energy::Form_1>(mesh, phis, rhos);
    }
    else if (dim == 3)
    {
        energy = std::make_shared<Energy::Form_2>(mesh, phis, rhos);
    }
    // Energy::Functional energy_form(mesh);
    // energy_form.phi = std::make_shared<df::Function>(phi);
    // energy_form.rho = std::make_shared<df::Function>(rho);
    return 0.5 * df::assemble(*energy);
}

double particle_potential_energy(Population &pop, const df::Function &phi)
{
    auto V = phi.function_space();
    auto mesh = V->mesh();
    auto tdim = mesh->topology().dim();
    auto element = V->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double PE = 0.0;

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        phi.restrict(&coefficients[0], *element, _cell,
                    vertex_coordinates.data(), ufc_cell);

        std::vector<double> basis(v_dim);
        basis_matrix.resize(v_dim);
        for (std::size_t i = 0; i < v_dim; ++i)
        {
            basis_matrix[i].resize(s_dim);
        }

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            std::vector<double> phii(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x.data(),
                                        vertex_coordinates.data(),
                                        cell_orientation);

                for (std::size_t j = 0; j < v_dim; ++j)
                {
                    basis_matrix[j][i] = basis[j];
                }
            }
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                for (std::size_t j = 0; j < v_dim; j++)
                {
                    phii[j] += coefficients[i] * basis_matrix[j][i];
                }
            }
            auto q = particle.q;
            for (std::size_t j = 0; j < v_dim; j++)
            {
                PE += 0.5 * q * phii[j];
            }
        }
    }
    return PE;
}


double accel(Population &pop, const df::Function &E, const double dt)
{
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto tdim = mesh->topology().dim();
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(&coefficients[0], *element, _cell,
                    vertex_coordinates.data(), ufc_cell);

        std::vector<double> basis(v_dim);
        basis_matrix.resize(v_dim);
        for (std::size_t i = 0; i < v_dim; ++i)
        {
            basis_matrix[i].resize(s_dim);
        }

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            std::vector<double> Ei(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x.data(),
                                        vertex_coordinates.data(),
                                        cell_orientation);

                for (std::size_t j = 0; j < v_dim; ++j)
                {
                    basis_matrix[j][i] = basis[j];
                }
            }
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                for (std::size_t j = 0; j < v_dim; j++)
                {
                    Ei[j] += coefficients[i] * basis_matrix[j][i];
                }
            }

            auto m = particle.m;
            auto q = particle.q;
            auto vel = particle.v;

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] *= dt * (q / m);
                KE += 0.5 * m * vel[j] * (vel[j] + Ei[j]);
            }
            for (std::size_t j = 0; j < v_dim; j++)
            {
                pop.cells[cell_id].particles[p_id].v[j] += Ei[j];
            }
        }
    }
    return KE;
}

double boris(Population &pop, df::Function &E,
             const std::vector<double> &B, const double dt)
{
    auto dim = B.size();
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto tdim = mesh->topology().dim();
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;
    double t_mag2;

    std::vector<double> v_minus(dim), v_prime(dim), v_plus(dim);
    std::vector<double> t(dim), s(dim);

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(&coefficients[0], *element, _cell,
                    vertex_coordinates.data(), ufc_cell);

        std::vector<double> basis(v_dim);
        basis_matrix.resize(v_dim);
        for (std::size_t i = 0; i < v_dim; ++i)
        {
            basis_matrix[i].resize(s_dim);
        }

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            std::vector<double> Ei(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x.data(),
                                        vertex_coordinates.data(),
                                        cell_orientation);

                for (std::size_t j = 0; j < v_dim; ++j)
                {
                    basis_matrix[j][i] = basis[j];
                }
            }
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                for (std::size_t j = 0; j < v_dim; j++)
                {
                    Ei[j] += coefficients[i] * basis_matrix[j][i];
                }
            }

            auto m = particle.m;
            auto q = particle.q;
            auto vel = particle.v;

            for (auto i = 0; i < dim; ++i)
            {
                t[dim] = (dt * q / (2.0 * m)) * B[i];
            }

            t_mag2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];

            for (auto i = 0; i < dim; ++i)
            {
                s[i] = 2 * t[dim] / (1 + t_mag2);
            }

            for (auto i = 0; i < dim; ++i)
            {
                v_minus[i] = vel[i] + 0.5 * dt * (q / m) * Ei[i];
            }

            for (auto i = 0; i < dim; i++)
            {
                KE += 0.5 * m * v_minus[i] * v_minus[i];
            }

            auto v_minus_cross_t = cross(v_minus, t);
            for (auto i = 0; i < dim; ++i)
            {
                v_prime[i] = v_minus[i] + v_minus_cross_t[i];
            }

            auto v_prime_cross_s = cross(v_prime, s);
            for (auto i = 0; i < dim; ++i)
            {
                v_plus[i] = v_minus[i] + v_prime_cross_s[i];
            }

            for (auto i = 0; i < dim; ++i)
            {
                pop.cells[cell_id].particles[p_id].v[i] = v_plus[i] + 0.5 * dt * (q / m) * Ei[i];
            }
        }
    }
    return KE;
}

void move_periodic(Population &pop, const double dt, const std::vector<double> &Ld)
{
    auto dim = Ld.size();
    auto num_cells = pop.num_cells;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t j = 0; j < dim; ++j)
            {
                pop.cells[cell_id].particles[p_id].x[j] += dt * pop.cells[cell_id].particles[p_id].v[j];
                pop.cells[cell_id].particles[p_id].x[j] -= Ld[j] * floor(pop.cells[cell_id].particles[p_id].x[j] / Ld[j]);
            }
        }
    }
}

void move(Population &pop, const double dt)
{
    auto dim = pop.gdim;
    auto num_cells = pop.num_cells;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t j = 0; j < dim; ++j)
            {
                pop.cells[cell_id].particles[p_id].x[j] += dt * pop.cells[cell_id].particles[p_id].v[j];
            }
        }
    }
}

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

int main()
{
    Timer timer;
    timer.reset();
    double dt = 0.04;
    std::size_t steps = 20;

    std::string fname{"/home/diako/Documents/Software/punc/mesh/2D/nothing_in_square"};

    auto mesh = load_mesh(fname);
    auto D = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();

    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto facet_vec = exterior_boundaries(boundaries, ext_bnd_id);

    bool remove_null_space = true;
    std::vector<double> Ld = get_mesh_size(mesh);
    std::vector<bool> periodic(D);
    std::vector<double> vd(D);
    for (std::size_t i = 0; i<D; ++i)
    {
        periodic[i] = true;
        vd[i] = 0.0;
    }

    auto constr = std::make_shared<PeriodicBoundary>(Ld, periodic);
    Potential2D::FunctionSpace V(mesh, constr);

    // df::Function phi(std::make_shared<df::FunctionSpace>(V));
    PoissonSolver poisson(V, boost::none, remove_null_space);
    ESolver esolver(V);

    auto dv_inv = voronoi_volume_approx(V);

    double vth = 0.0;
    int npc = 32;

    SpeciesList listOfSpecies(mesh, facet_vec, Ld[0]);

    double A = 0.5, mode = 1.0;
    double pdf_max = 1.0+A;

    auto pdfe = [A, mode, Ld](std::vector<double> t)->double{return 1.0+A*sin(2*mode*M_PI*t[0]/Ld[0]);};
    auto pdfi = [](std::vector<double> t)->double{return 1.0;};

    double e = boost::units::si::constants::codata::e / boost::units::si::coulomb;
    double me = boost::units::si::constants::codata::m_e / boost::units::si::kilograms;
    double mass_ratio = boost::units::si::constants::codata::m_e_over_m_p / boost::units::si::dimensionless();
    double mi = me/mass_ratio;

    listOfSpecies.append(-e, me, 100, npc, vth, vd, pdfe, pdf_max);
    listOfSpecies.append(e, mi, 100, npc, vth, vd, pdfi, 1.0);

    Population pop(mesh, boundaries);

    load_particles(pop, listOfSpecies);
    auto num1 = pop.num_of_positives();
    auto num2 = pop.num_of_negatives();
    auto num3 = pop.num_of_particles();
    std::cout << "+:  " << num1 << " - " << num2 << " total: " << num3 << '\n';

    std::vector<double> KE(steps-1);
    std::vector<double> PE(steps-1);
    std::vector<double> TE(steps-1);
    double KE0 = kinetic_energy(pop);

    auto t0 = timer.elapsed();
    printf("Initilazation: %e\n", t0);
    timer.reset();
    for(int i=1; i<steps;++i)
    {
        std::cout<<"step: "<< i<<'\n';
        auto rho = distribute(V, pop, dv_inv);
        // df::plot(rho);
        // df::interactive();
        auto phi = poisson.solve(rho);
        auto E = esolver.solve(phi);
        PE[i - 1] = particle_potential_energy(pop, phi);
        KE[i-1] = accel(pop, E, (1.0-0.5*(i == 1))*dt);
        move_periodic(pop, dt, Ld);
        pop.update();
    }
    t0 = timer.elapsed();
    printf("Loop: %e\n", t0);
    KE[0] = KE0;
    for(int i=0;i<KE.size(); ++i)
    {
        TE[i] = PE[i] + KE[i];
    }
    std::ofstream out;
    out.open("PE.txt");
    for (const auto &e : PE) out << e << "\n";
    out.close();
    std::ofstream out1;
    out1.open("KE.txt");
    for (const auto &e : KE) out1 << e << "\n";
    out1.close();
    std::ofstream out2;
    out2.open("TE.txt");
    for (const auto &e : TE) out2 << e << "\n";
    out2.close();

    return 0;
}
