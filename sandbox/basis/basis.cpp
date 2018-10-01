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
#include "../../punc/ufl/Potential1D.h"
#include "../../punc/ufl/Potential2D.h"
#include "../../punc/ufl/Potential3D.h"

namespace df = dolfin;

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
    int j;
    double norm;
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
                    normal[j] = -1 * cell.normal(i, j);
                    basis[j*gdim] = normal[j];
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
        norm = 0.0;
        for (std::size_t i = 0; i < gdim; ++i)
        {
            vertex[i] = vertices[i] - vertices[gdim+i];
            norm += vertex[i]*vertex[i];
        }
        for (std::size_t i = 0; i < gdim; ++i)
        {
            vertex[i] /= sqrt(norm);
            basis[i*gdim+1] = vertex[i];
        }

        if (gdim==3)
        {
            basis[2] = normal[1] * vertex[2] - normal[2] * vertex[1];
            basis[5] = normal[0] * vertex[2] - normal[2] * vertex[0];
            basis[8] = normal[0] * vertex[1] - normal[1] * vertex[0];
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

class PoissonSolver
{
public:
    PoissonSolver(const df::FunctionSpace &V,
                  boost::optional<std::vector<df::DirichletBC>& > ext_bc = boost::none,
                  bool remove_null_space = false,
                  std::string method = "gmres",
                  std::string preconditioner = "hypre_amg");

    df::Function solve(const df::Function &rho);

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

int main()
{

    std::string fname{"/home/diako/Documents/cpp/punc/mesh/2D/nothing_in_square"};
    // std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/injection/mesh/box"};

    auto mesh = load_mesh(fname);
    auto boundaries = load_boundaries(mesh, fname);
    auto gdim = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    printf("gdim: %zu, tdim: %zu \n", gdim, tdim);
    // df::plot(mesh);
    // df::interactive();
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto ext_facets = exterior_boundaries(boundaries, ext_bnd_id);

    auto num_facets = ext_facets.size();
    // for(int i = 0; i<num_facets; ++i)
    // {
    //     for(auto e:ext_facets[i].normal)
    //     {
    //         std::cout<<e<<"  ";
    //     }
    //     std::cout<<'\n';
    //     for(auto e:ext_facets[i].vertices)
    //     {
    //         std::cout<<e<<"  ";
    //     }
    //     std::cout<<'\n';
    //     for(auto e:ext_facets[i].basis)
    //     {
    //         std::cout<<e<<"  ";
    //     }
    //     std::cout<<'\n';
    //     std::cout<<"----------------------------------------------------"<<'\n';
    // }
    std::vector<int> v = {0,1,2,3,4,5,6,7,8,9,10};
    auto ind = std::distance(v.begin()+3,
                        std::lower_bound(v.begin()+3, v.begin()+6, 5));
    std::cout<<v[ind]<<'\n';

    Potential2D::FunctionSpace V(mesh);

    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V),
                       std::make_shared<df::Constant>(1.0),
                       std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
                       ext_bnd_id);
    std::vector<df::DirichletBC> ext_bc{bc};

    df::Function rho(std::make_shared<df::FunctionSpace>(V));

    PoissonSolver solver(V, ext_bc);
    auto phi = solver.solve(rho);

    return 0;
}
