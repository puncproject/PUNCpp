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
#include "../../punc/Potential.h"
#include "../../punc/Potential.h"
#include "../../punc/Flux.h"
#include "../../punc/EField.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

namespace df = dolfin;

// std::vector<Facet> exterior_boundaries_old(df::MeshFunction<std::size_t> &boundaries,
//                                        std::size_t ext_bnd_id)
// {
//     auto mesh = boundaries.mesh();
//     auto D = mesh->geometry().dim();
//     auto tdim = mesh->topology().dim();
//     auto values = boundaries.values();
//     auto length = boundaries.size();
//     int num_facets = 0;
//     for (std::size_t i = 0; i < length; ++i)
//     {
//         if (ext_bnd_id == values[i])
//         {
//             num_facets += 1;
//         }
//     }
//     df::SubsetIterator facet_iter(boundaries, ext_bnd_id);
//     std::vector<Facet> facet_vec;
//     double area;
//     std::vector<double> normal(D);
//     std::vector<double> vertices(D * D);
//
//     mesh->init(tdim - 1, tdim);
//     for (; !facet_iter.end(); ++facet_iter)
//     {
//         // area
//         df::Cell cell(*mesh, facet_iter->entities(tdim)[0]);
//         auto cell_facet = cell.entities(tdim - 1);
//         std::size_t num_facets = cell.num_entities(tdim - 1);
//
//         for (std::size_t i = 0; i < num_facets; ++i)
//         {
//             if (cell_facet[i] == facet_iter->index())
//             {
//                 area = cell.facet_area(i);
//             }
//         }
//         // vertices
//         const unsigned int *facet_vertices = facet_iter->entities(0);
//         for (std::size_t i = 0; i < D; ++i)
//         {
//             const df::Point p = mesh->geometry().point(facet_vertices[i]);
//             for (std::size_t j = 0; j < D; ++j)
//             {
//                 vertices[i * D + j] = p[j];
//             }
//         }
//         //normal
//         df::Facet facet(*mesh, facet_iter->index());
//         for (std::size_t i = 0; i < D; ++i)
//         {
//             normal[i] = -1 * facet.normal(i);
//         }
//         facet_vec.push_back(Facet(area, vertices, normal));
//     }
//
//     return facet_vec;
// }

class Facet
{
  public:
    double area;
    std::vector<double> vertices;
    std::vector<double> normal;

    Facet(double area,
          std::vector<double> vertices,
          std::vector<double> normal) :
          area(area),
          vertices(vertices),
          normal(normal) {}
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
        ext_facets.push_back(Facet(area, vertices, normal));
    }
    return ext_facets;
}


std::shared_ptr<const df::Mesh> load_mesh(std::string fname)
{
    // df::Mesh mesh(fname + ".xml");
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

int main()
{
    std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/capacitance/mesh/main02"};

    auto mesh = load_mesh(fname);
    auto boundaries = load_boundaries(mesh, fname);
    auto D = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    printf("gdim: %zu, tdim: %zu \n", D, tdim);

    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];
    std::size_t int_bnd_id = tags[2];

    auto ext_facets = exterior_boundaries(boundaries, ext_bnd_id);

    auto num_facets = ext_facets.size();
    printf("num_facets: %zu \n", num_facets);

    Potential::FunctionSpace V(mesh);

    auto Vptr = std::make_shared<const df::FunctionSpace>(V);
    auto Vptr1 = std::make_shared<const Potential::FunctionSpace>(V);

    auto u0 = std::make_shared<df::Constant>(0.0);
    auto u1 = std::make_shared<df::Constant>(1.0);
	df::DirichletBC bc(std::make_shared<const df::FunctionSpace>(V), u0,
                       std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
                       ext_bnd_id);

   df::DirichletBC object(std::make_shared<const df::FunctionSpace>(V), u1,
                      std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
                      int_bnd_id);

    df::PETScKrylovSolver psolver;
    Potential::BilinearForm a(Vptr,Vptr);
    Potential::LinearForm L(Vptr);
    df::PETScMatrix A;
    df::PETScVector b;

    df::Function rho(Vptr1);
    df::Function phi(Vptr1);

    L.rho = std::make_shared<df::Function>(rho);
    df::assemble(A, a);
    df::assemble(b, L);

    bc.apply(A);
    bc.apply(b);
    object.apply(A);
    object.apply(b);

    psolver.solve(A, *phi.vector(), b);

    std::shared_ptr<EField::FunctionSpace> W = std::make_shared<EField::FunctionSpace>(V.mesh());
    df::PETScKrylovSolver esolver;
    EField::BilinearForm ae(W,W);
    EField::LinearForm Le(W);
    df::PETScMatrix Ae;
    df::PETScVector be;

    df::assemble(Ae, ae);
    Le.phi = std::make_shared<df::Function>(phi);
    df::assemble(be, Le);
    df::Function E(W);
    esolver.solve(Ae, *E.vector(), be);

    std::vector<df::Function> efield;
    efield.emplace_back(E);

    auto tags_e = boundaries.values();
    auto size = boundaries.size();
    df::MeshFunction<std::size_t> bnd;
    bnd = boundaries;
    bnd.set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
    	if(tags_e[i] == int_bnd_id)
    	{
    		bnd.set_value(i, 9999);
    	}
    }

    Flux::Form_flux flux(V.mesh());
    flux.ds = std::make_shared<df::MeshFunction<std::size_t>>(bnd);
    flux.e = std::make_shared<df::Function>(efield[0]);
    auto cap = df::assemble(flux);
    printf("%e\n", cap);
    return 0;
}


// // std::string fname{"/home/diako/Documents/cpp/punc_experimental/prototypes/mesh/square"};
// std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/capacitance/mesh/main02"};
//
// auto mesh = load_mesh(fname);
// auto boundaries = load_boundaries(mesh, fname);
// auto D = mesh->geometry().dim();
// auto tdim = mesh->topology().dim();
// printf("gdim: %zu, tdim: %zu \n", D, tdim);
// // df::plot(mesh);
// // df::interactive();
// auto tags = get_mesh_ids(boundaries);
// std::size_t ext_bnd_id = tags[1];
//
// // timer.reset();
// // auto ext_facets_old = exterior_boundaries_old(boundaries, ext_bnd_id);
// // auto t1 = timer.elapsed();
// timer.reset();
// auto ext_facets = exterior_boundaries(boundaries, ext_bnd_id);
// auto t0 = timer.elapsed();
// // printf("time new: %e, time old: %e \n", t0);
// auto num_facets = ext_facets.size();
// printf("num_facets: %zu \n", num_facets);
//
// Potential::FunctionSpace V(mesh);
//
// //-------------------Define boundary condition------------------------------
// auto u0 = std::make_shared<df::Constant>(0.0);
// df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V), u0,
//                    std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
//                    ext_bnd_id);
//
// // std::vector<df::DirichletBC> ext_bc = {bc};
// // PoissonSolver poisson(V, ext_bc);
// // ESolver esolver(V);
// //
// // auto shared_V = std::make_shared<Potential::FunctionSpace>(V);
// // df::Function rho(shared_V);
// // df::Function phi(shared_V);
// // poisson.solve(phi, rho);
// // auto e_field = esolver.solve(phi);
// //__________________________________________________________________________
//
// //----------Create the object-----------------------------------------------
// Object object(V, boundaries, tags[2]);
// object.set_potential(1.0);
// std::vector<Object> obj = {object};
// //__________________________________________________________________________
//
// //---------------------Capacitance------------------------------------------
// auto r = 0.02;
// auto R = 0.2;
// auto cap = 4.*M_PI*r*R/(R-r);
//
// typedef boost::numeric::ublas::matrix<double> boost_matrix;
//
// boost_matrix inv_capacity = capacitance_matrix(V, obj, boundaries, ext_bnd_id);
//
// std::cout <<"Analytical value: "<< cap << '\n';
