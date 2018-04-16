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

const df::Mesh load_mesh(std::string fname)
{
    df::Mesh mesh(fname + ".xml");
    return mesh;
}

df::MeshFunction<std::size_t> load_boundaries(const df::Mesh mesh, std::string fname)
{
    df::MeshFunction<std::size_t> boundaries(std::make_shared<const df::Mesh>(mesh), fname + "_facet_region.xml");

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

    std::string fname{"/home/diako/Documents/cpp/punc_experimental/prototypes/mesh/square"};
    // std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/injection/mesh/box"};

    auto mesh = load_mesh(fname);
    auto boundaries = load_boundaries(mesh, fname);
    auto gdim = mesh.geometry().dim();
    auto tdim = mesh.topology().dim();
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
    return 0;
}
