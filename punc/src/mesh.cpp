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

/**
 * @file		mesh.cpp
 * @brief		Mesh handling
 */
#include "../include/punc/mesh.h"

#include <dolfin/io/HDF5File.h>
#include <dolfin/function/Constant.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/mesh/SubsetIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>

#include <string>
#include <boost/filesystem.hpp>

#include "../ufl/Volume.h"

namespace punc
{


Mesh::Mesh(const string &fname){
    
    load_file(fname);
    dim = mesh->geometry().dim();
    mesh->init(0, dim);
    mesh->init(dim-1, dim);

    auto tags = get_bnd_ids();

    // Subtract one for exterior boundary and one for interior facets
    num_objects = tags.size()-2;
    ext_bnd_id = 1;
    
    for(size_t i=0; i<tags.size(); i++){
        relabel_mesh_function(bnd, tags[i], i);
    }

    // Create exterior boundary facets 
    exterior_boundaries();
}

void Mesh::load_file(string fname){

    // Splits fname in (fname + fext)
    boost::filesystem::path path(fname);
    boost::filesystem::path stem(path.parent_path());
    stem /= path.stem();
    fname = stem.string();
    string fext = path.extension().string();
    if(fext=="") fext = ".xml";

    if(fext==".xml"){

        mesh = std::make_shared<const df::Mesh>(fname+fext);
        bnd  = df::MeshFunction<size_t>(mesh, fname+"_facet_region"+fext);

    } else if(fext==".h5"){

        auto com = MPI_COMM_WORLD;
        df::HDF5File hdf(com, fname+fext, "r");

        df::Mesh temp(com);
        hdf.read(temp, "/mesh", false);
        mesh = std::make_shared<const df::Mesh>(temp);

        bnd = df::MeshFunction<size_t>(mesh);
        hdf.read(bnd, "/boundaries");

    } else {
        df::error("Only .xml or .h5 meshes supported");
    }
}

std::vector<size_t> Mesh::get_bnd_ids() const
{
    auto comm = mesh->mpi_comm();
    auto values = bnd.values();
    auto length = bnd.size();
    std::vector<std::size_t> tags(length);

    for (std::size_t i = 0; i < length; ++i) {
        tags[i] = values[i];
    }

    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());

    if(df::MPI::size(comm)==1) {
        return tags;
    } else {
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

std::vector<double> Mesh::domain_size() const
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

double Mesh::volume() const
{
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

void Mesh::exterior_boundaries()
{
    auto t_dim = mesh->topology().dim();
    auto values = bnd.values();
    auto length = bnd.size();
    int num_facets = 0;
    for (std::size_t i = 0; i < length; ++i)
    {
        if (ext_bnd_id == values[i])
        {
            num_facets += 1;
        }
    }

    double area = 0.0;
    std::vector<double> normal(dim);
    std::vector<double> vertices(dim * dim);
    std::vector<double> basis(dim * dim);
    std::vector<double> vertex(dim);
    double norm;
    int j;
    df::SubsetIterator facet_iter(bnd, ext_bnd_id);
    for (; !facet_iter.end(); ++facet_iter)
    {
        df::Cell cell(*mesh, facet_iter->entities(t_dim)[0]);
        auto cell_facet = cell.entities(t_dim - 1);
        std::size_t num_facets = cell.num_entities(t_dim - 1);
        for (std::size_t i = 0; i < num_facets; ++i)
        {
            if (cell_facet[i] == facet_iter->index())
            {
                area = cell.facet_area(i);
                for (std::size_t j = 0; j < dim; ++j)
                {
                    normal[j] = -1 * cell.normal(i, j);
                    basis[j * dim] = normal[j];
                }
            }
        }
        assert(area != 0.0 && "The facet area cannot be zero!");

        j = 0;
        for (df::VertexIterator v(*facet_iter); !v.end(); ++v)
        {
            for (std::size_t i = 0; i < dim; ++i)
            {
                vertices[j * dim + i] = v->x(i);
            }
            j += 1;
        }
        norm = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
        {
            vertex[i] = vertices[i] - vertices[dim + i];
            norm += vertex[i] * vertex[i];
        }
        for (std::size_t i = 0; i < dim; ++i)
        {
            vertex[i] /= sqrt(norm);
            basis[i * dim + 1] = vertex[i];
        }

        if (dim == 3)
        {
            basis[2] = normal[1] * vertex[2] - normal[2] * vertex[1];
            basis[5] = normal[2] * vertex[0] - normal[0] * vertex[2];
            basis[8] = normal[0] * vertex[1] - normal[1] * vertex[0];
        }
        exterior_facets.push_back(ExteriorFacet{area, vertices, normal, basis});
    }
}

} // namespace punc
