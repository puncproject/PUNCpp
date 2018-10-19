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

#include "../include/punc/mesh.h"

#include <boost/filesystem.hpp>
#include <string>

#include "../ufl/Surface.h"
#include "../ufl/Volume.h"

namespace punc
{

Mesh::Mesh(const string &fname){
    
    load_file(fname);
    dim = mesh->geometry().dim();
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

std::vector<size_t> Mesh::get_mesh_ids() const
{
    auto comm = mesh->mpi_comm();
    auto values = bnd.values();
    auto length = bnd.size();
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

double Mesh::surface_area() const
{
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

} // namespace punc
