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

namespace punc
{

Mesh::Mesh(std::string fname){
    
    // Splits fname in name plus extension. I don't like adding big libraries
    // for small tasks, but Boost is to my knowledge the only robust way to
    // deal with file paths in C++.
    boost::filesystem::path path(fname);
    boost::filesystem::path stem(path.parent_path());
    stem /= path.stem();
    fname = stem.string();

    std::string fext = path.extension().string();
    if(fext=="") fext = ".xml";

    if(fext==".xml"){

        mesh = std::make_shared<const df::Mesh>(fname+fext);
        bnd  = df::MeshFunction<std::size_t>(mesh, fname+"_facet_region"+fext);

    } else if(fext==".h5"){

        auto com = MPI_COMM_WORLD;
        df::HDF5File hdf(com, fname+fext, "r");

        df::Mesh temp(com);
        hdf.read(temp, "/mesh", false);
        mesh = std::make_shared<const df::Mesh>(temp);

        bnd = df::MeshFunction<std::size_t>(mesh);
        hdf.read(bnd, "/boundaries");

    } else {
        df::error("Only .xml or .h5 meshes supported");
    }
}

}
