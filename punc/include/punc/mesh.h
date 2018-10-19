
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

#ifndef MESH_H
#define MESH_H

#include <dolfin.h>
#include <string>

namespace punc {

namespace df = dolfin;

class Mesh {
public:
    using string = std::string;
    using size_t = std::size_t;

    Mesh(const string &fname);
    std::shared_ptr<const df::Mesh> mesh;
    df::MeshFunction<size_t> bnd;
    size_t dim;

private:
    void load_file(string fname);
};

} // namespace punc

#endif // MESH_H
