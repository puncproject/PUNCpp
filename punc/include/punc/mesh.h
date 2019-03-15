
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
 * @file		mesh.h
 * @brief		Mesh handling
 */

#ifndef MESH_H
#define MESH_H

#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Mesh.h>
#include <string>

namespace punc {

namespace df = dolfin;

/**
 * @brief Exterior facet
 */
struct ExteriorFacet
{
    double area;                  ///< Area of the facet
    std::vector<double> vertices; ///< Vertices of the facet
    std::vector<double> normal;   ///< Normal vector of the facet
    std::vector<double> basis;    ///< Basis matrix for transforming from physical space to a space defined by the normal vector of the facet
};

/**
 * @brief The PUNC simulation mesh
 */
class Mesh {
public:
    using string = std::string;
    using size_t = std::size_t;

    std::shared_ptr<const df::Mesh> mesh; ///< Dolfin mesh
    size_t dim;                           ///< Number of geometric dimensions
    size_t ext_bnd_id;                    ///< Id of the exterior boundary
    size_t num_objects;                   ///< Number of objects in the domain
    std::vector<ExteriorFacet> exterior_facets; ///< Vector containing all the exterior facets
    
    /**
     * @brief Boundary markers
     *
     * Facets are marked as follows:
     * - 0 - all interior facets which are not part of a boundary
     * - 1 - the exterior boundary
     * - 2 - the first object
     * - 3 - the second object
     *
     * and so forth.
     */
    df::MeshFunction<size_t> bnd;

    /**
     * @brief Reads mesh from filename and initialize Mesh
     * @param   fname   filename
     *
     * Reads .xml or .h5 files depending on the extension of fname. If no
     * extension is present .xml will be assumed.
     */
    Mesh(const string &fname);

    //! Returns the size of the smallest box enclosing the domain.
    std::vector<double> domain_size() const;

    //! The volume of the domain.
    double volume() const;

private:
    //! Load file into Mesh. Used by Mesh().
    void load_file(string fname);

    //! Create a vector containing all the exterior boundary facets
    void exterior_boundaries();

    //! Return ids in bnd. Used by Mesh().
    std::vector<size_t> get_bnd_ids() const;
};

/**
 * @brief Relabel mesh functions (in-place)
 * @param[in,out]   f       Mesh function
 * @param           from    id to replace
 * @param           to      replacement id
 */
inline void relabel_mesh_function(df::MeshFunction<size_t> &f,
                                  std::size_t from, std::size_t to){

    auto values = f.values();
    for (std::size_t i = 0; i < f.size(); i++) {
    	if(values[i] == from) {
    		f.set_value(i, to);
    	}
    }
}


} // namespace punc

#endif // MESH_H
