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

#ifndef POPULATION_H
#define POPULATION_H

#include "poisson.h"
#include "distributions.h"
#include <fstream>
#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/systems/si/codata/electron_constants.hpp>
#include <boost/units/systems/si/codata/physico-chemical_constants.hpp>
#include <boost/units/systems/si/codata/universal_constants.hpp>

namespace punc
{

namespace df = dolfin;

/**
 * @brief Generic matrix-vector product
 * @param y[in]   the vector
 * @param A[in]   the matrix
 * @param x[in]   the vector resulting from the matrix-vector product
 * @param n       number of rows of matrix A
 * @param m       number of columns of matrix A 
 */
static inline void matrix_vector_product(double *y, const double *A,
                                         const double *x, std::size_t n,
                                         std::size_t m)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        y[i] = A[i * m];
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < m - 1; ++j)
        {
            y[i] += A[i * m + j + 1] * x[j];
        }
    }
}

/**
 * @brief Contains the most important physical constants needed in PIC simulations
 */
struct PhysicalConstants
{
    double e = boost::units::si::constants::codata::e / boost::units::si::coulomb;                                    ///< Elementary charge
    double m_e = boost::units::si::constants::codata::m_e / boost::units::si::kilograms;                              ///< Electron mass
    double ratio = boost::units::si::constants::codata::m_e_over_m_p / boost::units::si::dimensionless();             ///< Electron to proton mass ratio
    double m_i = m_e / ratio;                                                                                         ///< Proton mass

    double k_B = boost::units::si::constants::codata::k_B * boost::units::si::kelvin / boost::units::si::joules;      ///< Boltzmann constant
    double eps0 = boost::units::si::constants::codata::epsilon_0 * boost::units::si::meter / boost::units::si::farad; ///< Electric constant
};

/**
 * @brief A simulation particle
 */
template <std::size_t len = 2>
struct Particle
{
    double x[len];  ///< Position
    double v[len];  ///< Velocity
    double q;       ///< Charge
    double m;       ///< Mass
    Particle(const double *x, const double *v, double q, double m);
    Particle(){};
};

template <std::size_t len>
Particle<len>::Particle(const double *x, const double *v,
                      double q, double m) : q(q), m(m)
{
    for (std::size_t i = 0; i < len; i++)
    {
        this->x[i] = x[i];
        this->v[i] = v[i];
    }
}

/**
 * @brief Complete specification of a species.
 */
class Species 
{
public:
    double q; ///< Charge
    double m; ///< Mass
    double n; ///< Density
    int num;  ///< Initial number of particles
    Pdf &pdf; ///< Position distribution function (initially)
    Pdf &vdf; ///< Velocity distribution function (initially and at boundary)

    Species(double q, double m, double n, int num, Pdf &pdf, Pdf &vdf) :
            q(q), m(m), n(n), num(num), pdf(pdf), vdf(vdf) {}
};

/**
 * @brief Creates plasma species.
 */
class CreateSpecies
{
  public:
    double X; ///< Characteristic length - used for normalization
    int g_dim; ///< The geometrical dimension of the physical space
    double volume, num_cells; ///< The volume of the simulation domain and number of cells in the domain
    std::vector<Species> species; ///< A vector containing species
    double T = std::numeric_limits<double>::quiet_NaN(); ///< Characteristic time
    double Q = boost::units::si::constants::codata::e / boost::units::si::coulomb; ///< Characteristic charge - here set to elementary charge
    double M = std::numeric_limits<double>::quiet_NaN();                           ///< Characteristic mass
    double epsilon_0 = boost::units::si::constants::codata::epsilon_0 * boost::units::si::meter / boost::units::si::farad; ///< Electric constant

    /**
     * @brief Constructor
     * @param[in]  df::Mesh  
     * @param   X - characteristic length 
     */
    CreateSpecies(std::shared_ptr<const df::Mesh> &mesh, double X = 1.0);

    /**
     * @brief Creates species without normalization
     * @param[in]   q - species charge  
     * @param[in]   m - species mass 
     * @param[in]   n - species volumetric number density
     * @param[in]   pdf - position distribution function for the species
     * @param[in]   vdf - velocity distribution function for the species
     * @param[in]   npc - number of particles per cell for the species
     * @param[in]   num - total number of particles in the simulation domain for the species
     */
    void create_raw(double q, double m, double n, Pdf &pdf, Pdf &vdf, int npc = 4,
                    int num = 0);

    /**
     * @brief Creates species and normalizes the physical quantities
     * @param[in]   q - species charge  
     * @param[in]   m - species mass 
     * @param[in]   n - species volumetric number density
     * @param[in]   pdf - position distribution function for the species
     * @param[in]   vdf - velocity distribution function for the species
     * @param[in]   npc - number of particles per cell for the species
     * @param[in]   num - total number of particles in the simulation domain for the species
     */
    void create(double q, double m, double n, Pdf &pdf, Pdf &vdf, int npc = 4,
                int num = 0);
};

/**
 * @brief Generic class representing a cell in the simulation domain
 */
template <std::size_t len>
class Cell : public df::Cell
{
  public:
    std::size_t id; ///< Cell index or id
    std::size_t g_dim; ///< geometric dimension of the domain
    std::vector<std::size_t> neighbors; ///< Neighbors of the Cell - all the cells that share a vertex, facet or edge with the cell
    std::vector<signed long int> facet_adjacents; ///< Adjacent facets to the Cell
    std::vector<double> facet_plane_coeffs;       ///< Coefficients of the plane-equation for the facets of the Cell
    std::vector<Particle<len>> particles;         ///< Particles contained in the Cell
    std::vector<double> vertex_coordinates;       ///< Vertices of the Cell
    std::vector<double> basis_matrix;             ///< Basis matrix used for transformation between physical and barycentric coordinates
    ufc::cell ufc_cell;                           ///< The underlying UFC cell

    /**
     * @brief Constructor
     * @param[in]   df::Mesh  
     * @param[in]   Cell index/id
     * @param[in]   neighbors - a vector containing the indices of all the cells in the domain  
     */
    Cell(std::shared_ptr<const df::Mesh> &mesh,
         std::size_t id, std::vector<std::size_t> neighbors)
        : df::Cell(*mesh, id), id(id), 
        neighbors(neighbors)
    {
        auto g_dim = mesh->geometry().dim();
        const std::size_t num_vertices = (*this).num_vertices();
        const unsigned int *vertices = (*this).entities(0);

        vertex_coordinates.resize(num_vertices*g_dim);
        basis_matrix.resize(num_vertices*num_vertices);
        for (std::size_t i = 0; i < num_vertices; i++)
        {
            for (std::size_t j = 0; j < g_dim; j++)
            {
                vertex_coordinates[i * g_dim + j] = mesh->geometry().x(vertices[i])[j];
            }
        }

        (*this).get_cell_data(ufc_cell);

        if(g_dim == 1)
        {
            basis_matrix_1d();
        }else if(g_dim == 2){
            basis_matrix_2d();
        }else if(g_dim == 3){
            basis_matrix_3d();
        }
    }

    /**
     * @brief Generates the basis matrix (in 1D) used to transform physical 
     * coordinates of a given particle position to the corresponding barycentric 
     * coordinates of the cell 
     */
    void basis_matrix_1d()
    {
        double x1 = vertex_coordinates[0];
        double x2 = vertex_coordinates[1];

        double det = x2 - x1;

        basis_matrix[0] = x2 / det;
        basis_matrix[1] = -1.0 / det;
        basis_matrix[2] = -x1 / det;
        basis_matrix[3] = 1.0 / det;
    }

    /**
     * @brief Generates the basis matrix (in 2D) used to transform physical 
     * coordinates of a given particle position to the corresponding barycentric 
     * coordinates of the cell 
     */
    void basis_matrix_2d()
    {
        double x1 = vertex_coordinates[0];
        double y1 = vertex_coordinates[1];
        double x2 = vertex_coordinates[2];
        double y2 = vertex_coordinates[3];
        double x3 = vertex_coordinates[4];
        double y3 = vertex_coordinates[5];

        double det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);

        basis_matrix[0] = (x2 * y3 - x3 * y2) / det;
        basis_matrix[1] = (y2 - y3) / det;
        basis_matrix[2] = (x3 - x2) / det;
        basis_matrix[3] = (x3 * y1 - x1 * y3) / det;
        basis_matrix[4] = (y3 - y1) / det;
        basis_matrix[5] = (x1 - x3) / det;
        basis_matrix[6] = (x1 * y2 - x2 * y1) / det;
        basis_matrix[7] = (y1 - y2) / det;
        basis_matrix[8] = (x2 - x1) / det;
    }

    /**
     * @brief Generates the basis matrix (in 3D) used to transform physical 
     * coordinates of a given particle position to the corresponding barycentric 
     * coordinates of the cell 
     */
    void basis_matrix_3d()
    {
        double x1 = vertex_coordinates[0];
        double y1 = vertex_coordinates[1];
        double z1 = vertex_coordinates[2];
        double x2 = vertex_coordinates[3];
        double y2 = vertex_coordinates[4];
        double z2 = vertex_coordinates[5];
        double x3 = vertex_coordinates[6];
        double y3 = vertex_coordinates[7];
        double z3 = vertex_coordinates[8];
        double x4 = vertex_coordinates[9];
        double y4 = vertex_coordinates[10];
        double z4 = vertex_coordinates[11];

        double x12 = x1 - x2;
        double y12 = y1 - y2;
        double z12 = z1 - z2;
        double x13 = x1 - x3;
        double y13 = y1 - y3;
        double z13 = z1 - z3;
        double x14 = x1 - x4;
        double y14 = y1 - y4;
        double z14 = z1 - z4;
        double x21 = x2 - x1;
        double y21 = y2 - y1;
        double z21 = z2 - z1;

        double x24 = x2 - x4;
        double y24 = y2 - y4;
        double z24 = z2 - z4;
        double x31 = x3 - x1;
        double y31 = y3 - y1;
        double z31 = z3 - z1;
        double x32 = x3 - x2;
        double y32 = y3 - y2;
        double z32 = z3 - z2;
        double x34 = x3 - x4;
        double y34 = y3 - y4;
        double z34 = z3 - z4;

        double x42 = x4 - x2;
        double y42 = y4 - y2;
        double z42 = z4 - z2;
        double x43 = x4 - x3;
        double y43 = y4 - y3;
        double z43 = z4 - z3;

        double V01 = (x2 * (y3 * z4 - y4 * z3) + x3 * (y4 * z2 - y2 * z4) + x4 * (y2 * z3 - y3 * z2)) / 6.0;
        double V02 = (x1 * (y4 * z3 - y3 * z4) + x3 * (y1 * z4 - y4 * z1) + x4 * (y3 * z1 - y1 * z3)) / 6.0;
        double V03 = (x1 * (y2 * z4 - y4 * z2) + x2 * (y4 * z1 - y1 * z4) + x4 * (y1 * z2 - y2 * z1)) / 6.0;
        double V04 = (x1 * (y3 * z2 - y2 * z3) + x2 * (y1 * z3 - y3 * z1) + x3 * (y2 * z1 - y1 * z2)) / 6.0;
        double V = V01 + V02 + V03 + V04;
        double V6 = 6 * V;

        basis_matrix[0] = V01 / V;
        basis_matrix[4] = V02 / V;
        basis_matrix[8] = V03 / V;
        basis_matrix[12] = V04 / V;

        basis_matrix[1] = (y42 * z32 - y32 * z42) / V6;  
        basis_matrix[5] = (y31 * z43 - y34 * z13) / V6; 
        basis_matrix[9] = (y24 * z14 - y14 * z24) / V6;  
        basis_matrix[13] = (y13 * z21 - y12 * z31) / V6; 

        basis_matrix[2] = (x32 * z42 - x42 * z32) / V6;  
        basis_matrix[6] = (x43 * z31 - x13 * z34) / V6;  
        basis_matrix[10] = (x14 * z24 - x24 * z14) / V6; 
        basis_matrix[14] = (x21 * z13 - x31 * z12) / V6; 

        basis_matrix[3] = (x42 * y32 - x32 * y42) / V6;  
        basis_matrix[7] = (x31 * y43 - x34 * y13) / V6;  
        basis_matrix[11] = (x24 * y14 - x14 * y24) / V6; 
        basis_matrix[15] = (x13 * y21 - x12 * y31) / V6; 
    }
};

/**
 * @brief A collection of Particles
 */
template <std::size_t len>
class Population
{
  public:
    std::shared_ptr<const df::Mesh> mesh;   ///< df::Mesh of the domain
    const std::size_t g_dim;                ///< Number of geometric dimensions
    std::size_t t_dim;                      ///< Number of topological dimensions
    std::size_t num_cells;                  ///< Number of cells in the domain
    std::vector<Cell<len>> cells;           ///< All df::Cells in the domain

    Population(std::shared_ptr<const df::Mesh> &mesh,
               const df::MeshFunction<std::size_t> &bnd);
    void init_localizer(const df::MeshFunction<std::size_t> &bnd);
    void add_particles(const std::vector<double> &xs,
                       const std::vector<double> &vs,
                       double q, double m);
    void add_particles(const std::vector<Particle<len>> &ps);
    signed long int locate(const double *p);
    signed long int relocate(const double *p, signed long int cell_id);
    signed long int relocate_fast(const double *p, signed long int cell_id);
    void update(std::vector<ObjectBC>& objects);
    std::size_t num_of_particles();         ///< Returns number of particles
    std::size_t num_of_positives();         ///< Returns number of positively charged particles
    std::size_t num_of_negatives();         ///< Returns number of negatively charged particles

    /**
     * @brief Save particles to file
     * @param   fname   File name
     * @param   binary  Use binary file format
     * @see load_file
     * 
     * Saves particles to file using either binary or ASCII format.
     *
     * Binary files are typically 40% the size of ASCII files, and do not
     * suffer from the loss of precision associated with displaying numbers in
     * base 10. ASCII files display numbers in base 10, but this precision
     * lost should rarely be significant. Binary files merely stores the
     * Particle structs byte-by-bate, and this makes it depend on the platform.
     * Different platform may have different size for the datatypes in Particle,
     * different padding of structs, and different endianness. Reading a binary
     * file on a system where this differs from where the binary file was made
     * will fail. As such, ASCII files are more portable.
     */
    void save_file(const std::string &fname, bool binary=false);

    /**
     * @brief Load particles from file
     * @param   fname   File name
     * @param   binary  Use binary file format
     * @see save_file
     * 
     * Loads particles from binary or ASCII file.
     */
    void load_file(const std::string &fname, bool binary=false);
};

template <std::size_t len>
Population<len>::Population(std::shared_ptr<const df::Mesh> &mesh,
                            const df::MeshFunction<std::size_t> &bnd)
    : mesh(mesh), g_dim(mesh->geometry().dim()), t_dim(mesh->topology().dim()),
      num_cells(mesh->num_cells())
{
    mesh->init(0, t_dim);
    for (df::MeshEntityIterator e(*(mesh), t_dim); !e.end(); ++e)
    {
        std::vector<std::size_t> neighbors;
        auto cell_id = e->index();
        auto num_vertices = e->num_entities(0);
        for (std::size_t i = 0; i < num_vertices; ++i)
        {
            df::Vertex vertex(*mesh, e->entities(0)[i]);
            auto vertex_cells = vertex.entities(t_dim);
            auto num_adj_cells = vertex.num_entities(t_dim);
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

        Cell<len> cell(mesh, cell_id, neighbors);
        cells.emplace_back(cell);
    }

    init_localizer(bnd);
}

template <std::size_t len>
void Population<len>::init_localizer(const df::MeshFunction<std::size_t> &bnd)
{
    mesh->init(t_dim - 1, t_dim);
    for (auto &cell : cells)
    {
        std::vector<signed long int> facet_adjacents;
        std::vector<double> facet_plane_coeffs;

        auto cell_id = cell.id;
        auto facets = cell.entities(t_dim - 1);
        auto num_facets = cell.num_entities(t_dim - 1);

        for (std::size_t i = 0; i < num_facets; ++i)
        {
            df::Facet facet(*mesh, cell.entities(t_dim - 1)[i]);
            auto facet_cells = facet.entities(t_dim);
            auto num_adj_cells = facet.num_entities(t_dim);

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

            double dot_product = 0;
            for(std::size_t j = 0; j < g_dim; j++){
                dot_product += facet.midpoint()[j]*cell.normal(i)[j];
            }
            facet_plane_coeffs.push_back(-dot_product);
            for(std::size_t j = 0; j < g_dim; j++){
                facet_plane_coeffs.push_back(cell.normal(i)[j]);
            }

        }

        cells[cell_id].facet_adjacents = facet_adjacents;
        cells[cell_id].facet_plane_coeffs = facet_plane_coeffs;
    }
}

template <std::size_t len>
void Population<len>::add_particles(const std::vector<double> &xs,
                                    const std::vector<double> &vs,
                                    double q, double m)
{
    std::size_t num_particles = xs.size() / g_dim;
    double xs_tmp[g_dim];
    double vs_tmp[g_dim];

    std::size_t cell_id;
    for (std::size_t i = 0; i < num_particles; ++i)
    {
        for (std::size_t j = 0; j < g_dim; ++j)
        {
            xs_tmp[j] = xs[i * g_dim + j];
            vs_tmp[j] = vs[i * g_dim + j];
        }
        cell_id = locate(xs_tmp);
        if (cell_id >= 0)
        {
            Particle<len> _particles(xs_tmp, vs_tmp, q, m);
            cells[cell_id].particles.push_back(_particles);
        }
    }
}

template <std::size_t len>
void Population<len>::add_particles(const std::vector<Particle<len>> &ps)
{
    for (auto &p : ps){
        std::size_t cell_id = locate(p.x);
        if(cell_id >=0){
            cells[cell_id].particles.push_back(p);
        }
    }
}

template <std::size_t len>
signed long int Population<len>::locate(const double *p)
{
    return punc::locate(mesh, p);
}

template <std::size_t len>
signed long int Population<len>::relocate(const double *p, signed long int cell_id)
{
    // One element for each facet.
    // For 1D and 2D all aren't used, but slightly faster than vector.
    double proj[4];
    double *coeffs = cells[cell_id].facet_plane_coeffs.data();

    for (std::size_t i = 0; i < g_dim + 1; ++i)
    {
        proj[i] = *coeffs++;
        for (std::size_t j=0; j < g_dim; j++){
            proj[i] += *coeffs++ * p[j];
        }
    }

    double proj_max = proj[0];
    std::size_t proj_argmax = 0;
    for(std::size_t i = 1; i < g_dim + 1; i++){
        if(proj[i] > proj_max){
            proj_max = proj[i];
            proj_argmax = i;
        }
    }

    if(proj_max < 0){
        return cell_id;
    } else {
        auto new_cell_id = cells[cell_id].facet_adjacents[proj_argmax];

        // negative new_cell_id indicate that the particle hit a boundary with
        // id (-new_cell_id).
        if(new_cell_id >= 0){
            return relocate(p, new_cell_id);
        } else {
            return new_cell_id;
        }
    }
}

template <std::size_t len>
signed long int Population<len>::relocate_fast(const double *p, signed long int cell_id)
{
    // One element for each facet.
    // For 1D and 2D all aren't used, but slightly faster than vector.
    double proj;
    double *coeffs = cells[cell_id].facet_plane_coeffs.data();

    for (std::size_t i = 0; i < g_dim + 1; ++i)
    {
        proj = *coeffs++;
        for (std::size_t j=0; j < g_dim; j++){
            proj += *coeffs++ * p[j];
        }

        if(proj > 0){
            auto new_cell_id = cells[cell_id].facet_adjacents[i];

            // negative new_cell_id indicate that the particle hit a boundary with
            // id (-new_cell_id).
            if(new_cell_id >= 0){
                return relocate_fast(p, new_cell_id);
            } else {
                return new_cell_id;
            }
        }
    }

    return cell_id;

}

template <std::size_t len>
void Population<len>::update(std::vector<ObjectBC> &objects)
{
    // FIXME: Consider a different mechanism for boundaries than using negative
    // numbers, or at least circumvent the problem of casting num_cells to
    // signed. Not good practice. size_t may overflow to negative numbers upon
    // truncation for large numbers.
    signed long int new_cell_id;
    for (signed long int cell_id = 0; cell_id < (signed long int)num_cells; ++cell_id)
    {
        std::vector<std::size_t> to_delete;
        std::size_t num_particles = cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = cells[cell_id].particles[p_id];
            new_cell_id = relocate_fast(particle.x, cell_id);
            if (new_cell_id != cell_id)
            {
                to_delete.push_back(p_id);
                if (new_cell_id >= 0)
                {
                    // Particle will actually be checked again if
                    // new_cell_id>cell_id. Probably not worth avoiding.
                    cells[new_cell_id].particles.push_back(particle);
                }
                else
                {
                    // Standard numbering scheme on objects and exterior
                    // boundary would eliminate this loop.
                    for(auto &object : objects){
                        if ((std::size_t)(-new_cell_id) == object.id)
                        {
                            object.charge += particle.q;
                        }
                    }
                }
            }
        }
        std::size_t size_to_delete = to_delete.size();
        for (std::size_t it = size_to_delete; it-- > 0;)
        {
            auto p_id = to_delete[it];
            cells[cell_id].particles[p_id] = cells[cell_id].particles.back();
            cells[cell_id].particles.pop_back();
        }

    }
}

template <std::size_t len>
std::size_t Population<len>::num_of_particles()
{
    std::size_t num_particles = 0;
    for (auto &cell : cells)
    {
        num_particles += cell.particles.size();
    }
    return num_particles;
}

template <std::size_t len>
std::size_t Population<len>::num_of_positives()
{
    std::size_t num_positives = 0;
    for (auto &cell : cells)
    {
        for (auto &particle : cell.particles)
        {
            if (particle.q > 0)
            {
                num_positives++;
            }
        }
    }
    return num_positives;
}

template <std::size_t len>
std::size_t Population<len>::num_of_negatives()
{
    std::size_t num_negatives = 0;
    for (auto &cell : cells)
    {
        for (auto &particle : cell.particles)
        {
            if (particle.q < 0)
            {
                num_negatives++;
            }
        }
    }
    return num_negatives;
}

template <std::size_t len>
void Population<len>::save_file(const std::string &fname, bool binary)
{
    if(binary){

        FILE *fout = fopen(fname.c_str(), "wb");

        for (auto &cell : cells)
            for (auto &particle : cell.particles)
                fwrite(&particle, sizeof(particle), 1, fout);

        fclose(fout);

    } else {

        FILE *fout = fopen(fname.c_str(), "w");

        for (auto &cell : cells) {
            for (auto &particle : cell.particles) {

                for (std::size_t i = 0; i < g_dim; ++i)
                    fprintf(fout, "%.17g\t", particle.x[i]);
    
                for (std::size_t i = 0; i < g_dim; ++i)
                    fprintf(fout, "%.17g\t", particle.v[i]);
   
                fprintf(fout, "%.17g\t %.17g\t", particle.q, particle.m);
                fprintf(fout, "\n");
            }
        }
        fclose(fout);
    }
}

template <std::size_t len>
void Population<len>::load_file(const std::string &fname, bool binary)
{
    if(binary){

        FILE *fin = fopen(fname.c_str(), "rb");

        std::vector<Particle<len>> ps;
        Particle<len> p;

        while(fread(&p, sizeof(p), 1, fin))
            ps.push_back(p); 

        fclose(fin);
        add_particles(ps);

    } else {

        std::fstream in(fname);
        std::string line;
        std::vector<double> x(g_dim);
        std::vector<double> v(g_dim);
        double q = 0;
        double m = 0;
        std::size_t i;
        while (std::getline(in, line))
        {
            double value;
            std::stringstream ss(line);
            i = 0;
            while (ss >> value)
            {
                if (i < g_dim) x[i] = value;
                else if (i >= g_dim && i < 2 * g_dim) v[i % g_dim] = value;
                else if (i == 2 * g_dim) q = value;
                else if (i == 2 * g_dim + 1) m = value;
                ++i;
            }
            add_particles(x, v, q, m);
        }
    }
}

} // namespace punc

#endif // POPULATION_H
