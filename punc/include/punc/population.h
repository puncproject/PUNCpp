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
#include <fstream>
#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/systems/si/codata/electron_constants.hpp>
#include <boost/units/systems/si/codata/physico-chemical_constants.hpp>
#include <boost/units/systems/si/codata/universal_constants.hpp>

namespace punc
{

namespace df = dolfin;

struct PhysicalConstants
{
    double e = boost::units::si::constants::codata::e / boost::units::si::coulomb;
    double m_e = boost::units::si::constants::codata::m_e / boost::units::si::kilograms;
    double ratio = boost::units::si::constants::codata::m_e_over_m_p / boost::units::si::dimensionless();
    double m_i = m_e / ratio;

    double k_B = boost::units::si::constants::codata::k_B * boost::units::si::kelvin / boost::units::si::joules;
    double eps0 = boost::units::si::constants::codata::epsilon_0 * boost::units::si::meter / boost::units::si::farad;
};

signed long int locate(std::shared_ptr<const df::Mesh> mesh, const std::vector<double> &x);

class Pdf : public df::Expression
{
private:
    double vth_;
    std::vector<double> vd_;
    
public:
  bool has_cdf;
  virtual double operator()(const std::vector<double> &x) = 0;
  virtual double operator()(const std::vector<double> &x, const std::vector<double> &n)
  {
      double vn = 0.0;
      for (int i = 0; i < dim(); ++i)
      {
          vn += x[i] * n[i];
      }
      return (vn > 0.0) * vn * this->operator()(x); 
    };
    virtual int dim() = 0;
    virtual double max() = 0;
    virtual std::vector<double> domain() = 0;
    virtual double vth(){return vth_;}
    virtual std::vector<double> vd(){return vd_;}
    virtual void set_vth(double v) {vth_= v;}
    virtual void set_vd(std::vector<double> &v) { vd_ = v; }
    virtual double flux(const std::vector<double> &n) { return 0.0;}   
    virtual double flux_num(const std::vector<double> &n, double S) { return 0.0; } 
    virtual void set_flux_normal(std::vector<double> &n) {}
    virtual std::vector<double> cdf(const std::size_t N) {return {};}
};

/**
 * @brief A simulation particle
 */
struct Particle
{
    std::vector<double> x;  ///< Position
    std::vector<double> v;  ///< Velocity
    double q;               ///< Charge
    double m;               ///< Mass
};

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
    Pdf &pdf;  ///< Position distribution function (initially)
    Pdf &vdf;  ///< Velocity distribution function (initially and at boundary)

    Species(double q, double m, double n, int num, Pdf &pdf, Pdf &vdf) :
            q(q), m(m), n(n), num(num), pdf(pdf), vdf(vdf) {}
};

class CreateSpecies
{
  public:
    double X;
    int g_dim;
    double volume, num_cells;
    std::vector<Species> species;
    double T = std::numeric_limits<double>::quiet_NaN();
    double Q = boost::units::si::constants::codata::e / boost::units::si::coulomb;
    double M = std::numeric_limits<double>::quiet_NaN();
    double epsilon_0 = boost::units::si::constants::codata::epsilon_0*boost::units::si::meter/boost::units::si::farad;

    CreateSpecies(std::shared_ptr<const df::Mesh> &mesh, double X = 1.0);

    void create_raw(double q, double m, double n, Pdf &pdf, Pdf &vdf, int npc = 4,
                    int num = 0);

    void create(double q, double m, double n, Pdf &pdf, Pdf &vdf, int npc = 4,
                int num = 0);
};

class Cell : public df::Cell
{
  public:
    std::size_t id;
    std::vector<std::size_t> neighbors;
    std::vector<signed long int> facet_adjacents;
    std::vector<double> facet_normals;
    std::vector<double> facet_mids;
    std::vector<Particle> particles;
    std::vector<double> vertex_coordinates;
    std::vector<double> basis_matrix;
    ufc::cell ufc_cell;

    Cell() {};

    Cell(std::shared_ptr<const df::Mesh> &mesh,
        std::size_t id, std::vector<std::size_t> neighbors)
        : df::Cell(*mesh, id), id(id), neighbors(neighbors)
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
class Population
{
  public:
    std::shared_ptr<const df::Mesh> mesh;   ///< df::Mesh of the domain
    std::vector<Cell> cells;                ///< All df::Cells in the domain
    std::size_t num_cells;                  ///< Number of cells in the domain
    std::size_t g_dim;                       ///< Number of geometric dimensions
    std::size_t t_dim;                       ///< Number of topological dimensions

    Population(std::shared_ptr<const df::Mesh> &mesh,
               const df::MeshFunction<std::size_t> &bnd);
    void init_localizer(const df::MeshFunction<std::size_t> &bnd);
    void add_particles(std::vector<double> &xs, std::vector<double> &vs,
                       double q, double m);
    signed long int locate(std::vector<double> &p);
    signed long int relocate(std::vector<double> &p, signed long int cell_id);
    void update(boost::optional<std::vector<ObjectBC>& > objects = boost::none);
    std::size_t num_of_particles();         ///< Returns number of particles
    std::size_t num_of_positives();         ///< Returns number of positively charged particles
    std::size_t num_of_negatives();         ///< Returns number of negatively charged particles
    void save_file(const std::string &fname);
    void load_file(const std::string &fname);
    void save_vel(const std::string &fname);
};

}

#endif // POPULATION_H
