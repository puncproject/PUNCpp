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
    double _vth;
    std::vector<double> _vd;
    
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
    virtual double vth(){return _vth;}
    virtual std::vector<double> vd(){return _vd;}
    virtual void set_vth(double v) {_vth= v;}
    virtual void set_vd(std::vector<double> &v) { _vd = v; }
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
    int D;
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

class Cell
{
  public:
    std::size_t id;
    std::vector<std::size_t> neighbors;
    std::vector<signed long int> facet_adjacents;
    std::vector<double> facet_normals;
    std::vector<double> facet_mids;
    std::vector<Particle> particles;

    Cell() {};

    Cell(std::size_t id, std::vector<std::size_t> neighbors)
    : id(id), neighbors(neighbors) {};
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
    std::size_t gdim;                       ///< Number of geometric dimensions
    std::size_t tdim;                       ///< Number of topological dimensions

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
