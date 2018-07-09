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
#include "injector.h"
#include <fstream>
#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/systems/si/codata/electron_constants.hpp>
#include <boost/units/systems/si/codata/physico-chemical_constants.hpp>
#include <boost/units/systems/si/codata/universal_constants.hpp>

namespace punc
{

namespace df = dolfin;

struct Facet;

struct PhysicalConstants
{
    double e = boost::units::si::constants::codata::e / boost::units::si::coulomb;
    double m_e = boost::units::si::constants::codata::m_e / boost::units::si::kilograms;
    double ratio = boost::units::si::constants::codata::m_e_over_m_p / boost::units::si::dimensionless();
    double m_i = m_e / ratio;

    double k_B = boost::units::si::constants::codata::k_B * boost::units::si::kelvin / boost::units::si::joules;
    double eps0 = boost::units::si::constants::codata::epsilon_0 * boost::units::si::meter / boost::units::si::farad;
};

struct Particle
{
    std::vector<double> x;
    std::vector<double> v;
    double q;
    double m;
};

class Pdf {
public:
    virtual double operator()(const std::vector<double> &x) = 0;
    virtual double flux(const std::vector<double> &n) = 0;
    virtual double flux_num(const std::vector<double> &n, double S) = 0;
    virtual double max() = 0;
};

Kappa vdf(vth, vd, k);
z = vdf(x);
z = vdf.flux(x);

/**
 * @brief Complete specification of a species.
 */
class Species {
    double q; ///< Charge
    double m; ///< Mass
    double n; ///< Density
    int num;  ///< Initial number of particles
    Pdf pdf;  ///< Position distribution function (initially)
    Pdf vdf;  ///< Velocity distribution function (initially and at boundary)

    Species(double q, double m, double n, int num, Pdf pdf, Vdf vdf) :
            q(q), m(m), n(n), num(num), pdf(pdf), vdf(vdf) {}
};

class Species
{
  public:
    double q;
    double m;
    double n;
    int num;
    double vth;
    std::vector<double> vd;
    std::function<double(std::vector<double> &)> pdf;
    double pdf_max;
    std::function<double(std::vector<double> &)> vdf;

    std::shared_ptr<Flux> flux;

    Species(double q, double m, double n, int num, double vth,
            std::vector<double> &vd,
            std::function<double(std::vector<double> &)> pdf,
            double pdf_max, std::vector<Facet> &facets,
            VDFType vdf_type=VDFType::Maxwellian) : q(q), m(m), n(n), num(num),
            vth(vth), vd(vd), pdf(pdf), pdf_max(pdf_max)
    {
        if (vdf_type==VDFType::Maxwellian)
        {
            this->flux = std::make_shared<MaxwellianFlux>(vth, vd, facets);
            this->vdf = maxwellian_vdf(vth, vd);
        }
    }
};

class CreateSpecies
{
  public:
    std::vector<Facet> facets;
    double X;
    int D;
    double volume, num_cells;
    std::vector<Species> species;
    double T = std::numeric_limits<double>::quiet_NaN();
    double Q = boost::units::si::constants::codata::e / boost::units::si::coulomb;
    double M = std::numeric_limits<double>::quiet_NaN();
    double epsilon_0 = boost::units::si::constants::codata::epsilon_0*boost::units::si::meter/boost::units::si::farad;

    CreateSpecies(std::shared_ptr<const df::Mesh> &mesh,
                  std::vector<Facet> &facets, double X);

    void create_raw(double q, double m, double n, int npc, double vth,
                    std::vector<double> &vd,
                    std::function<double(std::vector<double> &)> pdf,
                    double pdf_max);

    void create(double q, double m, double n, int npc, double vth,
                std::vector<double> &vd,
                std::function<double(std::vector<double> &)> pdf,
                double pdf_max);
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

class Population
{
  public:
    std::shared_ptr<const df::Mesh> mesh;
    std::size_t num_cells;
    std::vector<Cell> cells;
    std::size_t gdim, tdim;

    Population(std::shared_ptr<const df::Mesh> &mesh,
               const df::MeshFunction<std::size_t> &bnd);
    void init_localizer(const df::MeshFunction<std::size_t> &bnd);
    void add_particles(std::vector<double> &xs, std::vector<double> &vs,
                       double q, double m);
    signed long int locate(std::vector<double> &p);
    signed long int relocate(std::vector<double> &p, signed long int cell_id);
    void update(boost::optional<std::vector<ObjectBC>& > objects = boost::none);
    std::size_t num_of_particles();
    std::size_t num_of_positives();
    std::size_t num_of_negatives();
    void save_file(const std::string &fname);
    void load_file(const std::string &fname);
    void save_vel(const std::string &fname);
};

}

#endif
