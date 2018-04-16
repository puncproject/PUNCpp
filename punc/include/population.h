#ifndef POPULATION_H
#define POPULATION_H

#include <dolfin.h>
#include "injector.h"
#include "object.h"
#include "poisson.h"
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
    void update(boost::optional<std::vector<Object>& > objects = boost::none);
    std::size_t num_of_particles();
    std::size_t num_of_positives();
    std::size_t num_of_negatives();
    void save_file(const std::string &fname);
    void load_file(const std::string &fname);
    void save_vel(const std::string &fname);
};

}

#endif
