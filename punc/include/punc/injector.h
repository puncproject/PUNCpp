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

#ifndef INJECTOR_H
#define INJECTOR_H

#include <dolfin.h>
#include <boost/math/special_functions/erf.hpp>
#include <chrono>
#include <random>

namespace punc
{

namespace df = dolfin;

class Population;
class Species;

class Timer
{
  public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

  private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};

struct random_seed_seq
{
    template <typename It>
    void generate(It begin, It end)
    {
        for (; begin != end; ++begin)
        {
            *begin = device();
        }
    }

    static random_seed_seq &get_instance()
    {
        static thread_local random_seed_seq result;
        return result;
    }

  private:
    std::random_device device;
};

std::vector<std::vector<double>> combinations(std::vector<std::vector<double>> vec, double dv);

struct Facet
{
    double area;
    std::vector<double> vertices;
    std::vector<double> normal;
    std::vector<double> basis;
};

std::vector<Facet> exterior_boundaries(df::MeshFunction<std::size_t> &boundaries,
                                       std::size_t ext_bnd_id);

class ORS
{
public:
    std::function<double(std::vector<double> &)> vdf;
    int dim, nbins, num_edges;
    std::vector<double> dv;
    std::vector<std::vector<double>> sp;

    std::vector<double> pdf_max;
    std::vector<double> cdf;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;

    ORS(double vth, std::vector<double> &vd,
        std::function<double(std::vector<double> &)> vdf, int num_sp=60);
    std::vector<double> sample(const std::size_t N);
};

enum VDFType {Generic, Maxwellian};

class Flux
{
public:
    std::vector<double> num_particles;
    virtual std::vector<double> sample(const std::size_t N, const std::size_t f){
        std::vector<double> x;
        return x;};
};

class GenericFlux
{
public:

    int dim, nbins, num_edges;
    std::vector<double> dv;
    std::vector<std::vector<double>> sp;

    std::vector<double> pdf_max;
    std::vector<double> cdf;
    std::vector<std::function<double(std::vector<double> &)>> vdf;
    std::vector<double> num_particles;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;

    GenericFlux();
    GenericFlux(double vth, std::vector<double> &vd,
        const std::vector<std::vector<double>> &cutoffs,
        int num_sp,
        std::vector<Facet> &facets);
    std::vector<double> sample(const std::size_t N, const std::size_t f);
};

class MaxwellianFlux : public Flux
{
private:
    std::vector<Facet> facets;

    int nsp;
    int dim;
    double v0, dv;

    std::vector<double> pdf_max;
    std::vector<double> cdf;
    std::vector<std::function<double(double)>> vdf;
    std::vector<std::function<double(double, int)>> maxwellian;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    distribution dist;
    random_source rng;

public:
    MaxwellianFlux(double vth, std::vector<double> &vd, std::vector<Facet> &facets);
    std::vector<double> sample(const std::size_t N, const std::size_t f) override;
};

signed long int locate(std::shared_ptr<const df::Mesh> mesh, std::vector<double> x);

std::function<double(std::vector<double> &)> create_mesh_pdf(std::function<double(std::vector<double> &)> pdf,
                                                             std::shared_ptr<const df::Mesh> mesh);

std::vector<double> random_domain_points(
    std::function<double(std::vector<double> &)> pdf,
    double pdf_max, int N,
    std::shared_ptr<const df::Mesh> mesh);

std::vector<double> random_facet_points(const int N, std::vector<double> &facet_vertices);

std::vector<double> maxwellian(double vth, std::vector<double> vd, const int &N);

std::function<double(std::vector<double> &)> maxwellian_vdf(double vth, std::vector<double> &vd);

void inject_particles(Population &pop, std::vector<Species> &species,
                      std::vector<Facet> &facets, const double dt);

void load_particles(Population &pop, std::vector<Species> &species);

}

#endif
