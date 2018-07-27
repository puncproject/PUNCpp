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
#include "population.h"
#include <boost/math/special_functions/erf.hpp>
#include <chrono>
#include <random>

namespace punc
{

namespace df = dolfin;

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

struct Facet
{
    double area;
    std::vector<double> vertices;
    std::vector<double> normal;
    std::vector<double> basis;
};

std::vector<Facet> exterior_boundaries(df::MeshFunction<std::size_t> &boundaries,
                                       std::size_t ext_bnd_id);

class UniformPosition : public Pdf
{
private:
    std::shared_ptr<const df::Mesh> mesh;
    int _dim;
    std::vector<double> _domain;
public:
    UniformPosition(std::shared_ptr<const df::Mesh> mesh) : mesh(mesh)
    {
        _dim = mesh->geometry().dim();
        auto coordinates = mesh->coordinates();
        auto Ld_min = *std::min_element(coordinates.begin(), coordinates.end());
        auto Ld_max = *std::max_element(coordinates.begin(), coordinates.end());
        _domain.resize(2 * _dim);
        for (int i = 0; i < _dim; ++i)
        {
            _domain[i] = Ld_min;
            _domain[i + _dim] = Ld_max;
        }
    }
    double operator()(const std::vector<double> &x) 
    { 
        return (locate(mesh, x) >= 0) * 1.0; 
    };
    double max() { return 1.0;};
    int dim() { return _dim; };
    std::vector<double> domain() { return _domain; };
};

class Maxwellian : public Pdf
{
private:
    double _vth;
    std::vector<double> _vd;
    int _dim;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> n_;
    bool has_flux = false;
    bool has_cdf = true;
public:
    Maxwellian(double vth, std::vector<double> &vd, double vdf_range = 5.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &x, const std::vector<double> &n);
    double max() { return factor; };
    int dim() { return _dim; }
    std::vector<double> domain() { return _domain; };
    double vth() { return _vth; };
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; };
    void set_vd(std::vector<double> &v) { _vd = v; };
    double flux(const std::vector<double> &n) { return 0; };
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        n_ = n;
    }
    double flux_num(const std::vector<double> &n, double S);
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    std::vector<double> cdf(const std::size_t N);
};

class Kappa : public Pdf
{
private:
    double _vth;
    std::vector<double> _vd;
    double k;
    int _dim;
    std::vector<double> _domain;
    double vth2, factor;
public:
    Kappa(double vth, std::vector<double> &vd, double k, double vdf_range = 7.0);
    double operator()(const std::vector<double> &v);
    double max() { return factor; }
    int dim() { return _dim; }
    std::vector<double> domain() { return _domain; }
    double vth() { return _vth; }
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; }
    void set_vd(std::vector<double> &v) { _vd = v; }
    double flux(const std::vector<double> &n) { return 0; }
    double flux_num(const std::vector<double> &n, double S) { return 0; }
};

class Cairns : public Pdf
{
  private:
    double _vth;
    std::vector<double> _vd;
    double alpha;
    int _dim;
    std::vector<double> _domain;
    double vth2, factor;

  public:
    Cairns(double vth, std::vector<double> &vd, double alpha, double vdf_range = 7.0);
    double operator()(const std::vector<double> &v);
    double max();
    int dim() { return _dim; }
    std::vector<double> domain() { return _domain; }
    double vth() { return _vth; }
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; }
    void set_vd(std::vector<double> &v) { _vd = v; }
    double flux(const std::vector<double> &n) { return 0; }
    double flux_num(const std::vector<double> &n, double S);
};

void inject_particles(Population &pop, std::vector<Species> &species,
                      std::vector<Facet> &facets, const double dt);

void load_particles(Population &pop, std::vector<Species> &species);

}

#endif
