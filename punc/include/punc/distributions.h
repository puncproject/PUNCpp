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
 * @file		distributions.h
 * @brief		configuration and velocity distribution functions
 *
 * Configuration and velocity distribution functions.
 */

#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <dolfin.h>
#include "population.h"
#include <boost/math/special_functions/erf.hpp>

namespace punc
{

namespace df = dolfin;

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
        return (locate(mesh, x.data()) >= 0) * 1.0;
    };
    double max() { return 1.0; };
    int dim() { return _dim; };
    std::vector<double> domain() { return _domain; };
};

class Maxwellian : public Pdf
{
  private:
    double _vth;
    std::vector<double> _vd;
    int _dim;
    bool _has_cdf;
    bool _has_flux_number;
    bool _has_flux_max;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> _n;
    bool has_flux = false;

  public:
    std::vector<double> pdf_max, num_particles;

    Maxwellian(double vth, std::vector<double> &vd, bool has_cdf = true,
               bool has_flux_num = true, bool has_flux_max = true,
               double vdf_range = 5.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &v, const std::vector<double> &n);
    int dim() { return _dim; }
    double max() { return factor; };
    std::vector<double> domain() { return _domain; };
    double vth() { return _vth; };
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; };
    void set_vd(std::vector<double> &v) { _vd = v; };
    bool has_cdf() { return _has_cdf; };
    bool has_flux_max() { return _has_flux_max; };
    bool has_flux_number() { return _has_flux_number; };
    std::vector<double> cdf(const std::vector<double> &r);
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        _n = n;
    }
    double flux_num_particles(const std::vector<double> &n, double S);
    double flux_max(std::vector<double> &n);
};

class Kappa : public Pdf
{
  private:
    double _vth;
    std::vector<double> _vd;
    double k;
    int _dim;
    bool _has_cdf;
    bool _has_flux_number;
    bool _has_flux_max;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> _n;
    bool has_flux = false;

  public:
    std::vector<double> pdf_max, num_particles;
    Kappa(double vth, std::vector<double> &vd, double k, bool has_cdf = false,
          bool has_flux_num = true, bool has_flux_max = true,
          double vdf_range = 7.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &v, const std::vector<double> &n);
    int dim() { return _dim; }
    double max() { return factor; }
    std::vector<double> domain() { return _domain; }
    double vth() { return _vth; }
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; }
    void set_vd(std::vector<double> &v) { _vd = v; }
    bool has_cdf() { return _has_cdf; };
    bool has_flux_max() { return _has_flux_max; };
    bool has_flux_number() { return _has_flux_number; };
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        _n = n;
    }
    double flux_num_particles(const std::vector<double> &n, double S);
    double flux_max(std::vector<double> &n);
};

class Cairns : public Pdf
{
  private:
    double _vth;
    std::vector<double> _vd;
    double alpha;
    int _dim;
    bool _has_cdf;
    bool _has_flux_number;
    bool _has_flux_max;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> _n;
    bool has_flux = false;

  public:
    std::vector<double> pdf_max, num_particles;
    Cairns(double vth, std::vector<double> &vd, double alpha,
           bool has_cdf = false, bool has_flux_num = true,
           bool has_flux_max = false, double vdf_range = 7.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &v, const std::vector<double> &n);
    int dim() { return _dim; }
    double max();
    std::vector<double> domain() { return _domain; }
    double vth() { return _vth; }
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; }
    void set_vd(std::vector<double> &v) { _vd = v; }
    bool has_cdf() { return _has_cdf; };
    bool has_flux_max() { return _has_flux_max; };
    bool has_flux_number() { return _has_flux_number; };
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        _n = n;
    }
    double flux_num_particles(const std::vector<double> &n, double S);
};

class KappaCairns : public Pdf
{
  private:
    double _vth;
    std::vector<double> _vd;
    double k;
    double alpha;
    int _dim;
    bool _has_cdf;
    bool _has_flux_number;
    bool _has_flux_max;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> _n;
    bool has_flux = false;

  public:
    std::vector<double> pdf_max, num_particles;
    KappaCairns(double vth, std::vector<double> &vd, double k, double alpha,
                bool has_cdf = false, bool has_flux_num = false,
                bool has_flux_max = false, double vdf_range = 35.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &v, const std::vector<double> &n);
    int dim() { return _dim; }
    double max();
    std::vector<double> domain() { return _domain; }
    double vth() { return _vth; }
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; }
    void set_vd(std::vector<double> &v) { _vd = v; }
    bool has_cdf() { return _has_cdf; };
    bool has_flux_max() { return _has_flux_max; };
    bool has_flux_number() { return _has_flux_number; };
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        _n = n;
    }
};

} // namespace punc

#endif // INJECTOR_H