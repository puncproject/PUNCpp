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

#include "../include/punc/distributions.h"

namespace punc
{

Maxwellian::Maxwellian(double vth, std::vector<double> &vd, bool has_cdf,
                       bool has_flux_num, bool has_flux_max, double vdf_range) 
                       : _vth(vth), _vd(vd), _dim(vd.size()), _has_cdf(has_cdf),
                       _has_flux_number(has_flux_num), _has_flux_max(has_flux_max)
{
    if (_vth == 0.0)
    {
        _vth = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }
    _domain.resize(2 * _dim);
    _n.resize(_dim);
    for (int i = 0; i < _dim; ++i)
    {
        _domain[i]        = _vd[i] - vdf_range * _vth;
        _domain[i + _dim] = _vd[i] + vdf_range * _vth;
        _n[i] = 1.0;
    }
    vth2 = _vth * _vth;
    factor = (1.0 / (pow(sqrt(2. * M_PI * vth2), _dim)));
}

double Maxwellian::operator()(const std::vector<double> &v)
{
    double v_sqrt = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v_sqrt += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    return factor * exp(-0.5 * v_sqrt / vth2);
}

double Maxwellian::operator()(const std::vector<double> &v, const std::vector<double> &n)
{
    double vn = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        vn += v[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(v);
}

std::vector<double> Maxwellian::cdf(const std::vector<double> &r)
{
    std::size_t N = r.size()/_dim;
    std::vector<double> vs(N * _dim);
    for (auto j = 0; j < _dim; ++j)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            vs[i * _dim + j] = _vd[j] - sqrt(2.0) * _vth * boost::math::erfc_inv(2 * r[i * _dim + j]);
        }
    }
    return vs;
}

void Maxwellian::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double vn, v_sqrt = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v_sqrt += (x[i] - _vd[i]) * (x[i] - _vd[i]);
    }
    if (has_flux)
    {
        vn = 0.0;
        for (int i = 0; i < _dim; ++i)
        {
            vn += x[i] * _n[i];
        }
    }else{
        vn = 1.0;
    }
    values[0] = vn * factor * exp(-0.5 * v_sqrt / vth2)*(vn>0);
}

double Maxwellian::flux_num_particles(const std::vector<double> &n, double S)
{
    auto vdn = std::inner_product(n.begin(), n.end(), _vd.begin(), 0.0);

    auto num_particles = S * (_vth / (sqrt(2 * M_PI)) *
                                  exp(-0.5 * (vdn / _vth) * (vdn / _vth)) +
                              0.5 * vdn * (1. + erf(vdn / (sqrt(2) * _vth))));
    return num_particles;
}

double Maxwellian::flux_max(std::vector<double> &n)
{
    auto vdn = std::inner_product(n.begin(), n.end(), _vd.begin(), 0.0);

    std::vector<double> tmp(_dim);
    for (auto i = 0; i < _dim; ++i)
    {
        tmp[i] = _vd[i] - 0.5 * n[i] * vdn + 0.5 * n[i] * sqrt(4 * vth2 + vdn * vdn);
    }
    return this->operator()(tmp, n);
}

Kappa::Kappa(double vth, std::vector<double> &vd, double k, bool has_cdf,
             bool has_flux_num, bool has_flux_max, double vdf_range)
    : _vth(vth), _vd(vd), k(k), _dim(vd.size()), _has_cdf(has_cdf),
      _has_flux_number(has_flux_num), _has_flux_max(has_flux_max)
{
    assert(k > 1.5 && "kappa must be bigger than 3/2");
    if (_vth == 0.0)
    {
        _vth = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }

    auto sum_vd = std::accumulate(vd.begin(), vd.end(), 0);
    if (sum_vd == 0)
    {
        _has_flux_number = true;
    }
    else
    {
        _has_flux_number = false;
    }

    _domain.resize(2 * _dim);
    for (int i = 0; i < _dim; ++i)
    {
        _domain[i]        = _vd[i] - vdf_range * _vth;
        _domain[i + _dim] = _vd[i] + vdf_range * _vth;
    }
    vth2 = _vth * _vth;
    factor = (1.0 / pow(sqrt(M_PI * (2 * k - _dim) * vth2), _dim)) *
             (tgamma(k + 1.0) / tgamma(k + ((2.0-_dim)/2.0)));
}

double Kappa::operator()(const std::vector<double> &v)
{
    double v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    return factor * pow(1.0 + v2 / ((2 * k - _dim) * vth2), -(k + 1.0));
}

double Kappa::operator()(const std::vector<double> &v, const std::vector<double> &n)
{
    double vn = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        vn += v[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(v);
}

void Kappa::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double vn, v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (x[i] - _vd[i]) * (x[i] - _vd[i]);
    }
    if (has_flux)
    {
        vn = 0.0;
        for (int i = 0; i < _dim; ++i)
        {
            vn += x[i] * _n[i];
        }
    }
    else
    {
        vn = 1.0;
    }
    values[0] = vn * (vn > 0) * factor *
                pow(1.0 + v2 / ((2 * k - _dim) * vth2), -(k + 1.0));
}

/* Number of particles for the case without any drift. */
double Kappa::flux_num_particles(const std::vector<double> &n, double S)
{
    auto num_particles = S * ((_vth / (sqrt(2 * M_PI))) *
                              (1.0/sqrt(k - _dim/2.0)) * 
                              (tgamma(k - ((_dim-1.0)/2.0) ) / 
                               tgamma(k - _dim/2.0)));
    return num_particles;
}

double Kappa::flux_max(std::vector<double> &n)
{
    std::vector<double> tmp(_dim);
    for (auto i = 0; i < _dim; ++i)
    {
        tmp[i] = sqrt( (2.*k-_dim)/(2.*k+1.0) )*n[i]*_vth;
    }
    return this->operator()(tmp, n);
}

Cairns::Cairns(double vth, std::vector<double> &vd, double alpha, bool has_cdf, 
               bool has_flux_num, bool has_flux_max, double vdf_range)
    : _vth(vth), _vd(vd), alpha(alpha), _dim(vd.size()), _has_cdf(has_cdf),
      _has_flux_number(has_flux_num), _has_flux_max(has_flux_max)
{
    if (_vth == 0.0)
    {
        _vth = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }
    _dim = vd.size();
    _domain.resize(2 * _dim);
    for (int i = 0; i < _dim; ++i)
    {
        _domain[i] = _vd[i] - vdf_range * _vth;
        _domain[i + _dim] = _vd[i] + vdf_range * _vth;
    }
    vth2 = _vth * _vth;
    factor = (1.0 / (pow(sqrt(2 * M_PI * vth2), _dim) * (1 + _dim * (_dim + 2) * alpha)));
}

double Cairns::operator()(const std::vector<double> &v)
{
    double v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    double v4 = v2 * v2;
    return factor * (1 + alpha * v4 / pow(vth2, 2)) * exp(-0.5 * v2 / vth2);
}

double Cairns::operator()(const std::vector<double> &v, const std::vector<double> &n)
{
    double vn = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        vn += v[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(v);
}

double Cairns::max()
{
    if (alpha < 0.25)
    {
        return factor;
    }
    else
    {
        std::vector<double> v_max(_dim);
        v_max = _vd;
        v_max[0] += _vth * sqrt(2.0 + sqrt(4.0 - 1.0 / alpha));
        double max = (*this)(v_max);
        max = factor > max ? factor : max;
        return max;
    }
}

void Cairns::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (x[i] - _vd[i]) * (x[i] - _vd[i]);
    }
    double v4 = v2 * v2;

    double vn;
    if (has_flux)
    {
        vn = 0.0;
        for (int i = 0; i < _dim; ++i)
        {
            vn += x[i] * _n[i];
        }
    }
    else
    {
        vn = 1.0;
    }
    values[0] = vn * (vn > 0) * factor *
                (1 + alpha * v4 / pow(vth2, 2)) * exp(-0.5 * v2 / vth2);
}

double Cairns::flux_num_particles(const std::vector<double> &n, double S)
{
    auto vdn = std::inner_product(n.begin(), n.end(), _vd.begin(), 0.0);

    auto num_particles = S * ((_vth / (sqrt(2 * M_PI))) *
                                  exp(-0.5 * (vdn / _vth) * (vdn / _vth)) *
                                  (1 + (_dim + 1) * (_dim + 3) * alpha +
                                   (vdn / _vth) * (vdn / _vth) * alpha) /
                                  (1 + _dim * (_dim + 2) * alpha) +
                              0.5 * vdn * (1. + erf(vdn / (sqrt(2) * _vth))));
    return num_particles;
}

KappaCairns::KappaCairns(double vth, std::vector<double> &vd, double k,
                         double alpha, bool has_cdf, bool has_flux_num, 
                         bool has_flux_max, double vdf_range)
                        : _vth(vth), _vd(vd), k(k), alpha(alpha), 
                        _dim(vd.size()), _has_cdf(has_cdf),
                        _has_flux_number(has_flux_num), 
                        _has_flux_max(has_flux_max)
{
    if (_vth == 0.0)
    {
        _vth = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }
    _dim = vd.size();
    _domain.resize(2 * _dim);
    for (int i = 0; i < _dim; ++i)
    {
        _domain[i] = _vd[i] - vdf_range * _vth;
        _domain[i + _dim] = _vd[i] + vdf_range * _vth;
    }
    vth2 = _vth * _vth;
    factor = (1.0 / pow(sqrt(M_PI * (2 * k - _dim) * vth2), _dim)) *
             (1.0 / (1.0 + _dim * (_dim + 2.0) * alpha * ((k - _dim/2.0) / (k - (_dim+2)/2.0)))) *
             (tgamma(k + 1.0) / tgamma(k + ((2.0 - _dim) / 2.0)));
}

double KappaCairns::operator()(const std::vector<double> &v)
{
    double v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    double v4 = v2 * v2;
    double vth4 = vth2 * vth2;
    return factor * (1.0 + alpha * v4 / vth4) *
           pow(1.0 + v2 / ((2 * k - _dim) * vth2), -(k + 1.0));
}

double KappaCairns::operator()(const std::vector<double> &v, const std::vector<double> &n)
{
    double vn = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        vn += v[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(v);
}

double KappaCairns::max()
{
    std::vector<double> v_max(_dim, 0.0);
    double max;
    v_max = _vd;
    if (alpha >= (k - 1.) * (k + 1.) / ((2*k - _dim ) * (2*k - _dim )))
    {
        v_max[0] += _vth * sqrt(((2.0 * k - _dim) / (k - 1.0)) +
                    sqrt(alpha * alpha * (2 * k - _dim) * (2 * k - _dim) -
                    alpha * (k - 1.0) * (k + 1.0)) / (alpha * (k - 1.0)));
        max = this->operator()(v_max);
    }else{
        max = factor;
    }
    return max;
}

void KappaCairns::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (x[i] - _vd[i]) * (x[i] - _vd[i]);
    }
    double v4 = v2 * v2;
    double vth4 = vth2 * vth2;
    double vn;
    if (has_flux)
    {
        vn = 0.0;
        for (int i = 0; i < _dim; ++i)
        {
            vn += x[i] * _n[i];
        }
    }
    else
    {
        vn = 1.0;
    }
    values[0] = vn * (vn > 0) * factor *
                (1.0 + alpha * v4 / vth4) *
                pow(1.0 + v2 / ((2 * k - _dim) * vth2), -(k + 1.0));
}

} // namespace punc
