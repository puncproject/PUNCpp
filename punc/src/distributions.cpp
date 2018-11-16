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

Maxwellian::Maxwellian(double vth, std::vector<double> &vd, bool has_icdf,
                       bool has_flux_num, bool has_flux_max, double vdf_range)
                      : Pdf(vth, vd, true, true, true, vdf_range)
{
    vth2 = _vth * _vth;
    factor = (1.0 / (pow(sqrt(2. * M_PI * vth2), dim)));
}

double Maxwellian::operator()(const std::vector<double> &v)
{
    double v_sqrt = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        v_sqrt += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    return factor * exp(-0.5 * v_sqrt / vth2);
}

std::vector<double> Maxwellian::icdf(const std::vector<double> &r)
{
    std::size_t N = r.size()/dim;
    std::vector<double> vs(N * dim);
    for (auto j = 0; j < dim; ++j)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            vs[i * dim + j] = _vd[j] - sqrt(2.0) * _vth * boost::math::erfc_inv(2 * r[i * dim + j]);
        }
    }
    return vs;
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

    std::vector<double> tmp(dim);
    for (auto i = 0; i < dim; ++i)
    {
        tmp[i] = _vd[i] - 0.5 * n[i] * vdn + 0.5 * n[i] * sqrt(4 * vth2 + vdn * vdn);
    }
    return Pdf::operator()(tmp, n);
}

double Maxwellian::debye(double m, double q, double n, double eps0)
{
    return sqrt(eps0*m/(n*q*q))*_vth;
}

Kappa::Kappa(double vth, std::vector<double> &vd, double k, bool has_icdf,
             bool has_flux_num, bool has_flux_max, double vdf_range)
            : Pdf(vth, vd, false, true, true, vdf_range), k(k)
{
    assert(k > 1.5 && "kappa must be bigger than 3/2");

    auto sum_vd = std::accumulate(_vd.begin(), _vd.end(), 0);
    if (sum_vd == 0)
    {
        has_flux_number = true;
    }
    else
    {
        has_flux_number = false;
    }

    vth2 = _vth * _vth;
    factor = (1.0 / pow(sqrt(M_PI * (2 * k - dim) * vth2), dim)) *
             (tgamma(k + 1.0) / tgamma(k + ((2.0-dim)/2.0)));
}

double Kappa::operator()(const std::vector<double> &v)
{
    double v2 = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        v2 += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    return factor * pow(1.0 + v2 / ((2 * k - dim) * vth2), -(k + 1.0));
}

/* Number of particles for the case without any drift. */
double Kappa::flux_num_particles(const std::vector<double> &n, double S)
{
    auto num_particles = S * ((_vth / (sqrt(2 * M_PI))) *
                              (1.0/sqrt(k - dim/2.0)) * 
                              (tgamma(k - ((dim-1.0)/2.0) ) / 
                               tgamma(k - dim/2.0)));
    return num_particles;
}

double Kappa::flux_max(std::vector<double> &n)
{
    std::vector<double> tmp(dim);
    for (auto i = 0; i < dim; ++i)
    {
        tmp[i] = sqrt( (2.*k-dim)/(2.*k+1.0) )*n[i]*_vth;
    }
    return Pdf::operator()(tmp, n);
}

double Kappa::debye(double m, double q, double n, double eps0)
{
    double B = (k-0.5)/(k-1.5);
    return sqrt(eps0 * m / (n * q * q * B)) * _vth;
}

Cairns::Cairns(double vth, std::vector<double> &vd, double alpha, bool has_icdf,
               bool has_flux_num, bool has_flux_max, double vdf_range)
              : Pdf(vth, vd, false, true, false, vdf_range), alpha(alpha)
{
    vth2 = _vth * _vth;
    vth4 = vth2 * vth2;
    factor = (1.0 / (pow(sqrt(2 * M_PI * vth2), dim) * (1 + dim * (dim + 2) * alpha)));
}

double Cairns::operator()(const std::vector<double> &v)
{
    double v2 = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        v2 += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    double v4 = v2 * v2;
    return factor * (1 + alpha * v4 / vth4) * exp(-0.5 * v2 / vth2);
}

double Cairns::max()
{
    if (alpha < 0.25)
    {
        return factor;
    }
    else
    {
        std::vector<double> v_max(dim);
        v_max = _vd;
        v_max[0] += _vth * sqrt(2.0 + sqrt(4.0 - 1.0 / alpha));
        double max = (*this)(v_max);
        max = factor > max ? factor : max;
        return max;
    }
}

double Cairns::flux_num_particles(const std::vector<double> &n, double S)
{
    auto vdn = std::inner_product(n.begin(), n.end(), _vd.begin(), 0.0);

    auto num_particles = S * ((_vth / (sqrt(2 * M_PI))) *
                         exp(-0.5 * (vdn / _vth) * (vdn / _vth)) *
                         (1. + (dim + 1.) * (dim + 3.) * alpha +
                         (vdn / _vth) * (vdn / _vth) * alpha) /
                         (1. + dim * (dim + 2.) * alpha) +
                         0.5 * vdn * (1. + erf(vdn / (sqrt(2) * _vth))));
    return num_particles;
}

double Cairns::debye(double m, double q, double n, double eps0)
{
    double B = (1. + 3. * alpha) / (1. + 15 * alpha);
    return sqrt(eps0 * m / (n * q * q * B)) * _vth;
}

KappaCairns::KappaCairns(double vth, std::vector<double> &vd, double k,
                         double alpha, bool has_icdf, bool has_flux_num,
                         bool has_flux_max, double vdf_range)
                        : Pdf(vth, vd, false, false, false, vdf_range), 
                          k(k), alpha(alpha)

{
    vth2 = _vth * _vth;
    vth4 = vth2 * vth2;
    factor = (1.0 / pow(sqrt(M_PI * (2. * k - dim) * vth2), dim)) *
             (1.0 / (1.0 + dim * (dim + 2.0) * alpha * ((k - dim/2.0) / (k - (dim+2)/2.0)))) *
             (tgamma(k + 1.0) / tgamma(k + ((2.0 - dim) / 2.0)));
}

double KappaCairns::operator()(const std::vector<double> &v)
{
    double v2 = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        v2 += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    double v4 = v2 * v2;
    return factor * (1.0 + alpha * v4 / vth4) *
           pow(1.0 + v2 / ((2 * k - dim) * vth2), -(k + 1.0));
}

double KappaCairns::max()
{
    std::vector<double> v_max(dim, 0.0);
    double max;
    v_max = _vd;
    if (alpha >= (k - 1.) * (k + 1.) / ((2.*k - dim ) * (2.*k - dim )))
    {
        v_max[0] += _vth * sqrt(((2.0 * k - dim) / (k - 1.0)) +
                    sqrt(alpha * alpha * (2 * k - dim) * (2 * k - dim) -
                    alpha * (k - 1.0) * (k + 1.0)) / (alpha * (k - 1.0)));
        max = this->operator()(v_max);
    }else{
        max = factor;
    }
    return max;
}

double KappaCairns::debye(double m, double q, double n, double eps0)
{
    double B = ((k - 0.5) / (k - 1.5)) * ((1. + 3. * alpha * ((k - 1.5) / (k - 0.5))) / (1. + 15 * alpha * ((k - 1.5) / (k - 2.5))));
    return sqrt(eps0 * m / (n * q * q * B)) * _vth;
}

} // namespace punc
