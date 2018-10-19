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
#include <boost/math/special_functions/erf.hpp>

namespace punc
{

namespace df = dolfin;

/**
 * @brief Finds whether a point x is within the mesh
 * @param mesh    - df::Mesh representing the simulation domain
 * @param x       - a pointer to the array containing the components of the point
 * @return   the index of the cell contaning x. Returns -1 if x is outside the mesh. 
 */
signed long int locate(std::shared_ptr<const df::Mesh> mesh, const double *x);

/**
 * @brief Generic class for probability distribution functions, PDF
 */
class Pdf : public df::Expression
{
private:
  double _vth;             ///< Thermal velocity
  std::vector<double> _vd; ///< Drift velocity
  bool _has_cdf;           ///< Whether or not analytical expression for CDF exists 
  bool _has_flux_number;   ///< Whether or not analytical expression for the number of paraticles through a surface exists
  bool _has_flux_max;      ///< Whether or not analytical expression for the maximum value of flux PDF exists

public:
  std::vector<double> pdf_max, num_particles; ///< Vectors containing the maximum value of PDF and the number particles for each facet

  /**
     * @brief The PDF it self represented by a mathematical expression
     * @param[in]  x  - a vector representing either a point in configuration or velocity space
     * @return     the value of the PDF at x
     */
  virtual double operator()(const std::vector<double> &x) = 0;

  /**
     * @brief The PDF of the flux for a given normal vector of a facet
     * @param[in]  x  - a vector representing a point in the velocity space
     * @param[in]  n  - the normal vector representing a facet
     * @return     the value of the flux PDF at x
     */
  virtual double operator()(const std::vector<double> &x, const std::vector<double> &n)
  {
    double vn = 0.0;
    for (int i = 0; i < dim(); ++i)
    {
      vn += x[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(x);
  }

  virtual int dim() = 0;    ///< The dimension of either configuration or velocity space 
  virtual double max() = 0; ///< Maximum value of the PDF
  virtual std::vector<double> domain() = 0; ///< Vector containing the range of either configuration or velocity space in each dimension
  virtual double vth() { return _vth; }     ///< Returns the thermal velocity
  virtual std::vector<double> vd() { return _vd; } ///< Returns the drift velocity
  virtual void set_vth(double v) { _vth = v; }     ///< Sets the normalized thermal velocity
  virtual void set_vd(std::vector<double> &v) { _vd = v; } ///< Sets the normalized drift velocity
  virtual bool has_cdf() { return _has_cdf; }              ///< Returns true if PDF has an analytical expression for the CDF
  virtual bool has_flux_max() { return _has_flux_max; }    ///< Returns true if the maximum value of the flux PDF is available
  virtual bool has_flux_number() { return _has_flux_number; } ///< Returns true if analytical expression for the flux particle number is available
  /**
     * @brief Analytical expression for the inverse of the CDF
     * @param[in]  r  - a vector uniformly distributed random numbers
     * @return     a vector of generated random velocities
     */
  virtual std::vector<double> cdf(const std::vector<double> &r) { return {}; }
  virtual void set_flux_normal(std::vector<double> &n) {} ///< Sets the normal vector for the flux PDF

  /**
     * @brief Number of particles for a given facet with normal vector n and area S
     * @param[in]  n  - the normal vector representing a facet
     * @param[in]  S  - the area of the facet
     * @return     number of particles to be injected through the facet
     */
  virtual double flux_num_particles(const std::vector<double> &n, double S) { return 0.0; }
  
  /**
     * @brief The maximum value of the flux PDF
     * @param[in]  n  - the normal vector representing a facet
     * @return     the maximum value of the flux PDF
     */
  virtual double flux_max(std::vector<double> &n) { return 0.0; };
};

/**
 * @brief Generic class for a uniform distribution of particle positions
 */
class UniformPosition : public Pdf
{
  private:
    std::shared_ptr<const df::Mesh> mesh; ///< df::Mesh of the simulation domain
    int _dim;                             ///< The geometrical dimension of the domain
    std::vector<double> _domain;          ///< The range of the domain in each dimension

  public:
    /**
     * @brief Constructor
     * @param[in]  df::Mesh  
     */
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

    /**
     * @brief The uniform distribution
     * @param[in]  x   - a vector representing a point in the domain
     * @return    1 if the point is within the domain, otherwise 0
     */
    double operator()(const std::vector<double> &x)
    {
        return (locate(mesh, x.data()) >= 0) * 1.0;
    };

    double max() { return 1.0; };                     ///< Returns the maximum value of the PDF
    int dim() { return _dim; };                       ///< Returns the geometrical dimension of the domain
    std::vector<double> domain() { return _domain; }; ///< Returns the range of the domain in each dimension
};

/**
 * @brief Maxwellian velocity distribution function
 * 
 * 
 * The Maxwellian velocity distribution function is given by
 * \f[
 *      f(\mathbf{v}; v_{\mathrm{th}}, \mathbf{v}_{\mathrm{D}}) = \frac{1}{(2\pi v_{\mathrm{th}})^{\frac{D}{2}}} \exp{-\frac{(\mathbf{v}-\mathbf{v}_{\mathrm{D}})^2}{2v_{\mathrm{th}}^2}},
 * \f]
 * where \f[D\f] is the geometrical dimension.
 */
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

/**
 * @brief Kappa velocity distribution function
 * 
 * 
 * The Kappa velocity distribution function is given by
 * \f[
 *      f(\mathbf{v}; v_{\mathrm{th}}, \kappa) = A_{\kappa}\bigg( 1 + \frac{v^2}{2(\kappa - \frac{D}{2}) v_{\mathrm{th}}^2}\bigg)^{-(\kappa + 1)},
 * \f]
 * where 
 * \f[
 *  A_{\kappa} = \frac{1}{(2\pi v_{\mathrm{th}}^2 (\kappa -\frac{D}{2}) )^{\frac{D}{2}}}\frac{\Gamma(\kappa + 1 )}{\Gamma(\kappa + \frac{2-D}{2})}
 * \f] 
 * is the normalization factor.
 */
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

/**
 * @brief Cairns velocity distribution function
 * 
 * 
 * 
 * The Cairns velocity distribution function is given by
 * \f[
 *      f(\mathbf{v}; v_{\mathrm{th}}, \alpha) = A_{\alpha}\left( 1 + \alpha\frac{v^4}{v_{\mathrm{th}}^4}\right) \exp{\left(-\frac{v^2}{2 v_{\mathrm{th}}^2}\right)},
 * \f]
 * where 
 * \f[
 *  A_{\alpha} = \frac{1}{(2\pi v_{\mathrm{th},s}^2)^{\frac{D}{2}} \left(1+D(D+2)\alpha\right)}
 * \f] 
 * is the normalization factor.
 */
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

/**
 * @brief Kappa-Cairns velocity distribution function
 * 
 * 
 * 
 * The Kappa-Cairns velocity distribution function is given by
 * \f[
 *      f(\mathbf{v}; v_{\mathrm{th}},\kappa, \alpha) = A_{\kappa,\alpha}
 *      \left( 1 + \alpha\frac{v^4}{v_{\mathrm{th}}^4}\right)
 *      \left( 1 + \frac{v^2}{2(\kappa - \frac{D}{2} ) v_{\mathrm{th}}^2}\right)^{-(\kappa + 1)},
 * \f]
 * where 
 * \f[
 *  A_{\kappa,\alpha} =\frac{ \Gamma(\kappa + 1 )/\Gamma(\kappa + \frac{2-D}{2})  }{(2\pi v_{\mathrm{th}}^2 (\kappa -\frac{D}{2}))^{\frac{D}{2}} \left[1+D(D+2)\alpha\frac{\kappa-\frac{D}{2}}{\kappa-\frac{D+2}{2}}\right] },
 *  \qquad \qquad \kappa>\frac{D}{2}+1,
 * \f] 
 * is the normalization factor.
 */
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