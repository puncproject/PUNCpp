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
class Pdf
{
public:
	double _vth;			  	///< Thermal velocity
	std::vector<double> _vd; 	///< Drift velocity
	int dim;                    ///< Dimension of configuration or velocity space
	bool has_icdf;              ///< Whether or not analytical expression for the inverse of CDF exists
	bool has_flux_number;       ///< Whether or not analytical expression for the number of paraticles through a surface exists
	bool has_flux_max;          ///< Whether or not analytical expression for the maximum value of flux PDF exists
	double vdf_range;			///< The range of PDF
	std::vector<double> domain; ///< Vector containing the range of either configuration or velocity space in each dimension

	std::vector<double> pdf_max, num_particles; ///< Vectors containing the maximum value of PDF and the number particles for each facet

	Pdf(){}

	Pdf(double vth, std::vector<double> &vd, bool has_icdf, bool has_flux_num,
		bool has_flux_max, double vdf_range) : _vth(vth), _vd(vd), dim(vd.size()), 
											has_icdf(has_icdf), 
											has_flux_number(has_flux_num), 
											has_flux_max(has_flux_max), vdf_range(vdf_range)
	{
		domain.resize(2 * dim);
		update_domain();
	}

	/**
	 * @brief The PDF it self represented by a mathematical expression
	 * @param[in]  x  - a vector representing either a point in configuration or velocity space
	 * @return     the value of the PDF at x
	 */
	virtual double
	operator()(const std::vector<double> &x) = 0;

	/**
	 * @brief The PDF of the flux for a given normal vector of a facet
	 * @param[in]  x  - a vector representing a point in the velocity space
	 * @param[in]  n  - the normal vector representing a facet
	 * @return     the value of the flux PDF at x
	 */
	virtual double operator()(const std::vector<double> &x, const std::vector<double> &n)
	{
		double vn = 0.0;
		for (int i = 0; i < dim; ++i)
		{
			vn += x[i] * n[i];
		}
		return (vn > 0.0) * vn * this->operator()(x);
	}

	virtual double max() = 0;						 ///< Maximum value of the PDF
	virtual double vth() { return _vth; }			 ///< Returns the thermal velocity
	virtual std::vector<double> vd() { return _vd; } ///< Returns the drift velocity

	/**
	 * @brief Sets a new thermal velocity. Used for normalization.
	 * @param[in]  v  - the (normalized) thermal velocity
	 * 
	 * If thermal velocity is 0, changes the range of VDF accordingly.
	 */
	virtual void set_vth(double v)
	{ 
		_vth = v;
		if (_vth == 0.0)
		{
			_vth = std::numeric_limits<double>::epsilon();
			vdf_range = 1.0;
		}
	}

	/**
	 * @brief Sets a new drift velocity. Used for normalization.
	 * @param[in]  v  - the (normalized) drift velocity
	 * 
	 * Updates the PDF domain, after the new drift velocity has been set. 
	 */
	virtual void set_vd(std::vector<double> &v) 
	{ 
		_vd = v;
		update_domain();
	}

	/**
	 * @brief Updates the PDF domain
	 */
	virtual void update_domain()
	{
		for (int i = 0; i < dim; ++i)
		{
			domain[i]       = _vd[i] - vdf_range * _vth;
			domain[i + dim] = _vd[i] + vdf_range * _vth;
		}
	}

	/**
	 * @brief Analytical expression for the inverse of the CDF
	 * @param[in]  r  - a vector uniformly distributed random numbers
	 * @return     a vector of generated random velocities
	 */
	virtual std::vector<double> icdf(const std::vector<double> &r) { return {}; }

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

public:

	/**
	 * @brief Constructor
	 * @param[in]  df::Mesh  
	 */
	UniformPosition(std::shared_ptr<const df::Mesh> mesh) :  mesh(mesh)
	{
		dim = mesh->geometry().dim();
		auto coordinates = mesh->coordinates();
		auto Ld_min = *std::min_element(coordinates.begin(), coordinates.end());
		auto Ld_max = *std::max_element(coordinates.begin(), coordinates.end());
		domain.resize(2 * dim);
		for (int i = 0; i < dim; ++i)
		{
			domain[i] = Ld_min;
			domain[i + dim] = Ld_max;
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

	double max() { return 1.0; }; ///< Returns the maximum value of the PDF
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
  double vth2, factor;

public:
  std::vector<double> pdf_max, num_particles;

  Maxwellian(double vth, std::vector<double> &vd, bool has_icdf = true,
             bool has_flux_num = true, bool has_flux_max = true,
             double vdf_range = 5.0);
  double operator()(const std::vector<double> &v);
  double operator()(const std::vector<double> &v, const std::vector<double> &n);
  double max() { return factor; };
  std::vector<double> icdf(const std::vector<double> &r);
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
  double vth2, factor;

public:
  double k; /// < Spectral index kappa

  std::vector<double> pdf_max, num_particles;

  Kappa(double vth, std::vector<double> &vd, double k, bool has_icdf = false,
        bool has_flux_num = true, bool has_flux_max = true,
        double vdf_range = 7.0);
  double operator()(const std::vector<double> &v);
  double operator()(const std::vector<double> &v, const std::vector<double> &n);
  double max() { return factor; }
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
  double vth2, vth4, factor;

public:
  double alpha; /// < Spectral index alpha

  std::vector<double> pdf_max, num_particles;

  Cairns(double vth, std::vector<double> &vd, double alpha,
         bool has_icdf = false, bool has_flux_num = true,
         bool has_flux_max = false, double vdf_range = 7.0);
  double operator()(const std::vector<double> &v);
  double operator()(const std::vector<double> &v, const std::vector<double> &n);
  double max();
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
  double vth2, vth4, factor;

public:
  double k;
  double alpha;

  std::vector<double> pdf_max, num_particles;

  KappaCairns(double vth, std::vector<double> &vd, double k, double alpha,
              bool has_icdf = false, bool has_flux_num = false,
              bool has_flux_max = false, double vdf_range = 15.0);
  double operator()(const std::vector<double> &v);
  double operator()(const std::vector<double> &v, const std::vector<double> &n);
  double max();
};

} // namespace punc

#endif // DISTRIBUTIONS_H