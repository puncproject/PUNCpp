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
 * @file		diagnostics.h
 * @brief		Kinetic and potential energy calculations
 *
 * Functions for calculating the kinetic and potential energies.
 */

#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "population.h"
#include <chrono>
#include <iomanip>

namespace punc
{

namespace df = dolfin;

/**
 * @brief Saves and loads the state of simulation and objects
 */
class State 
{
public:
    std::string fname; ///< std::string - name of the state file
    /**
       * @brief   State constructor 
       * @param   fname - a string representing the name of the file
       */
    State(std::string fname);
    /**
       * @brief   Loads the state of the simulation and objects 
       * @param   n - time-steps from previous simulation
       * @param   t - simulated time of previous simulation
       * @param   objects - a vector of objects 
       */
    void load(std::size_t &n, double &t, std::vector<ObjectBC> &objects);
    /**
       * @brief   Saves the state of the simulation and objects 
       * @param   n - number of time-steps of current simulation
       * @param   t - current simulated total time 
       * @param   objects - a vector of objects 
       */
    void save(std::size_t n, double t, std::vector<ObjectBC> &objects);
};

/**
 * @brief Saves simulated quantities (such as time-step, time, energies, and 
 * object state) at each time step
 */
class History
{
public:
    std::ofstream ofile; ///< std::string - name of the history file

    /**
         * @brief   History constructor 
         * @param   fname - a string representing the name of the file
         * @param   objects - a vector of objects
         * @continue_simulation  boolean - if false creates a preamble for history file
         */
    History(std::string fname, std::vector<ObjectBC> &objects, bool continue_simulation = false);

    /**
         * @brief   History destructor - closes the file 
         */
    ~History() { ofile.close(); };

    /**
         * @brief   Saves the history
         * @param   n - time-step
         * @param   t - simulated time
         * @param   num_e - number of electrons in the simulation domain
         * @param   num_i - number of ions in the simulation domain
         * @param   KE  - total kinetic energy
         * @param   PE  - total potential energy
         * @param   objects - a vector of objects
         */
    void save(std::size_t n, double t, double num_e, double num_i, double KE,
              double PE, std::vector<ObjectBC> &objects);

};

/**
 * @brief Measures time for a given set of tasks
 */
class Timer
{
  public:
    /**
       * Timer constructor - starts timing of tasks
       * @param  tasks - a vector of tasks
       */
    Timer(std::vector<std::string> tasks);
    
    /**
       * Resets the timer to current time
       */
    void reset();

    /**
       * Shows the progress of the program and prints the remaining time.
       * @param  n - timestep
       * @param  steps - number of timesteps
       * @param  n_previous - number of timesteps from previous simulation  
       */
    void progress(std::size_t n, std::size_t steps, std::size_t n_previous,
                  bool override_status_print);

    /**
       * Starts measuring time for a given task
       * @param   tag - (std::string) name of the task
       */
    void tic(std::string tag);

    /**
       * Stops measuring time for the task started by tic()
       */
    void toc();

    /**
       * Calculates the elapsed time
       * @return The time elapsed
       */
    double elapsed() const;

    /**
       * Prints the total time elapsed by each task, and print to the screen.
       */
    void summary();

    /**
       * Given a time (in seconds), returns the time in the format day hour:min:sec
       * @param   time_range - time in seconds
       * @return time in the format day hour:min:sec
       */
    std::string formatter(double time_range);

    /**
       * Given a vector of strings, finds the number of blank spaces on the right
       * side of each string so that all the strings in the vector appear to have
       * the same length.
       * @param   v    vector of strings 
       * @return  vector containing number of blank spaces
       */
    std::vector<int> aligner(std::vector<std::string> v);

  private:
    std::vector<std::string> tasks;
    std::vector<double> times;

    typedef std::chrono::high_resolution_clock _clock;
    typedef std::chrono::duration<double, std::ratio<1>> _second;
    typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<24>>::type> days;

    std::chrono::time_point<_clock> _begin;
    std::chrono::time_point<_clock> _time;
    int _index;
};


/**
 * @brief Calculates the total kinetic energy
 * @param[in]   pop     Population
 * @return              Total kinetic energy 
 * 
 * The total kinetic energy is given by 
 * \f[
 *      E_k = \sum_{i=0}^{N}\frac{1}{2}m_i \mathbf{v}_i\cdot\mathbf{v}_i,
 * \f]
 * where \f$N\f$ is the number of particles in the simulation domain.
 */
template <typename PopulationType>
double kinetic_energy(PopulationType &pop)
{
    double KE = 0.0;
    for (auto &cell : pop.cells)
    {
        for (auto &particle : cell.particles)
        {
            auto m = particle.m;
            auto v = particle.v;
            for (std::size_t i = 0; i < pop.g_dim; ++i)
            {
                KE += 0.5 * m * v[i] * v[i];
            }
        }
    }
    return KE;
}

/**
 * @brief Calculates the total potential energy using FEM approach 
 * @param   phi     Electric potential
 * @param   rho     Volume charge density
 * @return          Total potential energy
 *  
 * The total potential energy is given by
 * \f[
 *      E_p = \int_{\Omega} \, \phi \,\rho \, \mathrm{d}x,
 * \f]
 * where \f$\Omega\f$ is the simulation domain.
 */
double mesh_potential_energy(df::Function &phi, df::Function &rho);

/**
 * @brief Calculates the total potential energy by interpolating the electric potential 
 * @param[in]   pop     Population
 * @param       phi     Electric potential
 * @return              Total potential energy
 *  @see particle_potential_energy_cg1
 * 
 * The total potential energy is given by
 * \f[
 *      E_p = \sum_{i=0}^{N}\frac{1}{2}q_i\phi(\mathbf{x}_i),
 * \f]
 * where \f$N\f$ is the number of particles in the simulation domain, and 
 * \f$\mathbf{x}_i\f$ is the position of particle \f$i\f$.
 */
template <typename PopulationType>
double particle_potential_energy(PopulationType &pop, const df::Function &phi)
{
    auto V = phi.function_space();
    auto mesh = V->mesh();
    auto t_dim = mesh->topology().dim();
    auto element = V->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double PE = 0.0;

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, t_dim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        phi.restrict(&coefficients[0], *element, _cell,
                     vertex_coordinates.data(), ufc_cell);

        std::vector<double> basis(v_dim);
        basis_matrix.resize(v_dim);
        for (std::size_t i = 0; i < v_dim; ++i)
        {
            basis_matrix[i].resize(s_dim);
        }

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            std::vector<double> phii(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x,
                                        vertex_coordinates.data(),
                                        cell_orientation);

                for (std::size_t j = 0; j < v_dim; ++j)
                {
                    basis_matrix[j][i] = basis[j];
                }
            }
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                for (std::size_t j = 0; j < v_dim; j++)
                {
                    phii[j] += coefficients[i] * basis_matrix[j][i];
                }
            }
            auto q = particle.q;
            for (std::size_t j = 0; j < v_dim; j++)
            {
                PE += 0.5 * q * phii[j];
            }
        }
    }
    return PE;
}

/**
 * @brief Calculates the total potential energy by interpolating the electric potential in CG1 function space
 * @param[in]   pop     Population
 * @param       phi     Electric potential in CG1
 * @return              Total potential energy
 * @see particle_potential_energy
 * 
 * The total potential energy is given by
 * \f[
 *      E_p = \sum_{i=0}^{N}\frac{1}{2}q_i\phi(\mathbf{x}_i),
 * \f]
 * where \f$N\f$ is the number of particles in the simulation domain, and 
 * \f$\mathbf{x}_i\f$ is the position of particle \f$i\f$.
 */
template <typename PopulationType>
double particle_potential_energy_cg1(PopulationType &pop, const df::Function &phi)
{
    auto V = phi.function_space();
    auto mesh = V->mesh();
    auto element = V->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / v_dim; // Number of vertices

    double phi_x, PE = 0.0;

    double coeffs[n_dim];
    double values[s_dim];
    for (auto &cell : pop.cells)
    {
        phi.restrict(values, *element, cell, cell.vertex_coordinates.data(), cell.ufc_cell);

        for (auto &particle : cell.particles)
        {
            matrix_vector_product(&coeffs[0], cell.basis_matrix.data(),
                                  particle.x, n_dim, n_dim);
            phi_x = 0.0;
            for (std::size_t i = 0; i < n_dim; ++i)
            {
                phi_x += coeffs[i] * values[i];
            }

            PE += 0.5 * particle.q * phi_x;
        }
    }
    return PE;
}

/**
 * @brief                 Volumetric number density in DG0
 * @param[in]   Q         FunctionSpace DG0
 * @param       pop       Population
 * @param      ne, ni     Function - the volumetric number densities
 * @see density_cg1
 * 
 * Calculates the volumetric number density in \f$\mathrm{DG}_0\f$ function 
 * space. The number density in each cell \f$T_k\f$, is simply calculated by 
 * adding together the number of particles of each species, s, inside the cell, 
 * and then dividing the total number inside the cell by the volume of the cell:
 * 
 * \f[
 *       n_{s,k} = \frac{1}{\mathrm{Vol}(T_k)}\sum_{p, q_p=q_s} 1,
 * \f]
 */
template <typename PopulationType>
void density_dg0(const df::FunctionSpace &Q, PopulationType &pop,
                 df::Function &ne, df::Function &ni)
{
    auto ne_vec = ne.vector();
    auto ni_vec = ni.vector();

    std::vector<double> ne0(ne_vec->size());
    std::vector<double> ni0(ni_vec->size());
    ne_vec->get_local(ne0);
    ni_vec->get_local(ni0);

    for (auto &cell : pop.cells)
    {
        auto dof_id = Q.dofmap()->cell_dofs(cell.id);
        double accum_e = 0.0, accum_i = 0.0;
        for (auto &particle : cell.particles)
        {
            if (particle.q>0){
                accum_i += 1;
            }else{
                accum_e += 1;
            }
        }
        ne0[dof_id[0]] = accum_e / cell.volume();
        ni0[dof_id[0]] = accum_i / cell.volume();
    }
    ne.vector()->set_local(ne0);
    ni.vector()->set_local(ni0);
}

/**
 * @brief                 Volumetric number density in CG1
 * @param[in]   V         FunctionSpace CG1
 * @param       pop       Population
 * @param       ne, ni     Function - the volumetric number densities
 * @param       dv_inv    Vector containing the volumes of each element (e.g. Voronoi cell)
 * @see density_dg0()
 * 
 * Calculates the volumetric number density in \f$\mathrm{CG}_1\f$ function 
 * space. The number density at each mesh vertex \f$\mathbf{x}_j\f$, is 
 * calculated by interpolating the number of particles inside all the cells 
 * sharing vertex \f$\mathbf{x}_j\f$, i.e. the patch \f$\mathcal{M}_j\f$. The
 * interpolation is done by evaluating the \f$\mathrm{CG}_1\f$ basis 
 * function \f$\psi_j\f$, at the particle position \f$\mathbf{x}_{p}\f$. The 
 * interpolated number at each mesh vertex is divided by a proper volume 
 * \f$\mathcal{V}_j\f$ associated with \f$\mathbf{x}_j\f$, to obtain the 
 * volumetric number density for each species s: 
 * 
 * \f[
 *       n_{s,j} = \frac{1}{\mathcal{V}_j}\sum_{p, q_p=q_s}\psi_j(\mathbf{x}_{p}).
 * \f]
 */

template <typename PopulationType>
void density_cg1(const df::FunctionSpace &V, PopulationType &pop,
                 df::Function &ne, df::Function &ni, 
                 const std::vector<double> &dv_inv)
{
    auto mesh = V.mesh();
    auto ne_vec = ne.vector();
    auto ni_vec = ni.vector();

    std::vector<double> ne0(ne_vec->size());
    std::vector<double> ni0(ni_vec->size());
    ne_vec->get_local(ne0);
    ni_vec->get_local(ni0);

    auto element = V.element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / v_dim;

    double cell_coords[n_dim];

    for (auto &cell : pop.cells)
    {
        auto dof_id = V.dofmap()->cell_dofs(cell.id);
        std::vector<double> accum_e(n_dim, 0.0);
        std::vector<double> accum_i(n_dim, 0.0);
        for (auto &particle : cell.particles)
        {
            matrix_vector_product(&cell_coords[0], cell.basis_matrix.data(),
                                  particle.x, n_dim, n_dim);

            if (particle.q < 0)
            {
                for (std::size_t i = 0; i < n_dim; ++i)
                {              
                    accum_e[i] += cell_coords[i];
                }
            }else{
                for (std::size_t i = 0; i < n_dim; ++i)
                {
                    accum_i[i] += cell_coords[i];
                }
            }
        }

        for (std::size_t i = 0; i < s_dim; ++i)
        {
            ne0[dof_id[i]] += accum_e[i];
            ni0[dof_id[i]] += accum_i[i];
        }
    }
    for (std::size_t i = 0; i < ne_vec->size(); ++i)
    {
        ne0[i] *= dv_inv[i];
        ni0[i] *= dv_inv[i];
    }
    ne.vector()->set_local(ne0);
    ni.vector()->set_local(ni0);
}

/**
 * @brief                 Exponential moving average
 * @param      f          dolfin Function - data (scalar)
 * @param      g          dolfin Function - averaged data (scalar)
 * @param      dt         timestep
 * @param      tau        relaxation time 
 * 
 * Calculates the exponential moving average of dolfin Function \f$f(x_i,t_i)\f$, 
 * at each element node \f$x_i\f$ at time \f$t_i\f$ by 
 * 
 * \f[
 *       g(x_i, t_i) = \omega f(x_i,t_i) + (1-\omega) g(x_i, t_{i-1}),
 * \f]
 * 
 * where \f$\omega = 1-\exp{(-dt/\tau)}\f$.
 */
template <typename PopulationType>
void ema(const df::Function &f, df::Function &g, double dt, double tau)
{
    double w = 1.0 - exp(-dt / tau);

    auto len_g = g.vector()->size();
    std::vector<double> g0(len_g);
    std::vector<double> f0(len_g);
    g.vector()->get_local(g0);
    f.vector()->get_local(f0);

    for (std::size_t i = 0; i < len_g; ++i)
    {
        g0[i] = w * f0[i]  + (1.0 - w) * g0[i];
    }
    g.vector()->set_local(g0);
}


} // namespace punc

#endif // DIAGNOSTICS_H
