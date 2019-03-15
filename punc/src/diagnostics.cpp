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
 * @file		diagnostics.cpp
 * @brief		Kinetic and potential energy calculations
 *
 * Functions for calculating the kinetic and potential energies.
 */

#include "../include/punc/diagnostics.h"
#include "../ufl/Energy.h"

namespace punc
{

FieldWriter::FieldWriter(const std::string &phi_fname, const std::string &E_fname,
                         const std::string &rho_fname, const std::string &ne_fname,
                         const std::string &ni_fname) : ofile_phi(phi_fname),
                         ofile_E(E_fname), ofile_rho(rho_fname), 
                         ofile_ne(ne_fname), ofile_ni(ni_fname)
{
    // Do nothing
}

void FieldWriter::save(const df::Function &phi, const df::Function &E, 
                       const df::Function &rho, const df::Function &ne, 
                       const df::Function &ni, double t)
{
    ofile_phi.write(phi, t);
    ofile_E.write(E, t);
    ofile_rho.write(rho, t);
    ofile_ne.write(ne, t);
    ofile_ni.write(ni, t);
}

State::State(std::string fname) : fname(fname)
{
    // Do nothing
}

void State::load(std::size_t &n, double &t, std::vector<ObjectBC> &objects)
{
    std::ifstream ifile(fname);

    std::string line;
    std::getline(ifile, line);

    char *s = (char *)line.c_str();
    n = strtol(s, &s, 10);
    t = strtod(s, &s);
    for (auto &o : objects)
    {
        o.charge = strtod(s, &s);
        o.current = strtod(s, &s);
    }
    ifile.close();
}
void State::save(std::size_t n, double t, std::vector<ObjectBC> &objects)
{
    std::ofstream ofile;
    ofile.open(fname, std::ofstream::out);
    ofile << n + 1 << "\t" << t << "\t";
    for (auto &o : objects)
    {
        ofile << o.charge << "\t";
        ofile << o.current << "\n";
    }
    ofile.close();
}

History::History(const std::string &fname, std::vector<ObjectBC> &objects, 
                 std::size_t dim, bool continue_simulation)
{
    if (continue_simulation)
    {
        ofile.open(fname, std::ofstream::out | std::ofstream::app);
    }
    else
    {
        ofile.open(fname, std::ofstream::out);

        ofile << "#:xaxis\tt\n";
        ofile << "#:name\tn\tt\tne\tni\tKE\tPE";
        for (std::size_t i = 0; i < objects.size(); ++i)
        {
            ofile << "\tV[" << i <<"]";
            ofile << "\tI[" << i << "]";
            ofile << "\tQ[" << i << "]";
        }
        ofile << "\n";

        ofile << "#:long\ttimestep\ttime\t\"number of electrons\"\t";
        ofile << "\"number of ions\"\t\"kinetic energy\"\t";
        ofile << "\"potential energy\"";
        for (std::size_t i = 0; i < objects.size(); ++i)
        {
            ofile << "\tvoltage";
            ofile << "\tcurrent";
            ofile << "\tcharge";
        }
        ofile << "\n";

        ofile << "#:units\t1\ts\tm**(-3)\tm**(-3)\tJ\tJ";

        for (std::size_t i = 0; i < objects.size(); ++i)
        {
            ofile << "\tV";
            if (dim == 1)
            {
                ofile << "\tA";
            }else if (dim == 2)
            {
                ofile << "\tA/m";
            }else if (dim == 3)
            {
                ofile << "\tA/m**2";
            }
            ofile << "\tC";
        }
        ofile << "\n";
    }
}

void History::save(std::size_t n, double t, double num_e, double num_i, double KE,
                   double PE, std::vector<ObjectBC> &objects)
{
    ofile << n << "\t";
    ofile << t << "\t";
    ofile << num_e << "\t";
    ofile << num_i << "\t";
    ofile << KE << "\t";
    ofile << PE;
    for (auto &o : objects)
    {
        ofile << "\t" << o.potential << "\t";
        ofile << o.current << "\t";
        ofile << o.charge;
    }
    ofile << std::endl;
}

Timer::Timer(std::vector<std::string> tasks) 
            : tasks(tasks), times(std::vector<double>(tasks.size(), 0.0)),
              _begin(_clock::now()) 
{
    // Do nothing
}

void Timer::reset() 
{ 
    _begin = _clock::now(); 
}

void Timer::progress(std::size_t n, std::size_t steps, std::size_t n_previous,
                     bool override_status_print)
{
    auto time_taken = std::chrono::duration_cast<_second>(_clock::now() - _begin).count();
    auto percent = (double)(n - n_previous) / (steps - n_previous);
    auto time_left = time_taken * (1.0 / percent - 1.0);

    if (override_status_print)
        std::cout << "\r";
    std::cout << "Step " << n << " of " << steps;
    if (n - n_previous > 0)
    {
        std::cout << ". Time remaining: ";
        std::cout << std::setw(13) << formatter(time_left);
    }
    if (!override_status_print)
        std::cout << '\n';
}

void Timer::tic(std::string tag)
{
    _index = std::distance(tasks.begin(), std::find(tasks.begin(), tasks.end(), tag));
    _time = _clock::now();
}

void Timer::toc()
{
    times[_index] += std::chrono::duration_cast<_second>(_clock::now() - _time).count();
}

double Timer::elapsed() const
{
    return std::chrono::duration_cast<_second>(_clock::now() - _begin).count();
}

void Timer::summary()
{
    auto total_time = elapsed();

    std::size_t len = 0;
    for (std::size_t i = 0; i < tasks.size(); ++i)
    {
        len = tasks[i].length() > len ? tasks[i].length() : len;
    }

    std::vector<std::string> times_formatted(tasks.size());
    for (std::size_t i = 0; i < tasks.size(); ++i)
    {
        times_formatted[i] = formatter(times[i]);
    }

    std::cout << std::setw(len + 8 + 24 + 5 + 5) << std::setfill('-') << "-" << '\n';
    std::cout << std::setfill(' ') << std::right << std::setw(len + 16) 
              << "Summary of tasks" << '\n';
    std::cout << std::setw(len + 8 + 24 + 5 + 5) << std::setfill('-') << "-" << '\n';
    std::cout << std::setfill(' ') << std::left << std::setw(len + 13) << "Task"
              << std::left << std::setw(16) << "Time"
              << std::left << std::setw(13) << "Percentage" << '\n';
    std::cout << std::setw(len + 8 + 24 + 5 + 5) << std::setfill('-') << "-" << '\n';

    for (std::size_t i = 0; i < times.size(); ++i)
    {
        std::cout <<  std::setfill(' ') 
                  << std::left << std::setw(len + 8) << tasks[i];
        std::cout << std::left << std::setw(24) << times_formatted[i];
        std::cout << std::setw(5) << std::right << std::setfill(' ')
                  << std::setprecision(2) << std::fixed
                  << 100 * times[i] / total_time << '\n';
    }
    std::cout << std::setw(len + 8 + 24 + 5 + 5) << std::setfill('-') << '\n';
    std::cout << std::setfill(' ') << std::right << std::setw(len + 8) 
              << "Total run time:" << formatter(total_time) << '\n';
    std::cout << std::setw(len + 8 + 24 + 5 + 5) << std::setfill('-') << '\n';
}

std::string Timer::formatter(double time_range)
{
    std::ostringstream oss;
    auto duration = std::chrono::duration<double>(time_range);
    if (duration.count() < 1)
    {
        auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        oss << std::setw(11) << int_ms.count() << " ms";
        std::string output(oss.str());
        return output;
    }
    else
    {
        days day = std::chrono::duration_cast<days>(duration);
        duration -= day;
        std::chrono::hours hour = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hour;
        std::chrono::minutes min = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= min;
        std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= sec;

        oss << std::setw(3) << day.count() << " d "
            << std::setw(2) << std::setfill('0') << hour.count() << ':'
            << std::setw(2) << std::setfill('0') << min.count() << ':'
            << std::setw(2) << std::setfill('0') << sec.count();

        std::string output(oss.str());
        return output;
    }
}

double mesh_potential_energy(df::Function &phi, df::Function &rho)
{
    auto mesh = phi.function_space()->mesh();
    auto g_dim = mesh->geometry().dim();
    auto phi_ptr = std::make_shared<df::Function>(phi);
    auto rho_ptr = std::make_shared<df::Function>(rho);
    std::shared_ptr<df::Form> energy;
    switch (g_dim)
    {
    case 1:
        energy = std::make_shared<Energy::Form_0>(mesh, phi_ptr, rho_ptr);
        break;
    case 2:
        energy = std::make_shared<Energy::Form_1>(mesh, phi_ptr, rho_ptr);
        break;
    case 3:
        energy = std::make_shared<Energy::Form_2>(mesh, phi_ptr, rho_ptr);
        break;
    }
    return 0.5 * df::assemble(*energy);
}

} // namespace punc
