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

    auto blanks_tasks = aligner(tasks);
    std::vector<std::string> times_formatted(tasks.size());
    for (std::size_t i = 0; i < tasks.size(); ++i)
    {
        times_formatted[i] = formatter(times[i]);
    }
    auto blanks_times = aligner(times_formatted);

    std::cout << "-----------------------------------------------------------" << '\n';
    std::cout << "                      Summary of tasks                     " << '\n';
    std::cout << "-----------------------------------------------------------" << '\n';
    std::cout << " Task                        Time             Percentage   " << '\n';
    std::cout << "-----------------------------------------------------------" << '\n';
    for (std::size_t i = 0; i < times.size(); ++i)
    {
        std::cout << tasks[i] << std::setw(blanks_tasks[i]);
        std::cout << times_formatted[i] << std::setw(blanks_times[i]);
        std::cout << std::setprecision(4) << 100 * times[i] / total_time << '\n';
    }

    std::cout << "-----------------------------------------------------------" << '\n';
    std::cout << "            Total run time:    ";
    std::cout << formatter(total_time);
    std::cout << '\n';
    std::cout << "-----------------------------------------------------------" << '\n';
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

std::vector<int> Timer::aligner(std::vector<std::string> v)
{
    std::size_t len = 0;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        len = v[i].length() > len ? v[i].length() : len;
    }
    std::vector<int> blanks(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        blanks[i] = len - v[i].length() + 8;
    }
    return blanks;
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
