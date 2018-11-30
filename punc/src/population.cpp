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

#include "../include/punc/population.h"
#include <dolfin/geometry/Point.h>
#include <dolfin/geometry/BoundingBoxTree.h>

namespace punc {

signed long int locate(std::shared_ptr<const df::Mesh> mesh, const double *x)
{
    auto g_dim = mesh->geometry().dim();
    auto tree = mesh->bounding_box_tree();
    df::Point p(g_dim, x);
    unsigned int cell_id = tree->compute_first_entity_collision(p);

    if (cell_id == UINT32_MAX)
    {
        return -1;
    }
    else
    {
        return cell_id;
    }
}

Species::Species(double charge, double mass, double density, double amount,
                 ParticleAmountType type, const Mesh &mesh,
                 std::shared_ptr<Pdf> pdf, std::shared_ptr<Pdf> vdf, double eps0)
                : pdf(pdf), vdf(vdf) {

    // NB: Deliberate fall-through
    switch(type){
    case ParticleAmountType::per_cell:
        amount *= mesh.mesh->num_cells();
    case ParticleAmountType::in_total:
        amount /= mesh.volume();
    case ParticleAmountType::per_volume:
        amount = density / amount;
    case ParticleAmountType::phys_per_sim:
        break;
    default:
        std::cerr << "Invalid ParticleAmountType";
        exit(1);
    }

    // Amount is now the number of physical particles per simulation particle
    q = charge  * amount;
    m = mass    * amount;
    n = density / amount;

    num = n * mesh.volume();

    debye = vdf->debye(m, q, n, eps0);
}

double min_plasma_period(const std::vector<Species> &species, double eps0){

    double wp_max = 0;
    for(auto &s : species){
        double wp = sqrt(pow(s.q,2)*s.n/(eps0*s.m));
        if(wp > wp_max) wp_max = wp;
    }
    return 2*M_PI/wp_max;
}

double min_gyro_period(const std::vector<Species> &species,
                       const std::vector<double> &B){

    double B_norm = std::accumulate(B.begin(), B.end(), 0.0);
    double wc_max = 0;
    for(auto &s : species){
        double wc = s.q*B_norm/s.m;
        if(wc > wc_max) wc_max = wc;
    }
    return 2*M_PI/wc_max;
}

double max_speed(const std::vector<Species> &species,
                 double k, double phi_min, double phi_max){

    double v_max = 0;
    for(auto &s : species){
        std::vector<double> vd_vec = s.vdf->vd();
        double vd = sqrt(std::accumulate(vd_vec.begin(), vd_vec.end(), 0.0));
        double vth = s.vdf->vth();

        // Maximum and minimum phi on boundary. Not implemented to take into
        // account B-field which may cause it to be non-zero.
        const double phi_bnd_min = 0;
        const double phi_bnd_max = 0;

        double dphi = 0;
        if(s.q<0){
            dphi = phi_max-phi_bnd_min;
        } else {
            dphi = phi_min-phi_bnd_max;
        }
        
        double v_max_s = sqrt(pow(vd+k*vth,2)-2*s.q*dphi/s.m);
        if(v_max_s > v_max) v_max = v_max_s;
    }
    return v_max;
}

} // namespace punc
