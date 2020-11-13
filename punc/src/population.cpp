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
    weight = amount;
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

/**
 * @brief Generates the basis matrix (in 1D) used to transform physical 
 * coordinates of a given particle position to the corresponding barycentric 
 * coordinates of the cell 
 */
template<>
void Cell<1>::init_barycentric_matrix()
{
    double x1 = vertex_coordinates[0];
    double x2 = vertex_coordinates[1];

    double det = x2 - x1;

    // Row 1
    barycentric_matrix[0] = x2 / det;
    barycentric_matrix[1] = -1.0 / det;

    // Row 2 (not used due to optimizations)
    // barycentric_matrix[2] = -x1 / det;
    // barycentric_matrix[3] = 1.0 / det;
}

/**
 * @brief Generates the basis matrix (in 2D) used to transform physical 
 * coordinates of a given particle position to the corresponding barycentric 
 * coordinates of the cell 
 */
template<>
void Cell<2>::init_barycentric_matrix()
{
    double x1 = vertex_coordinates[0];
    double y1 = vertex_coordinates[1];
    double x2 = vertex_coordinates[2];
    double y2 = vertex_coordinates[3];
    double x3 = vertex_coordinates[4];
    double y3 = vertex_coordinates[5];

    double det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);

    // Row 1
    barycentric_matrix[0] = (x2 * y3 - x3 * y2) / det;
    barycentric_matrix[1] = (y2 - y3) / det;
    barycentric_matrix[2] = (x3 - x2) / det;

    // Row 2
    barycentric_matrix[3] = (x3 * y1 - x1 * y3) / det;
    barycentric_matrix[4] = (y3 - y1) / det;
    barycentric_matrix[5] = (x1 - x3) / det;

    // Row 3 (not used due to optimizations)
    // barycentric_matrix[6] = (x1 * y2 - x2 * y1) / det;
    // barycentric_matrix[7] = (y1 - y2) / det;
    // barycentric_matrix[8] = (x2 - x1) / det;
}

/**
 * @brief Generates the basis matrix (in 3D) used to transform physical 
 * coordinates of a given particle position to the corresponding barycentric 
 * coordinates of the cell 
 */
template<>
void Cell<3>::init_barycentric_matrix()
{
    double x1 = vertex_coordinates[0];
    double y1 = vertex_coordinates[1];
    double z1 = vertex_coordinates[2];
    double x2 = vertex_coordinates[3];
    double y2 = vertex_coordinates[4];
    double z2 = vertex_coordinates[5];
    double x3 = vertex_coordinates[6];
    double y3 = vertex_coordinates[7];
    double z3 = vertex_coordinates[8];
    double x4 = vertex_coordinates[9];
    double y4 = vertex_coordinates[10];
    double z4 = vertex_coordinates[11];

    // double x12 = x1 - x2; double y12 = y1 - y2; double z12 = z1 - z2;
    double x13 = x1 - x3; double y13 = y1 - y3; double z13 = z1 - z3;
    double x14 = x1 - x4; double y14 = y1 - y4; double z14 = z1 - z4;
    // double x21 = x2 - x1; double y21 = y2 - y1; double z21 = z2 - z1;
    double x24 = x2 - x4; double y24 = y2 - y4; double z24 = z2 - z4;
    double x31 = x3 - x1; double y31 = y3 - y1; double z31 = z3 - z1;
    double x32 = x3 - x2; double y32 = y3 - y2; double z32 = z3 - z2;
    double x34 = x3 - x4; double y34 = y3 - y4; double z34 = z3 - z4;
    double x42 = x4 - x2; double y42 = y4 - y2; double z42 = z4 - z2;
    double x43 = x4 - x3; double y43 = y4 - y3; double z43 = z4 - z3;

    double V01 = (x2 * (y3 * z4 - y4 * z3) + x3 * (y4 * z2 - y2 * z4) + x4 * (y2 * z3 - y3 * z2)) / 6.0;
    double V02 = (x1 * (y4 * z3 - y3 * z4) + x3 * (y1 * z4 - y4 * z1) + x4 * (y3 * z1 - y1 * z3)) / 6.0;
    double V03 = (x1 * (y2 * z4 - y4 * z2) + x2 * (y4 * z1 - y1 * z4) + x4 * (y1 * z2 - y2 * z1)) / 6.0;
    double V04 = (x1 * (y3 * z2 - y2 * z3) + x2 * (y1 * z3 - y3 * z1) + x3 * (y2 * z1 - y1 * z2)) / 6.0;
    double V = V01 + V02 + V03 + V04;
    double V6 = 6 * V;

    // Row 4 not used due to optimizations

    // Column 1
    barycentric_matrix[0] = V01 / V;
    barycentric_matrix[4] = V02 / V;
    barycentric_matrix[8] = V03 / V;
    // barycentric_matrix[12] = V04 / V;

    // Column 2
    barycentric_matrix[1] = (y42 * z32 - y32 * z42) / V6;  
    barycentric_matrix[5] = (y31 * z43 - y34 * z13) / V6; 
    barycentric_matrix[9] = (y24 * z14 - y14 * z24) / V6;  
    // barycentric_matrix[13] = (y13 * z21 - y12 * z31) / V6; 

    // Column 3
    barycentric_matrix[2] = (x32 * z42 - x42 * z32) / V6;  
    barycentric_matrix[6] = (x43 * z31 - x13 * z34) / V6;  
    barycentric_matrix[10] = (x14 * z24 - x24 * z14) / V6; 
    // barycentric_matrix[14] = (x21 * z13 - x31 * z12) / V6; 

    // Column 4
    barycentric_matrix[3] = (x42 * y32 - x32 * y42) / V6;  
    barycentric_matrix[7] = (x31 * y43 - x34 * y13) / V6;  
    barycentric_matrix[11] = (x24 * y14 - x14 * y24) / V6; 
    // barycentric_matrix[15] = (x13 * y21 - x12 * y31) / V6; 
}


} // namespace punc
