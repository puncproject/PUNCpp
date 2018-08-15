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

namespace punc
{

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

CreateSpecies::CreateSpecies(std::shared_ptr<const df::Mesh> &mesh, double X)
                             : X(X)
{
    g_dim = mesh->geometry().dim();
    volume = punc::volume(mesh);
    num_cells = mesh->num_cells();
}

void CreateSpecies::create_raw(double q, double m, double n, Pdf &pdf, Pdf &vdf, 
                               int npc, int num)
{
    if (num==0)
    {
        num = npc * num_cells;
    }
    double w = (n / num) * volume;
    q *= w;
    m *= w;
    n /= w;

    Species s(q, m, n, num, pdf, vdf);
    species.emplace_back(s);
}

void CreateSpecies::create(double q, double m, double n, Pdf &pdf, Pdf &vdf,
                           int npc, int num)
{
    if (std::isnan(T))
    {
        double wp = sqrt((n * q * q) / (epsilon_0 * m));
        T = 1.0 / wp;
    }
    if (std::isnan(M))
    {
        if (num==0)
        {
            num = npc * num_cells;
        }
        double w =  (n / num) * volume;
        Q *= w;
        
        M = (T * T * Q * Q) /
                  (epsilon_0 * pow(X, g_dim));
    }

    q /= Q;
    m /= M;
    n *= pow(X, g_dim);

    vdf.set_vth(vdf.vth()/(X / T));

    std::vector<double> tmp_v(g_dim);
    auto tmp_vd = vdf.vd();

    for (int i = 0; i < g_dim; ++i)
    {
 
        tmp_v[i] = tmp_vd[i] / (X/T);
    }
    vdf.set_vd(tmp_v);

    create_raw(q, m, n, pdf, vdf, npc, num);
}

}
