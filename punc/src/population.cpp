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

signed long int locate(std::shared_ptr<const df::Mesh> mesh, const std::vector<double> &x)
{
    auto dim = mesh->geometry().dim();
    auto tree = mesh->bounding_box_tree();
    df::Point p(dim, x.data());
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
    D = mesh->geometry().dim();
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
                  (epsilon_0 * pow(X, D));
    }

    q /= Q;
    m /= M;
    n *= pow(X, D);

    vdf.set_vth(vdf.vth()/(X / T));

    std::vector<double> tmp_v(D);
    auto tmp_vd = vdf.vd();

    for (int i = 0; i < D; ++i)
    {
 
        tmp_v[i] = tmp_vd[i] / (X/T);
    }
    vdf.set_vd(tmp_v);

    create_raw(q, m, n, pdf, vdf, npc, num);
}

Population::Population(std::shared_ptr<const df::Mesh> &mesh,
                       const df::MeshFunction<std::size_t> &bnd):mesh(mesh)
{
    num_cells = mesh->num_cells();
    cells.resize(num_cells);
    tdim = mesh->topology().dim();
    gdim = mesh->geometry().dim();

    mesh->init(0, tdim);
    for (df::MeshEntityIterator e(*(mesh), tdim); !e.end(); ++e)
    {
        std::vector<std::size_t> neighbors;
        auto cell_id = e->index();
        auto num_vertices = e->num_entities(0);
        for (std::size_t i = 0; i < num_vertices; ++i)
        {
            df::Vertex vertex(*mesh, e->entities(0)[i]);
            auto vertex_cells = vertex.entities(tdim);
            auto num_adj_cells = vertex.num_entities(tdim);
            for (std::size_t j = 0; j < num_adj_cells; ++j)
            {
                if (cell_id != vertex_cells[j])
                {
                    neighbors.push_back(vertex_cells[j]);
                }
            }
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        Cell cell(mesh, cell_id, neighbors);
        cells[cell_id] = cell;
    }

    init_localizer(bnd);
}

void Population::init_localizer(const df::MeshFunction<std::size_t> &bnd)
{
    mesh->init(tdim - 1, tdim);
    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        std::vector<signed long int> facet_adjacents;
        std::vector<double> facet_normals;
        std::vector<double> facet_mids;

        auto cell_id = e->index();
        df::Cell single_cell(*mesh, cell_id);
        auto facets = e->entities(tdim - 1);
        auto num_facets = e->num_entities(tdim - 1);
        for (std::size_t i = 0; i < num_facets; ++i)
        {
            df::Facet facet(*mesh, e->entities(tdim - 1)[i]);
            auto facet_cells = facet.entities(tdim);
            auto num_adj_cells = facet.num_entities(tdim);
            for (std::size_t j = 0; j < num_adj_cells; ++j)
            {
                if (cell_id != facet_cells[j])
                {
                    facet_adjacents.push_back(facet_cells[j]);
                }
            }
            if (num_adj_cells == 1)
            {
                facet_adjacents.push_back(-1 * bnd.values()[facets[i]]);
            }

            for (std::size_t j = 0; j < gdim; ++j)
            {
                facet_mids.push_back(facet.midpoint()[j]);
                facet_normals.push_back(single_cell.normal(i)[j]);
            }
        }

        cells[cell_id].facet_adjacents = facet_adjacents;
        cells[cell_id].facet_normals = facet_normals;
        cells[cell_id].facet_mids = facet_mids;
    }
}

void Population::add_particles(std::vector<double> &xs, std::vector<double> &vs,
                               double q, double m)
{
    std::size_t num_particles = xs.size() / gdim;
    std::vector<double> xs_tmp(gdim);
    std::vector<double> vs_tmp(gdim);
    std::size_t cell_id;
    for (std::size_t i = 0; i < num_particles; ++i)
    {
        for (std::size_t j = 0; j < gdim; ++j)
        {
            xs_tmp[j] = xs[i * gdim + j];
            vs_tmp[j] = vs[i * gdim + j];
        }
        cell_id = locate(xs_tmp);
        if (cell_id >= 0)
        {
            cells[cell_id].particles.push_back(Particle{xs_tmp, vs_tmp, q, m});
        }
    }
}

signed long int Population::locate(std::vector<double> &p)
{
    return punc::locate(mesh, p);
}

signed long int Population::relocate(std::vector<double> &p, signed long int cell_id)
{
    df::Cell _cell_(*mesh, cell_id);
    df::Point point(gdim, p.data());
    if (_cell_.contains(point))
    {
        return cell_id;
    }
    else
    {
        std::vector<double> proj(gdim + 1);
        for (std::size_t i = 0; i < gdim + 1; ++i)
        {
            proj[i] = 0.0;
            for (std::size_t j = 0; j < gdim; ++j)
            {
                proj[i] += (p[j] - cells[cell_id].facet_mids[i * gdim + j]) *
                        cells[cell_id].facet_normals[i * gdim + j];
            }
        }
        auto projarg = std::distance(proj.begin(), std::max_element(proj.begin(), proj.end()));
        auto new_cell_id = cells[cell_id].facet_adjacents[projarg];
        if (new_cell_id >= 0)
        {
            return relocate(p, new_cell_id);
        }
        else
        {
            return new_cell_id;
        }
    }
}

// FIXME: Consider using default argument rather than boost::optional.
// Empty default vector would be nice, but compiler may not be okay with
// temporaries.
void Population::update(boost::optional<std::vector<ObjectBC>& > objects)
{
    std::size_t num_objects = 0;
    if(objects)
    {
        num_objects = objects->size();
    }

    // FIXME: Consider a different mechanism for boundaries than using negative
    // numbers, or at least circumvent the problem of casting num_cells to
    // signed. Not good practice. size_t may overflow to negative numbers upon
    // truncation for large numbers.
    signed long int new_cell_id;
    for (signed long int cell_id = 0; cell_id < (signed long int)num_cells; ++cell_id)
    {
        std::vector<std::size_t> to_delete;
        std::size_t num_particles = cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = cells[cell_id].particles[p_id];
            new_cell_id = relocate(particle.x, cell_id);
            if (new_cell_id != cell_id)
            {
                to_delete.push_back(p_id);
                if (new_cell_id >= 0)
                {
                    cells[new_cell_id].particles.push_back(particle);
                }else{
                    for (std::size_t i = 0; i < num_objects; ++i)
                    {
                        if((std::size_t)(-new_cell_id) == objects.get()[i].id)
                        {
                            objects.get()[i].charge += particle.q;
                        }
                    }
                }
            }
        }
        std::size_t size_to_delete = to_delete.size();
        for (std::size_t it = size_to_delete; it-- > 0;)
        {
            auto p_id = to_delete[it];
            if (p_id == num_particles - 1)
            {
                cells[cell_id].particles.pop_back();
            }
            else
            {
                std::swap(cells[cell_id].particles[p_id], cells[cell_id].particles.back());
                cells[cell_id].particles.pop_back();
            }
        }
    }
}

std::size_t Population::num_of_particles()
{
    std::size_t num_particles = 0;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        num_particles += cells[cell_id].particles.size();
    }
    return num_particles;
}

std::size_t Population::num_of_positives()
{
    std::size_t num_positives = 0;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = cells[cell_id].particles[p_id];
            if (particle.q > 0)
            {
                num_positives++;
            }
        }
    }
    return num_positives;
}

std::size_t Population::num_of_negatives()
{
    std::size_t num_negatives = 0;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = cells[cell_id].particles[p_id];
            if (particle.q < 0)
            {
                num_negatives++;
            }
        }
    }
    return num_negatives;
}

void Population::save_vel(const std::string &fname)
{
    FILE *fout = fopen(fname.c_str(), "w");
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = cells[cell_id].particles.size();
        if (num_particles > 0)
        {
            for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
            {
                auto particle = cells[cell_id].particles[p_id];
                for (std::size_t i = 0; i < gdim; ++i)
                {
                    fprintf(fout, "%.17g\t", particle.v[i]);
                }
                fprintf(fout, "\n");
            }
        }
    }
    fclose(fout);
}

void Population::save_file(const std::string &fname)
{
    FILE *fout = fopen(fname.c_str(), "w");
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = cells[cell_id].particles.size();
        if (num_particles > 0)
        {
            for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
            {
                auto particle = cells[cell_id].particles[p_id];
                for (std::size_t i = 0; i < gdim; ++i)
                {
                    fprintf(fout, "%.17g\t", particle.x[i]);
                }
                for (std::size_t i = 0; i < gdim; ++i)
                {
                    fprintf(fout, "%.17g\t", particle.v[i]);
                }
                fprintf(fout, "%.17g\t %.17g\t", particle.q, particle.m);
                fprintf(fout, "\n");
            }
        }
    }
    fclose(fout);
}

void Population::load_file(const std::string &fname)
{
    std::fstream in(fname);
    std::string line;
    std::vector<double> x(gdim);
    std::vector<double> v(gdim);
    double q = 0;
    double m = 0;
    std::size_t i;
    while (std::getline(in, line))
    {
        double value;
        std::stringstream ss(line);
        i = 0;
        while (ss >> value)
        {
            if (i < gdim)
            {
                x[i] = value;
            }
            else if (i >= gdim && i < 2 * gdim)
            {
                v[i % gdim] = value;
            }
            else if (i == 2 * gdim)
            {
                q = value;
            }
            else if (i == 2 * gdim + 1)
            {
                m = value;
            }
            ++i;
            add_particles(x,v,q,m);
        }
    }
}

}
