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

#include "../include/punc/injector.h"

namespace punc
{

typedef std::uniform_real_distribution<double> rand_uniform;
typedef std::vector<std::uniform_real_distribution<double>> rand_uniform_vec;

std::vector<Facet> exterior_boundaries(df::MeshFunction<std::size_t> &boundaries,
                                       std::size_t ext_bnd_id)
{
    auto mesh = boundaries.mesh();
    auto g_dim = mesh->geometry().dim();
    auto t_dim = mesh->topology().dim();
    auto values = boundaries.values();
    auto length = boundaries.size();
    int num_facets = 0;
    for (std::size_t i = 0; i < length; ++i)
    {
        if (ext_bnd_id == values[i])
        {
            num_facets += 1;
        }
    }
    std::vector<Facet> ext_facets;
 
    double area = 0.0;
    std::vector<double> normal(g_dim);
    std::vector<double> vertices(g_dim * g_dim);
    std::vector<double> basis(g_dim * g_dim);
    std::vector<double> vertex(g_dim);
    double norm;
    int j;
    mesh->init(t_dim - 1, t_dim);
    df::SubsetIterator facet_iter(boundaries, ext_bnd_id);
    for (; !facet_iter.end(); ++facet_iter)
    {
        df::Cell cell(*mesh, facet_iter->entities(t_dim)[0]);
        auto cell_facet = cell.entities(t_dim - 1);
        std::size_t num_facets = cell.num_entities(t_dim - 1);
        for (std::size_t i = 0; i < num_facets; ++i)
        {
            if (cell_facet[i] == facet_iter->index())
            {
                area = cell.facet_area(i);
                for (std::size_t j = 0; j < g_dim; ++j)
                {
                    normal[j] = -1 * cell.normal(i, j);
                    basis[j * g_dim] = normal[j];
                }
            }
        }
        assert(area != 0.0 && "The facet area cannot be zero!");

        j = 0;
        for (df::VertexIterator v(*facet_iter); !v.end(); ++v)
        {
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                vertices[j * g_dim + i] = v->x(i);
            }
            j += 1;
        }
        norm = 0.0;
        for (std::size_t i = 0; i < g_dim; ++i)
        {
            vertex[i] = vertices[i] - vertices[g_dim + i];
            norm += vertex[i] * vertex[i];
        }
        for (std::size_t i = 0; i < g_dim; ++i)
        {
            vertex[i] /= sqrt(norm);
            basis[i * g_dim + 1] = vertex[i];
        }

        if (g_dim == 3)
        {
            basis[2] = normal[1] * vertex[2] - normal[2] * vertex[1];
            basis[5] = normal[2] * vertex[0] - normal[0] * vertex[2];
            basis[8] = normal[0] * vertex[1] - normal[1] * vertex[0];
        }
        ext_facets.push_back(Facet{area, vertices, normal, basis});
    }
    return ext_facets;
}

Maxwellian::Maxwellian(double vth, std::vector<double> &vd, 
                       double vdf_range) : vth_(vth), vd_(vd), dim_(vd.size())
{
    if (vth_ == 0.0)
    {
        vth_ = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }
    domain_.resize(2 * dim_);
    n_.resize(dim_);
    for (int i = 0; i < dim_; ++i)
    {
        domain_[i] = -vdf_range * vth_;
        domain_[i + dim_] = vdf_range * vth_;
        n_[i] = 1.0;
    }
    vth2 = vth_ * vth_;
    factor = (1.0 / (pow(sqrt(2. * M_PI * vth2), dim_)));
}

double Maxwellian::operator()(const std::vector<double> &v)
{
    double v_sqrt = 0.0;
    for (int i = 0; i < dim_; ++i)
    {
        v_sqrt += (v[i] - vd_[i]) * (v[i] - vd_[i]);
    }
    return factor * exp(-0.5 * v_sqrt / vth2);
}

double Maxwellian::operator()(const std::vector<double> &x, const std::vector<double> &n)
{
    double vn = 0.0;
    for (int i = 0; i < dim(); ++i)
    {
        vn += x[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(x);
}

double Maxwellian::flux_num(const std::vector<double> &n, double S) 
{
    auto vdn = std::inner_product(n.begin(), n.end(), vd_.begin(), 0.0);

    auto num_particles = S * (vth_ / (sqrt(2 * M_PI)) *
                        exp(-0.5 * (vdn / vth_) * (vdn / vth_)) +
                        0.5 * vdn * (1. + erf(vdn / (sqrt(2) * vth_))));
    return num_particles; 
}

void Maxwellian::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double vn, v_sqrt = 0.0;
    for (int i = 0; i < dim_; ++i)
    {
        v_sqrt += (x[i] - vd_[i]) * (x[i] - vd_[i]);
    }
    if (has_flux)
    {
        vn = 0.0;
        for (int i = 0; i < dim_; ++i)
        {
            vn += x[i] * n_[i];
        }
    }else{
        vn = 1.0;
    }
    values[0] = vn * factor * exp(-0.5 * v_sqrt / vth2)*(vn>0);
}

std::vector<double> Maxwellian::cdf(const std::size_t N)
{
    std::mt19937_64 rng(random_seed_seq::get_instance());
    rand_uniform rand(0.0, 1.0);

    double r;
    std::vector<double> vs(N * dim_);
    for (auto j = 0; j < dim_; ++j)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            r = rand(rng);
            vs[i*dim_+j] = vd_[j] - sqrt(2.0)*vth_*boost::math::erfc_inv(2*r);
        }
    }
    return vs;
}

Kappa::Kappa(double vth, std::vector<double> &vd, double k, double vdf_range) 
            : vth_(vth), vd_(vd), k(k)
{
    if (vth_ == 0.0)
    {
        vth_ = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }
    dim_ = vd.size();
    domain_.resize(2 * dim_);
    for (int i = 0; i < dim_; ++i)
    {
        domain_[i] = -vdf_range * vth_;
        domain_[i + dim_] = vdf_range * vth_;
    }
    vth2 = vth_ * vth_;
    factor = (1.0 / pow(sqrt(M_PI * (2 * k - 3.) * vth2), dim_)) *
                ((tgamma(k + 0.5 * (dim_ - 1.0))) / (tgamma(k - 0.5)));
}

double Kappa::operator()(const std::vector<double> &v)
{
    double v_sqrt = 0.0;
    for (int i = 0; i < dim_; ++i)
    {
        v_sqrt += (v[i] - vd_[i]) * (v[i] - vd_[i]);
    }
    return factor * pow(1.0 + v_sqrt / ((2 * k - 3.) * vth2), -(k + 0.5 * (dim_ - 1.)));
}

std::vector<double> rejection_sampler(const std::size_t N,
                                      std::function<double(std::vector<double> &)> pdf,
                                      double pdf_max, int dim,
                                      const std::vector<double> &domain)
{
    rand_uniform rand(0.0, pdf_max);
    rand_uniform_vec rand_vec(dim);

    for (int i = 0; i < dim; ++i)
    {
        rand_vec[i] = rand_uniform(domain[i], domain[i + dim]);
    }

    std::mt19937_64 rng(random_seed_seq::get_instance());

    std::vector<double> xs(N * dim), tmp(dim);
    std::size_t n = 0;
    while (n < N)
    {
        for (int i = 0; i < dim; ++i)
        {
            tmp[i] = rand_vec[i](rng);
        }
        if (rand(rng) < pdf(tmp))
        {
            for (std::size_t i = n * dim; i < (n + 1) * dim; ++i)
            {
                xs[i] = tmp[i % dim];
            }
            n += 1;
        }
    }
    return xs;
}

std::vector<double> random_facet_points(const std::size_t N, 
                                        const std::vector<double> &vertices)
{
    auto size = vertices.size();
    auto g_dim = sqrt(size);
    std::vector<double> xs(N * g_dim);

    std::mt19937_64 rng{random_seed_seq::get_instance()};
    rand_uniform rand(0.0, 1.0);

    double r;
    for (std::size_t i = 0; i < N; ++i)
    {
        r = rand(rng);
        for (int j = 0; j < g_dim; ++j)
        {
            xs[i*g_dim + j] = (1.0 - r) * vertices[j] + r * vertices[j + g_dim];
        }
        for (int j = 1; j < g_dim-1; ++j)
        {
            r = sqrt(rand(rng));
            for (int k = 0; k < g_dim; ++k)
            {
                xs[i*g_dim+k] = r * xs[i*g_dim+k] + (1.0 - r) * vertices[(j + 1) * g_dim + k];
            }
        }
    }
    return xs;
}

void inject_particles(Population &pop, std::vector<Species> &species,
                      std::vector<Facet> &facets, const double dt)
{
    std::mt19937_64 rng{random_seed_seq::get_instance()};
    rand_uniform rand(0.0, 1.0);

    auto g_dim = pop.g_dim;
    auto num_species = species.size();
    auto num_facets = facets.size();
    std::vector<double> xs_tmp(g_dim);

    for (std::size_t i = 0; i < num_species; ++i)
    {
        std::vector<double> xs, vs;
        for (std::size_t j = 0; j < num_facets; ++j)
        {
            auto normal_i = facets[j].normal;
            auto N_float = species[i].n*dt*species[i].vdf.flux_num(normal_i, facets[j].area);
            int N = int(N_float);
            if (rand(rng) < (N_float - N))
            {
                N += 1;
            }
            auto vdf = [i, &normal_i, &species](std::vector<double> &v)->double{
                        return species[i].vdf(v, normal_i);
            };
            auto count = 0;
            while (count <N)
            {
                auto n = N - count;
                auto xs_new = random_facet_points(n, facets[j].vertices);
                auto vs_new = rejection_sampler(n, vdf, species[i].vdf.max(), 
                                                species[i].vdf.dim(), 
                                                species[i].vdf.domain());

                for(auto k=0; k<n; ++k)
                {
                    auto r = rand(rng);
                    for (std::size_t l = 0; l < g_dim; ++l)
                    {
                        xs_tmp[l] = xs_new[k*g_dim + l] + dt*r*vs_new[k*g_dim + l];
                    }
                    if (pop.locate(xs_tmp) >= 0)
                    {
                        for (std::size_t l = 0; l < g_dim; ++l)
                        {
                            xs.push_back(xs_tmp[l]);
                            vs.push_back(vs_new[k * g_dim + l]);
                        }
                    }
                    count += 1;
                }
            }
        }
        pop.add_particles(xs, vs, species[i].q, species[i].m);
    }
}

void load_particles(Population &pop, std::vector<Species> &species)
{
    auto num_species = species.size();
    std::vector<double> xs, vs;
    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto s = species[i];
        auto pdf = [&s](std::vector<double> &x) -> double { return s.pdf(x); };
        auto vdf = [&s](std::vector<double> &v) -> double { return s.vdf(v); };

        xs = rejection_sampler(s.num, pdf, s.pdf.max(), s.pdf.dim(), s.pdf.domain());
        if(s.vdf.has_cdf){
            vs = s.vdf.cdf(s.num);
        }else{
            vs = rejection_sampler(s.num, vdf, s.vdf.max(), s.vdf.dim(), s.vdf.domain());
        }
        pop.add_particles(xs, vs, s.q, s.m);
    }
}

}
