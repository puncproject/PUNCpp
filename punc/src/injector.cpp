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
#include "../ufl/Number.h"

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

Maxwellian::Maxwellian(double vth, std::vector<double> &vd, bool has_cdf,
                       bool has_flux_num, bool has_flux_max, double vdf_range) 
                       : _vth(vth), _vd(vd), _dim(vd.size()), _has_cdf(has_cdf),
                       _has_flux_number(has_flux_num), _has_flux_max(has_flux_max)
{
    if (_vth == 0.0)
    {
        _vth = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }
    _domain.resize(2 * _dim);
    _n.resize(_dim);
    for (int i = 0; i < _dim; ++i)
    {
        _domain[i]        = _vd[i] - vdf_range * _vth;
        _domain[i + _dim] = _vd[i] + vdf_range * _vth;
        _n[i] = 1.0;
    }
    vth2 = _vth * _vth;
    factor = (1.0 / (pow(sqrt(2. * M_PI * vth2), _dim)));
}

double Maxwellian::operator()(const std::vector<double> &v)
{
    double v_sqrt = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v_sqrt += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    return factor * exp(-0.5 * v_sqrt / vth2);
}

double Maxwellian::operator()(const std::vector<double> &v, const std::vector<double> &n)
{
    double vn = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        vn += v[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(v);
}

std::vector<double> Maxwellian::cdf(const std::size_t N)
{
    std::mt19937_64 rng(random_seed_seq::get_instance());
    rand_uniform rand(0.0, 1.0);

    double r;
    std::vector<double> vs(N * _dim);
    for (auto j = 0; j < _dim; ++j)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            r = rand(rng);
            vs[i * _dim + j] = _vd[j] - sqrt(2.0) * _vth * boost::math::erfc_inv(2 * r);
        }
    }
    return vs;
}

void Maxwellian::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double vn, v_sqrt = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v_sqrt += (x[i] - _vd[i]) * (x[i] - _vd[i]);
    }
    if (has_flux)
    {
        vn = 0.0;
        for (int i = 0; i < _dim; ++i)
        {
            vn += x[i] * _n[i];
        }
    }else{
        vn = 1.0;
    }
    values[0] = vn * factor * exp(-0.5 * v_sqrt / vth2)*(vn>0);
}

double Maxwellian::flux_num_particles(const std::vector<double> &n, double S)
{
    auto vdn = std::inner_product(n.begin(), n.end(), _vd.begin(), 0.0);

    auto num_particles = S * (_vth / (sqrt(2 * M_PI)) *
                                  exp(-0.5 * (vdn / _vth) * (vdn / _vth)) +
                              0.5 * vdn * (1. + erf(vdn / (sqrt(2) * _vth))));
    return num_particles;
}

double Maxwellian::flux_max(std::vector<double> &n)
{
    auto vdn = std::inner_product(n.begin(), n.end(), _vd.begin(), 0.0);

    std::vector<double> tmp(_dim);
    for (auto i = 0; i < _dim; ++i)
    {
        tmp[i] = _vd[i] - 0.5 * n[i] * vdn + 0.5 * n[i] * sqrt(4 * vth2 + vdn * vdn);
    }
    return this->operator()(tmp, n);
}

Kappa::Kappa(double vth, std::vector<double> &vd, double k, bool has_cdf,
             bool has_flux_num, bool has_flux_max, double vdf_range)
    : _vth(vth), _vd(vd), k(k), _dim(vd.size()), _has_cdf(has_cdf),
      _has_flux_number(has_flux_num), _has_flux_max(has_flux_max)
{
    assert(k > 1.5 && "kappa must be bigger than 3/2");
    if (_vth == 0.0)
    {
        _vth = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }

    auto sum_vd = std::accumulate(vd.begin(), vd.end(), 0);
    if (sum_vd == 0)
    {
        _has_flux_number = true;
    }
    else
    {
        _has_flux_number = false;
    }

    _domain.resize(2 * _dim);
    for (int i = 0; i < _dim; ++i)
    {
        _domain[i]        = _vd[i] - vdf_range * _vth;
        _domain[i + _dim] = _vd[i] + vdf_range * _vth;
    }
    vth2 = _vth * _vth;
    factor = (1.0 / pow(sqrt(M_PI * (2 * k - 3.) * vth2), _dim)) *
             ((tgamma(k + 0.5 * (_dim - 1.0))) / (tgamma(k - 0.5)));
}

double Kappa::operator()(const std::vector<double> &v)
{
    double v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    return factor * pow(1.0 + v2 / ((2 * k - 3.) * vth2), -(k + 0.5 * (_dim - 1.)));
}

double Kappa::operator()(const std::vector<double> &v, const std::vector<double> &n)
{
    double vn = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        vn += v[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(v);
}

void Kappa::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double vn, v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (x[i] - _vd[i]) * (x[i] - _vd[i]);
    }
    if (has_flux)
    {
        vn = 0.0;
        for (int i = 0; i < _dim; ++i)
        {
            vn += x[i] * _n[i];
        }
    }
    else
    {
        vn = 1.0;
    }
    values[0] = vn * (vn > 0) * factor *
                pow(1.0 + v2 / ((2 * k - 3.) * vth2), -(k + 0.5 * (_dim - 1.)));
}

/* Number of particles for the case without any drift. */
double Kappa::flux_num_particles(const std::vector<double> &n, double S)
{
    auto num_particles = S * ((_vth / (sqrt(2 * M_PI))) * pow(k - 1.5, 0.5) * 
                              (tgamma(k - 1.0) / tgamma(k - 0.5)));
    return num_particles;
}

double Kappa::flux_max(std::vector<double> &n)
{
    auto vdn = std::inner_product(n.begin(), n.end(), _vd.begin(), 0.0);

    std::vector<double> tmp(_dim);
    auto q = k + 0.5 * (_dim - 1.);
    auto qm = 1.0-2.0*q;
    auto q2 = q*q;
    auto k2 = 2.0*k-3.0;
    for (auto i = 0; i < _dim; ++i)
    {
        tmp[i] = _vd[i] + (q / qm) * n[i] * vdn - (n[i] / qm) * 
                          sqrt(q2 * vdn * vdn - qm * k2 * vth2);
    }
    return this->operator()(tmp, n);
}

Cairns::Cairns(double vth, std::vector<double> &vd, double alpha, bool has_cdf, 
               bool has_flux_num, bool has_flux_max, double vdf_range)
    : _vth(vth), _vd(vd), alpha(alpha), _dim(vd.size()), _has_cdf(has_cdf),
      _has_flux_number(has_flux_num), _has_flux_max(has_flux_max)
{
    if (_vth == 0.0)
    {
        _vth = std::numeric_limits<double>::epsilon();
        vdf_range = 1.0;
    }
    _dim = vd.size();
    _domain.resize(2 * _dim);
    for (int i = 0; i < _dim; ++i)
    {
        _domain[i] = -vdf_range * _vth;
        _domain[i + _dim] = vdf_range * _vth;
    }
    vth2 = _vth * _vth;
    factor = (1.0 / (pow(sqrt(2 * M_PI * vth2), _dim) * (1 + _dim * (_dim + 2) * alpha)));
}

double Cairns::operator()(const std::vector<double> &v)
{
    double v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (v[i] - _vd[i]) * (v[i] - _vd[i]);
    }
    double v4 = v2 * v2;
    return factor * (1 + alpha * v4 / pow(vth2, 2)) * exp(-0.5 * v2 / vth2);
}

double Cairns::operator()(const std::vector<double> &v, const std::vector<double> &n)
{
    double vn = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        vn += v[i] * n[i];
    }
    return (vn > 0.0) * vn * this->operator()(v);
}

double Cairns::max()
{
    if (alpha < 0.25)
    {
        return factor;
    }
    else
    {
        std::vector<double> v_max(_dim);
        v_max = _vd;
        v_max[0] += _vth * sqrt(2.0 + sqrt(4.0 - 1.0 / alpha));
        double max = (*this)(v_max);
        max = factor > max ? factor : max;
        return max;
    }
}

void Cairns::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double v2 = 0.0;
    for (int i = 0; i < _dim; ++i)
    {
        v2 += (x[i] - _vd[i]) * (x[i] - _vd[i]);
    }
    double v4 = v2 * v2;

    double vn;
    if (has_flux)
    {
        vn = 0.0;
        for (int i = 0; i < _dim; ++i)
        {
            vn += x[i] * _n[i];
        }
    }
    else
    {
        vn = 1.0;
    }
    values[0] = vn * (vn > 0) * factor *
                (1 + alpha * v4 / pow(vth2, 2)) * exp(-0.5 * v2 / vth2);
}

double Cairns::flux_num_particles(const std::vector<double> &n, double S)
{
    auto vdn = std::inner_product(n.begin(), n.end(), _vd.begin(), 0.0);

    auto num_particles = S * ((_vth / (sqrt(2 * M_PI))) *
                                  exp(-0.5 * (vdn / _vth) * (vdn / _vth)) *
                                  (1 + (_dim + 1) * (_dim + 3) * alpha +
                                   (vdn / _vth) * (vdn / _vth) * alpha) /
                                  (1 + _dim * (_dim + 2) * alpha) +
                              0.5 * vdn * (1. + erf(vdn / (sqrt(2) * _vth))));
    return num_particles;
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

void create_flux_FEM(std::vector<Species> &species, std::vector<Facet> &facets)
{
    auto num_species = species.size();
    auto num_facets = facets.size();
    std::vector<int> nsp = {60, 60, 60};

    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto dim = species[i].vdf.dim();
        auto domain = species[i].vdf.domain();
        df::Point p0, p1;
        for (int i = 0; i < dim; ++i)
        {
            p0[i] = domain[i];
            p1[i] = domain[i + dim];
        }

        std::shared_ptr<const df::Mesh> mesh;
        if (dim == 1)
        {
            df::IntervalMesh interval(nsp[0], domain[0], domain[1]);
            mesh = std::make_shared<const df::Mesh>(interval);
        }
        else if (dim == 2)
        {
            df::RectangleMesh rectangle(p0, p1, nsp[0], nsp[1]);
            mesh = std::make_shared<const df::Mesh>(rectangle);
        }
        else if (dim == 3)
        {
            df::BoxMesh box(p0, p1, nsp[0], nsp[1], nsp[2]);
            mesh = std::make_shared<const df::Mesh>(box);
        }

        auto V = Number::CoefficientSpace_w0(mesh);
        for (std::size_t j = 0; j < num_facets; ++j)
        {
            species[i].vdf.set_flux_normal(facets[j].normal);
            
            df::Function vdf_func(std::make_shared<df::FunctionSpace>(V));
            vdf_func.interpolate(species[i].vdf);
            auto vdf_vector = vdf_func.vector();
            
            if (species[i].vdf.has_flux_number())
            {
                auto num = species[i].vdf.flux_num_particles(facets[j].normal, facets[j].area);
                species[i].vdf.num_particles.push_back(num);
            }
            else
            {
                std::shared_ptr<df::Form> form;
                auto vdf_func_ptr = std::make_shared<df::Function>(vdf_func);
                if (dim == 1)
                {
                    form = std::make_shared<Number::Form_0>(mesh, vdf_func_ptr);
                }
                else if (dim == 2)
                {
                    form = std::make_shared<Number::Form_1>(mesh, vdf_func_ptr);
                }
                else if (dim == 3)
                {
                    form = std::make_shared<Number::Form_2>(mesh, vdf_func_ptr);
                }
                species[i].vdf.num_particles.push_back(df::assemble(*form));
            }

            if (species[i].vdf.has_flux_max())
            {
                species[i].vdf.pdf_max.push_back(species[i].vdf.flux_max(facets[j].normal));
            }
            else
            {
                species[i].vdf.pdf_max.push_back(vdf_vector->max());
            }
        }
    }
}

void create_flux(std::vector<Species> &species, std::vector<Facet> &facets)
{
    rand_uniform rand(0.0, 1.0);
    std::mt19937_64 rng(random_seed_seq::get_instance());

    auto num_species = species.size();
    auto num_facets = facets.size();

    double pdf_x, volume, sum, max;
    int n_iter = 500000;
    
    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto dim = species[i].vdf.dim();
        auto domain = species[i].vdf.domain();
        volume = 1.0;
        for (int k = 0; k < dim; k++)
        {
            volume *= domain[k + dim] - domain[k];
        }

        for (std::size_t j = 0; j < num_facets; ++j)
        {
            if (species[i].vdf.has_flux_number())
            {
                auto num = species[i].vdf.flux_num_particles(facets[j].normal, facets[j].area);
                species[i].vdf.num_particles.push_back(num);
            }
            else
            {
                sum = 0; 
                std::vector<double> x(dim);
                for (int n = 0; n < n_iter; ++n)
                {
                    for (int k = 0; k < dim; ++k)
                    {
                        x[k] = domain[k] + rand(rng) * (domain[k + dim] - domain[k]);
                    }

                    sum += species[i].vdf(x, facets[j].normal);
                }
                auto num = facets[j].area*sum*volume/n_iter;
                species[i].vdf.num_particles.push_back(num);
            }

            if (species[i].vdf.has_flux_max())
            {
                species[i].vdf.pdf_max.push_back(species[i].vdf.flux_max(facets[j].normal));
            }
            else
            {
                max = 0;
                std::vector<double> x(dim);
                for (int n = 0; n < n_iter; ++n)
                {
                    for (int k = 0; k < dim; ++k)
                    {
                        x[k] = domain[k] + rand(rng) * (domain[k + dim] - domain[k]);
                    }

                    pdf_x = species[i].vdf(x, facets[j].normal);
                    max = max > pdf_x ? max : pdf_x;
                }
                species[i].vdf.pdf_max.push_back(max*1.01);
            }
        }
    }
}

// void inject_particles(Population &pop, std::vector<Species> &species,
//                       std::vector<Facet> &facets, const double dt)
// {
//     std::mt19937_64 rng{random_seed_seq::get_instance()};
//     rand_uniform rand(0.0, 1.0);

//     auto g_dim = pop.g_dim;
//     auto num_species = species.size();
//     auto num_facets = facets.size();
//     std::vector<double> xs_tmp(g_dim);

//     for (std::size_t i = 0; i < num_species; ++i)
//     {
//         std::vector<double> xs, vs;
//         for (std::size_t j = 0; j < num_facets; ++j)
//         {
//             auto normal_i = facets[j].normal;
//             auto N_float = species[i].n*dt*species[i].vdf.flux_num(normal_i, facets[j].area);
//             int N = int(N_float);
//             if (rand(rng) < (N_float - N))
//             {
//                 N += 1;
//             }
//             auto vdf = [i, &normal_i, &species](std::vector<double> &v)->double{
//                         return species[i].vdf(v, normal_i);
//             };
//             auto count = 0;
//             while (count <N)
//             {
//                 auto n = N - count;
//                 auto xs_new = random_facet_points(n, facets[j].vertices);
//                 auto vs_new = rejection_sampler(n, vdf, species[i].vdf.max(), 
//                                                 species[i].vdf.dim(), 
//                                                 species[i].vdf.domain());

//                 for(auto k=0; k<n; ++k)
//                 {
//                     auto r = rand(rng);
//                     for (std::size_t l = 0; l < g_dim; ++l)
//                     {
//                         xs_tmp[l] = xs_new[k*g_dim + l] + dt*r*vs_new[k*g_dim + l];
//                     }
//                     if (pop.locate(xs_tmp.data()) >= 0)
//                     {
//                         for (std::size_t l = 0; l < g_dim; ++l)
//                         {
//                             xs.push_back(xs_tmp[l]);
//                             vs.push_back(vs_new[k * g_dim + l]);
//                         }
//                     }
//                     count += 1;
//                 }
//             }
//         }
//         pop.add_particles(xs, vs, species[i].q, species[i].m);
//     }
// }

// void load_particles(Population &pop, std::vector<Species> &species)
// {
//     auto num_species = species.size();
//     std::vector<double> xs, vs;
//     for (std::size_t i = 0; i < num_species; ++i)
//     {
//         auto s = species[i];
//         auto pdf = [&s](std::vector<double> &x) -> double { return s.pdf(x); };
//         auto vdf = [&s](std::vector<double> &v) -> double { return s.vdf(v); };

//         xs = rejection_sampler(s.num, pdf, s.pdf.max(), s.pdf.dim(), s.pdf.domain());
//         if(s.vdf.has_cdf){
//             vs = s.vdf.cdf(s.num);
//         }else{
//             vs = rejection_sampler(s.num, vdf, s.vdf.max(), s.vdf.dim(), s.vdf.domain());
//         }
//         pop.add_particles(xs, vs, s.q, s.m);
//     }
// }

}
