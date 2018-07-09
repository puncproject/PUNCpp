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

#include "../include/population.h"

namespace punc
{

std::vector<std::vector<double>> combinations(std::vector<std::vector<double>> vec, double dv)
{
    auto dim = vec.size() / 2 + 1;
    auto len = pow(2, dim);
    std::vector<std::vector<double>> arr;
    arr.resize(len, std::vector<double>(dim, 0.0));

    auto plen = int(len / 2);

    for (auto i = 0; i < len; ++i)
    {
        for (auto j = 0; j < dim - 1; ++j)
        {
            arr[i][j] = vec[i % plen][j];
        }
    }
    for (auto i = 0; i < pow(2, dim - 1); ++i)
    {
        arr[i][dim - 1] = 0.0;
        arr[pow(2, dim - 1) + i][dim - 1] = dv;
    }

    return arr;
}

ORS::ORS(double vth, std::vector<double> &vd,
         std::function<double(std::vector<double> &)> vdf, int num_sp):vdf(vdf),
         dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
{

    dim = vd.size();
    dv.resize(dim);
    std::vector<double> nsp(3,1.0), diff(dim);

    nsp[0] = num_sp;
    for (auto i = 0; i < dim; ++i)
    {
        diff[i] = 10.0*vth;
    }
    for (auto i = 1; i < dim; ++i)
    {
        nsp[i] = nsp[i - 1] * diff[i] / diff[i - 1];
    }
    for (auto i = 0; i < dim; ++i)
    {
        dv[i] = diff[i] / nsp[i];
    }

    nbins = std::accumulate(nsp.begin(), nsp.end(), 1, std::multiplies<int>());

    std::vector<std::vector<double>> points, edges{{0.0}, {dv[0]}};
    for (auto i = 1; i < dim; ++i)
    {
        points = combinations(edges, dv[i]);
        edges = points;
    }
    num_edges = edges.size();

    int rows, cols, cells;
    rows = (int)nsp[0];
    cols = (int)nsp[1];
    cells = (int)nsp[2];
    int num_bins = rows*cols*cells;
    std::vector<int> indices(dim);

    sp.resize(num_bins*num_edges);
    for (auto i = 0; i < num_bins; ++i)
    {
        indices[0] = i/(cols*cells);
        indices[1] = (i/cells)%cols;
        indices[2] = i - indices[0] * cols*cells - indices[1] * cells;
        for (auto j = 0; j < num_edges; ++j)
        {
            for (auto k = 0; k < dim; ++k)
            {
                sp[i*num_edges+j].push_back(vd[k]-5.*vth + dv[k]*indices[k]+edges[j][k]);
            }
        }
    }

    pdf_max.resize(nbins);
    double max, value;

    for (auto j = 0; j < nbins; ++j)
    {
        max = 0.0;
        for (auto k = 0; k < num_edges; ++k)
        {
            value = vdf(sp[j * num_edges + k]);
            max = std::max(max, value);
        }
        pdf_max[j] = max;
    }

    auto normalization_factor = std::accumulate(pdf_max.begin(), pdf_max.end(), 0.0);

    std::partial_sum(pdf_max.begin(), pdf_max.end(), std::back_inserter(cdf));

    std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                   std::bind1st(std::multiplies<double>(), 1. / normalization_factor));
}

std::vector<double> ORS::sample(const std::size_t N)
{
    random_source rng(random_seed_seq::get_instance());
    std::vector<double> vs(N*dim), vs_new(dim);
    int index, n = 0;
    double p_vs, value;
    while (n < N)
    {
        index = std::distance(cdf.begin(),
                std::lower_bound(cdf.begin(), cdf.end(), dist(rng)));

        for (int i = n * dim; i < (n + 1) * dim; ++i)
        {
            vs_new[i%dim] = sp[index*num_edges][i%dim] + dv[i % dim] * dist(rng);
            vs[i] = vs_new[i%dim];
        }
        value = vdf(vs_new);
        p_vs = pdf_max[index] * dist(rng);
        n = n + (p_vs<value);
    }
    return vs;
}

GenericFlux::GenericFlux(){}

GenericFlux::GenericFlux(double vth, std::vector<double> &vd,
         const std::vector<std::vector<double>> &cutoffs,
         int num_sp,
         std::vector<Facet> &facets) :
         dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
{
    double vth2 = vth * vth;
    double factor = (1.0 / (sqrt(2. * M_PI * vth2)));

    auto num_facets = facets.size();
    num_particles.resize(num_facets);
    vdf.resize(num_facets);

    dim = vd.size();
    dv.resize(dim);
    std::vector<double> nsp(3,1.0), diff(dim);

    nsp[0] = num_sp;
    for (auto i = 0; i < dim; ++i)
    {
        diff[i] = cutoffs[i][1] - cutoffs[i][0];
    }
    for (auto i = 1; i < dim; ++i)
    {
        nsp[i] = nsp[i - 1] * diff[i] / diff[i - 1];
    }
    for (auto i = 0; i < dim; ++i)
    {
        dv[i] = diff[i] / nsp[i];
    }

    nbins = std::accumulate(nsp.begin(), nsp.end(), 1, std::multiplies<int>());

    std::vector<std::vector<double>> points, edges{{0.0}, {dv[0]}};
    for (auto i = 1; i < dim; ++i)
    {
        points = combinations(edges, dv[i]);
        edges = points;
    }
    num_edges = edges.size();

    int rows, cols, cells;
    rows = (int)nsp[0];
    cols = (int)nsp[1];
    cells = (int)nsp[2];
    int num_bins = rows*cols*cells;
    std::vector<int> indices(dim);

    sp.resize(num_bins*num_edges);
    for (auto i = 0; i < num_bins; ++i)
    {
        indices[0] = i/(cols*cells);
        indices[1] = (i/cells)%cols;
        indices[2] = i - indices[0] * cols*cells - indices[1] * cells;
        for (auto j = 0; j < num_edges; ++j)
        {
            for (auto k = 0; k < dim; ++k)
            {
                sp[i*num_edges+j].push_back(cutoffs[k][0] + dv[k]*indices[k]+edges[j][k]);
            }
        }
    }

    auto vdf_maxwellian = maxwellian_vdf(vth, vd);

    pdf_max.resize(num_facets*nbins);
    vdf.resize(num_facets);
    double vdn, max, value;

    for(auto i=0; i<num_facets; ++i)
    {
        auto n = facets[i].normal;
        vdn = std::inner_product(n.begin(), n.end(), vd.begin(), 0.0);
        num_particles[i] = facets[i].area * (vth / (sqrt(2 * M_PI)) *
                        exp(-0.5 * (vdn / vth) * (vdn / vth)) +
                        0.5 * vdn * (1. + erf(vdn / (sqrt(2) * vth))));

        vdf[i] = [vdf_maxwellian, n](std::vector<double> &v) {
                auto vn = std::inner_product(std::begin(n), std::end(n), std::begin(v), 0.0);
                if(vn>0.0)
                {
                    return vn * vdf_maxwellian(v);
                }else{
                    return 0.0;
                }
                // return (vn > 0.0) * vn * vdf_maxwellian(v);
            };

        for (auto j = 0; j < nbins; ++j)
        {
            max = 0.0;
            for (auto k = 0; k < num_edges; ++k)
            {
                value = vdf[i](sp[j * num_edges + k]);
                max = std::max(max, value);
            }
            pdf_max[i*nbins + j] = max;
        }

        auto normalization_factor = std::accumulate(pdf_max.begin()+i*nbins, pdf_max.begin()+(i+1)*nbins, 0.0);

        std::partial_sum(pdf_max.begin()+i*nbins, pdf_max.begin()+(i+1)*nbins, std::back_inserter(cdf));

        std::transform(cdf.begin()+i*nbins, cdf.begin()+(i+1)*nbins, cdf.begin()+i*nbins,
                       std::bind1st(std::multiplies<double>(), 1. / normalization_factor));
    }
}

std::vector<double> GenericFlux::sample(const std::size_t N, const std::size_t f)
{
    random_source rng(random_seed_seq::get_instance());
    std::vector<double> vs(N*dim), vs_new(dim);
    int index, n = 0;
    double p_vs, value;
    while (n < N)
    {
        index = std::distance(cdf.begin()+f*nbins,
                std::lower_bound(cdf.begin()+f*nbins, cdf.begin()+(f+1)*nbins, dist(rng)));

        for (int i = n * dim; i < (n + 1) * dim; ++i)
        {
            vs_new[i%dim] = sp[index*num_edges][i%dim] + dv[i % dim] * dist(rng);
            vs[i] = vs_new[i%dim];
        }
        value = vdf[f](vs_new);
        p_vs = pdf_max[index+f*nbins] * dist(rng);
        n = n + (p_vs<value);
    }
    return vs;
}

MaxwellianFlux::MaxwellianFlux(double vth, std::vector<double> &vd,
                               std::vector<Facet> &facets)
                               : facets(facets), dist(0.0, 1.0),
                                 rng{random_seed_seq::get_instance()}
{
    if (vth == 0.0)
    {
        vth = std::numeric_limits<double>::epsilon();
    }
    nsp = 60;
    std::vector<double> cutoffs = {*std::max_element(vd.begin(), vd.end()) - 5.0 * vth,
                                   *std::max_element(vd.begin(), vd.end()) + 5.0 * vth};
    double vth2 = vth * vth;
    double factor = (1.0 / (sqrt(2. * M_PI * vth2)));
    auto num_facets = facets.size();
    Flux::num_particles.resize(num_facets);
    vdf.resize(num_facets);
    maxwellian.resize(num_facets);

    dim = vd.size();
    v0 = cutoffs[0];
    dv = (cutoffs[1] - cutoffs[0]) / nsp;
    std::vector<double> vdfv(nsp);

    for (auto i = 0; i < num_facets; ++i)
    {
        auto n = facets[i].normal;
        std::vector<double> vdn(dim);
        vdn[0] = std::inner_product(n.begin(), n.end(), vd.begin(), 0.0);
        for (int j = 1; j < dim; ++j)
        {
            for (int k = 0; k < dim; ++k)
            {
                vdn[j] += facets[i].basis[k * dim + j] * vd[k];
            }
        }

        Flux::num_particles[i] = facets[i].area * (vth / (sqrt(2 * M_PI)) *
                           exp(-0.5 * (vdn[0] / vth) * (vdn[0] / vth)) +
                           0.5 * vdn[0] * (1. + erf(vdn[0] / (sqrt(2) * vth))));

        vdf[i] = [vth2, vdn, factor](double v) {
            return (v>0)*v*factor* exp(-0.5 * (v - vdn[0]) * (v - vdn[0]) / vth2);
        };

        for (auto j = 0; j < nsp; ++j)
        {
            vdfv[j] = vdf[i](v0 + j * dv);
        }

        std::transform(vdfv.begin(), vdfv.end() - 1, vdfv.begin() + 1,
                       std::back_inserter(pdf_max),
                       [](double a, double b) { return std::max(a, b); });

        auto normalization_factor = std::accumulate(pdf_max.begin() + i * (nsp - 1),
                                                    pdf_max.begin() + (i + 1) * (nsp - 1),
                                                    0.0);

        std::partial_sum(pdf_max.begin() + i * (nsp - 1),
                         pdf_max.begin() + (i + 1) * (nsp - 1),
                         std::back_inserter(cdf));

        std::transform(cdf.begin() + i * (nsp - 1),
                       cdf.begin() + (i + 1) * (nsp - 1),
                       cdf.begin() + i * (nsp - 1),
                       std::bind1st(std::multiplies<double>(),
                       1. / normalization_factor));

        maxwellian[i] = [vth, vdn](double v, int k){
            return vdn[k] - sqrt(2.0) * vth * boost::math::erfc_inv(2 * v);
        };
    }
}

std::vector<double> MaxwellianFlux::sample(const std::size_t N, const std::size_t f)
{
    random_source rng(random_seed_seq::get_instance());
    std::vector<double> vs(N * dim), vs_new(dim);
    int index, n = 0;
    double p_vs, value;
    while (n < N)
    {
        index = std::distance(cdf.begin() + f * (nsp - 1),
                std::lower_bound(cdf.begin() + f * (nsp - 1),
                                 cdf.begin() + (f + 1) * (nsp - 1),
                                 dist(rng)));

        vs_new[0] = v0 + dv * (index + dist(rng));
        value = vdf[f](vs_new[0]);
        p_vs = pdf_max[index + f * (nsp - 1)] * dist(rng);
        if (p_vs < value)
        {
            for (int k = 1; k < dim; ++k)
            {
                auto r = dist(rng);
                vs_new[k] = maxwellian[f](r, k);
            }
            for (int i = 0; i < dim; ++i)
            {
                for (int j = 0; j < dim; ++j)
                {
                    vs[n * dim + i] += facets[f].basis[i * dim + j] * vs_new[j];
                }
            }
            n += 1;
        }
    }
    return vs;
}

std::vector<Facet> exterior_boundaries(df::MeshFunction<std::size_t> &boundaries,
                                       std::size_t ext_bnd_id)
{
    auto mesh = boundaries.mesh();
    auto gdim = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    auto values = boundaries.values();
    auto length = boundaries.size();
    int num_facets = 0;
    for (auto i = 0; i < length; ++i)
    {
        if (ext_bnd_id == values[i])
        {
            num_facets += 1;
        }
    }
    std::vector<Facet> ext_facets;

    double area;
    std::vector<double> normal(gdim);
    std::vector<double> vertices(gdim * gdim);
    std::vector<double> basis(gdim * gdim);
    std::vector<double> vertex(gdim);
    double norm;
    int j;
    mesh->init(tdim - 1, tdim);
    df::SubsetIterator facet_iter(boundaries, ext_bnd_id);
    for (; !facet_iter.end(); ++facet_iter)
    {
        df::Cell cell(*mesh, facet_iter->entities(tdim)[0]);
        auto cell_facet = cell.entities(tdim - 1);
        std::size_t num_facets = cell.num_entities(tdim - 1);
        for (auto i = 0; i < num_facets; ++i)
        {
            if (cell_facet[i] == facet_iter->index())
            {
                area = cell.facet_area(i);
                for (auto j = 0; j < gdim; ++j)
                {
                    normal[j] = -1*cell.normal(i, j);
                    basis[j*gdim] = normal[j];
                }
            }
        }
        j = 0;
        for (df::VertexIterator v(*facet_iter); !v.end(); ++v)
        {
            for (auto i = 0; i < gdim; ++i)
            {
                vertices[j * gdim + i] = v->x(i);
            }
            j += 1;
        }
        norm = 0.0;
        for (auto i = 0; i < gdim; ++i)
        {
            vertex[i] = vertices[i] - vertices[gdim + i];
            norm += vertex[i] * vertex[i];
        }
        for (auto i = 0; i < gdim; ++i)
        {
            vertex[i] /= sqrt(norm);
            basis[i * gdim + 1] = vertex[i];
        }

        if (gdim == 3)
        {
            basis[2] = normal[1] * vertex[2] - normal[2] * vertex[1];
            basis[5] = normal[2] * vertex[0] - normal[0] * vertex[2];
            basis[8] = normal[0] * vertex[1] - normal[1] * vertex[0];
        }
        ext_facets.push_back(Facet{area, vertices, normal, basis});
    }
    return ext_facets;
}

signed long int locate(std::shared_ptr<const df::Mesh> mesh, std::vector<double> x)
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

std::function<double(std::vector<double> &)> create_mesh_pdf(std::function<double(std::vector<double> &)> pdf,
                                                             std::shared_ptr<const df::Mesh> mesh)
{
    auto mesh_pdf = [mesh, pdf](std::vector<double> x) -> double {
            return (locate(mesh, x) >= 0)*pdf(x);
        };

    return mesh_pdf;
}

std::vector<double> random_domain_points(
    std::function<double(std::vector<double> &)> pdf,
    double pdf_max, int N,
    std::shared_ptr<const df::Mesh> mesh)
{
    auto mesh_pdf = create_mesh_pdf(pdf, mesh);

    auto D = mesh->geometry().dim();
    auto coordinates = mesh->coordinates();
    int num_vertices = mesh->num_vertices();
    auto Ld_min = *std::min_element(coordinates.begin(), coordinates.end());
    auto Ld_max = *std::max_element(coordinates.begin(), coordinates.end());

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    std::vector<std::uniform_real_distribution<double>> dists(D);

    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, pdf_max);

    for (int i = 0; i < D; ++i)
    {
        dists[i] = distribution(Ld_min, Ld_max);
    }

    std::vector<double> xs(N * D), v(D);
    int n = 0;
    while (n < N)
    {
        for (int i = 0; i < D; ++i)
        {
            v[i] = dists[i](rng);
        }
        if (dist(rng) < mesh_pdf(v))
        {
            for (int i = n * D; i < (n + 1) * D; ++i)
            {
                xs[i] = v[i % D];
            }
            n += 1;
        }
    }
    return xs;
}

std::vector<double> random_facet_points(const int N, std::vector<double> &facet_vertices)
{
    auto size = facet_vertices.size();
    auto dim = sqrt(size);
    std::vector<double> xs(N * dim), v(dim);

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, 1.0);

    for (auto i = 0; i < N; ++i)
    {
        for (auto j = 0; j < dim; ++j)
        {
            xs[i*dim+j] = facet_vertices[j];
        }
        for (auto k = 1; k < dim; ++k)
        {
            auto r = dist(rng);
            if(k==dim-k+1)
            {
                r = 1.0 - sqrt(r);
            }
            for (auto j = 0; j < dim; ++j)
            {
                xs[i * dim + j] += r * (facet_vertices[k*dim+j] - xs[i * dim + j]);
            }
        }
    }
    return xs;
}

std::vector<double> maxwellian(double vth, std::vector<double> vd, const int &N)
{
    auto dim = vd.size();
    if (vth == 0.0)
    {
        vth = std::numeric_limits<double>::epsilon();
    }
    using namespace boost::math;
    typedef std::mt19937_64 random_source;
    random_source rng{random_seed_seq::get_instance()};
    typedef std::uniform_real_distribution<double> distribution;
    distribution random(0.0, 1.0);

    double r;
    auto cdf = [vth, vd](double &v, int i) { return vd[i] - sqrt(2.0) * vth * erfc_inv(2 * v); };
    std::vector<double> vs(N * dim);
    for (auto j = 0; j < dim; ++j)
    {
        for (auto i = 0; i < N; ++i)
        {
            r = random(rng);
            vs[i * dim + j] = cdf(r, j);
        }
    }
    return vs;
}

std::function<double(std::vector<double> &)> maxwellian_vdf(double vth, std::vector<double> &vd)
{
    auto dim = vd.size();
    auto vth2 = vth*vth;
    auto factor =  (1.0 / (pow(sqrt(2. * M_PI * vth2), dim)));
    auto pdf = [vth2, vd, factor, dim](std::vector<double> &v) {
        double v_sqrt = 0.0;
        for (auto i = 0; i < dim; ++i)
        {
            v_sqrt += (v[i] - vd[i]) * (v[i] - vd[i]);
        }
        return factor*exp(-0.5 * v_sqrt / vth2);
    };

    return pdf;
}

void inject_particles(Population &pop, std::vector<Species> &species,
                      std::vector<Facet> &facets, const double dt)
{
    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, 1.0);

    auto dim = pop.gdim;
    auto num_species = species.size();
    auto num_facets = facets.size();
    std::vector<double> xs_tmp(dim);

    for (auto i = 0; i < num_species; ++i)
    {
        std::vector<double> xs, vs;
        for(auto j = 0; j < num_facets; ++j)
        {
            auto normal_i = facets[j].normal;
            int N = int(species[i].n*dt*species[i].flux->num_particles[j]);
            if (dist(rng) < (species[i].n*dt*species[i].flux->num_particles[j]-N))
            {
                N += 1;
            }
            auto count = 0;
            auto outside = 0;
            while (count <N)
            {
                auto n = N - count;
                auto xs_new = random_facet_points(n, facets[j].vertices);
                auto vs_new = species[i].flux->sample(n, j);

                for(auto k=0; k<n; ++k)
                {
                    auto r = dist(rng);
                    for (auto l = 0; l <dim; ++l)
                    {
                        xs_tmp[l] = xs_new[k*dim + l] + dt*r*vs_new[k*dim + l];
                    }
                    if (pop.locate(xs_tmp) >= 0)
                    {
                        for (auto l = 0; l < dim; ++l)
                        {
                            xs.push_back(xs_tmp[l]);
                            vs.push_back(vs_new[k * dim + l]);
                        }
                    }else{
                        outside +=1;
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
    auto dim = pop.gdim;
    auto num_species = species.size();
    for (auto i = 0; i < num_species; ++i)
    {
        auto s = species[i];
        auto xs = random_domain_points(s.pdf, s.pdf_max, s.num, pop.mesh);
        auto vs = maxwellian(s.vth, s.vd, s.num);
        pop.add_particles(xs, vs, s.q, s.m);
    }
}

}
