#include <iostream>
#include <math.h>
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <chrono>

class Timer
{
  public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

  private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};

struct random_seed_seq
{
    template <typename It>
    void generate(It begin, It end)
    {
        for (; begin != end; ++begin)
        {
            *begin = device();
        }
    }

    static random_seed_seq &get_instance()
    {
        static thread_local random_seed_seq result;
        return result;
    }

  private:
    std::random_device device;
};


template <typename T>
std::vector<double> linspace(const T &start_in, const T &end_in, const int &num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);

    return linspaced;
}


std::vector<std::vector<double>> comb(std::vector<std::vector<double>> vec, double dv)
{
    auto dim = vec.size()/2 + 1;
    auto len = pow(2,dim);
    std::vector<std::vector<double>> arr;
    arr.resize(len, std::vector<double>(dim, 0.0));

    auto plen = int(len/2);

    for (auto i = 0; i < len; ++i)
    {
        for (auto j = 0; j <dim-1; ++j)
        {
            arr[i][j] = vec[i%plen][j];
        }
    }
    for (auto i = 0; i < pow(2, dim-1); ++i)
    {
        arr[i][dim-1] = 0.0;
        arr[pow(2, dim-1)+i][dim-1] = dv;
    }

    return arr;
}

std::function<double(std::vector<double> &)> shifted_maxwellian(double vth, std::vector<double> &vd)
{
    auto dim = vd.size();
    auto pdf = [vth, vd, dim](std::vector<double> &v) {
        double v_sqrt = 0.0;
        for (auto i = 0; i <dim; ++i)
        {
            v_sqrt += (v[i] - vd[i]) * (v[i] - vd[i]);
        }
        return (1.0/(pow(sqrt(2.*M_PI*vth*vth), dim)))*exp(-0.5 * v_sqrt / (vth * vth));
    };

    return pdf;
}

class Bin
{
public:
    int id;
    std::vector<double> vertex;
    int dim;
    Bin(int dim, int id, double x=0.0, double y=0.0, double z = 0.0)
    {
        this->id = id;
        std::vector<double> tmp{x,y,z};
        vertex.resize(dim);
        for (auto i = 0; i < dim; ++i)
        {
            vertex[i] = tmp[i];
        }
    }

    double get_max(std::function<double(std::vector<double> &)> pdf, 
                   std::vector<std::vector<double>> edges)
    {
        double value, max = 0.0;
        auto num_edges = edges.size();
        auto dim = edges[0].size();
        std::vector<double> edge(dim);
        for (auto j = 0; j < num_edges; ++j)
        {
            for (auto k = 0; k < dim; ++k)
            {
                edge[k] = vertex[k] + edges[j][k];

            }
            value = pdf(edge);
            if (value > max)
            {
                max = value;
            }
        }
        return max;
    }
};

int main()
{
    Timer timer;
    
    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, 1.0);

    int num_sp = 40;
    std::vector<std::vector<double>> cutoffs{{-6., 6.}, {-6., 6.},{-6.,6.}};

    double vth = 1.0;
    std::vector<double> vd{0.0, 0.0,0.0};
    std::vector<double> normal{1.0, 0.0,0.0};

    auto pdf = shifted_maxwellian(vth, vd);

    auto vdf_flux = [pdf, normal](std::vector<double> &v) {
        auto vdn = std::inner_product(std::begin(normal), std::end(normal), std::begin(v), 0.0);
        if (vdn >= 0.0)
        {
            return vdn * pdf(v);
        }
        else
        {
            return 0.0;
        }
    };
    timer.reset();
    int dim = vd.size();
    std::vector<double> nsp(3, 1.0), dv(3,0.0), diff(dim);
    
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

    // double x=cutoffs[0][0], y, z;
    // std::vector<Bin> bins;
    int id;
    int length, depth, height;
    length = (int)nsp[0];
    depth  = (int)nsp[1];
    height = (int)nsp[2];
    std::vector<double> nodes, _nodes(3);
    for (auto i = 0; i < 3; ++i)
    {
        _nodes[i] = cutoffs[i][0];
    }
    for (auto i = 0; i < length; ++i)
    {
        // y=cutoffs[1][0];
        _nodes[1] = cutoffs[1][0];
        for (auto j = 0; j < depth; ++j)
        {
            // z=cutoffs[2][0];
            _nodes[2] = cutoffs[2][0];
            for (auto k = 0; k < height; ++k)
            {   
                // z += dv[2];
                _nodes[2] += dv[2];
                // id = (i*height + j)*depth+k;
                // Bin bin(dim, id, x, y, z);
                // bins.push_back(bin);
                for (auto l = 0; l < dim; ++l)
                {
                    nodes.push_back(_nodes[l]);
                }
            }
            // y += dv[1];
            _nodes[1] += dv[1];
        }
        // x += dv[0];
        _nodes[0] += dv[0];
    }
     
    auto t0 = timer.elapsed();
    std::cout << "sp: " << t0 << '\n';
    timer.reset();
    auto volume = std::accumulate(dv.begin(), dv.begin()+dim, 1.0, std::multiplies<double>());

    std::vector<std::vector<double>> points, edges{{0.0}, {dv[0]}};
    for (auto i = 1; i < dim; ++i)
    {
        points = comb(edges, dv[i]);
        edges = points;
    }
    t0 = timer.elapsed();
    std::cout << "comb: " << t0 << '\n';
    timer.reset();
    // auto num_bins = bins.size();
    auto num_bins = nodes.size()/dim;
    std::vector<double> f_max(num_bins), integrand(num_bins);

    double value, max;
    auto num_edges = edges.size();
    std::vector<double> edge(dim);
    double vn, vv;
    double coeff = (1.0/(pow(sqrt(2.*M_PI*vth*vth), dim)));
    for (auto i = 0; i < num_bins; ++i)
    {
        max = 0.0;    
        for (auto j = 0; j < num_edges; ++j)
        {
            vv = 0.0;
            // vn = 0.0;
            for (auto k = 0; k < dim; ++k)
            {
                // edge[k] = nodes[i*dim+k] + edges[j][k];
               vv += (nodes[i*dim+k] + edges[j][k]-vd[k])*(nodes[i*dim+k] + edges[j][k]-vd[k]);
            //    vn += (nodes[i*dim+k] + edges[j][k])*normal[i];
            }
            // value = pdf(edge);
            value = coeff*expf(-0.5 * vv / (vth * vth));
            if (value >= max)
            {
                max = value;
            }
        }
        f_max[i] = max;
        integrand[i] = volume*max;
    }

    // for (auto i = 0; i < num_bins; ++i)
    // {
    //     f_max[i] = bins[i].get_max(vdf_flux, edges);
    //     integrand[i] = volume*f_max[i];
    // }
    t0 = timer.elapsed();
    std::cout << "proposal pdf: " << t0 << '\n';
    timer.reset();
    auto integral = std::accumulate(integrand.begin(), integrand.end(), 0.0);
    t0 = timer.elapsed();
    std::cout << "integral: " << t0 << '\n';
    timer.reset();    
    std::vector<double> w_integral(num_bins), weights(num_bins);
    
    for (auto i = 0; i < num_bins; ++i)
    {
        w_integral[i] = integrand[i] / integral;
    }
    t0 = timer.elapsed();
    std::cout << "pdf: " << t0 << '\n';
    timer.reset();
    std::partial_sum(w_integral.begin(), w_integral.end(), weights.begin());    
    t0 = timer.elapsed();
    std::cout << "weights: " << t0 << '\n';
    timer.reset();

    // width_index=index/(height*depth); 
    // height_index=(index-width_index*height*depth)/depth;
    // depth_index=index-width_index*height*depth- height_index*depth;

    // exit(EXIT_SUCCESS);
    int N = 10000000;
    std::vector<double> vs(N * dim), vs_new(dim);
    int n = 0;
    double p_vs;
    int rej=0;
    while (n < N)
    {
        auto ind = std::distance(weights.begin(), std::lower_bound(weights.begin(), weights.end(), dist(rng)));
        vv = 0.0;
        for (int j = 0; j < dim; j++)
        {
            // vs_new[j] = bins[ind].vertex[j] + dv[j] * dist(rng);
            vs_new[j] = nodes[ind*dim + j] + dv[j] * dist(rng);
            vv += (vs_new[j]) * (vs_new[j]);
        }
        value = coeff * expf(-0.5 * vv / (vth * vth));
        p_vs = f_max[ind] * dist(rng);
        if (p_vs < value)
        {
            for (int i = n * dim; i < (n + 1) * dim; ++i)
            {
                vs[i] = vs_new[i % dim];
            }
            n += 1;
        }else{
            rej +=1;
        }
    }
    t0 = timer.elapsed();
    std::cout << "Sampling time: " << t0 << '\n'; 
    std::cout<<"Number of rejections: "<<rej<<'\n'; 
    std::cout << "Number of samples: " << vs.size() << '\n';
    std::ofstream file;
    file.open("vs.txt");
    for (const auto &e : vs)
        file << e << "\n";
    file.close();
    return 0;
}