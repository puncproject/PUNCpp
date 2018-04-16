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

std::function<double(std::vector<double> )> shifted_maxwellian2(double vth, std::vector<double> &vd)
{
    auto dim = vd.size();
    auto pdf = [](std::vector<double> v) {
        double v_sqrt = 0.0;
        for (auto it = v.begin(); it < v.end(); ++it)
        {
            // auto i = it - v.begin();
            v_sqrt += *it;
        }
        return v_sqrt;
    };

    return pdf;
}

int main()
{
    Timer timer;
    
    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, 1.0);

    int num_sp = 6;
    std::vector<std::vector<double>> cutoffs{{-6., 6.}, {-6., 6.},{0.,0.}};

    double vth = 1.0;
    std::vector<double> vd{0.0,0.0};
    std::vector<double> normal{1.0,0.0};

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
    //--------------------------------------------------------------------------
    // Precalculated stuff
    //--------------------------------------------------------------------------
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
        std::cout << nsp[i] << "  ";
    }
    std::cout << '\n';
    for (auto i = 0; i < dim; ++i)
    {
        dv[i] = diff[i] / nsp[i];
        std::cout << "dv: " << dv[i] << " nsp: " << nsp[i] << " ";
    }
    std::cout << '\n';
    auto volume = std::accumulate(dv.begin(), dv.begin() + dim, 1.0, std::multiplies<double>());
    std::cout << "volume: "<<volume <<'\n';
    std::vector<std::vector<double>> points, edges{{0.0}, {dv[0]}};
    for (auto i = 1; i < dim; ++i)
    {
        points = comb(edges, dv[i]);
        edges = points;
    }
    auto num_edges = edges.size();

    int rows, cols, cells;
    rows = (int)nsp[0];
    cols = (int)nsp[1];
    cells = (int)nsp[2];
    int num_bins = rows*cols*cells;
    std::vector<int> indices(dim);
    std::vector<std::vector<double>> vertices;
    vertices.resize(num_bins*num_edges);
    for (auto i = 0; i < num_bins; ++i)
    {
        indices[0] = i/(cols*cells);
        indices[1] = (i/cells)%cols;
        indices[2] = i - indices[0] * cols*cells - indices[1] * cells;
        for (auto k = 0; k < dim; ++k)
        {
            vertices[i].push_back(cutoffs[k][0] + dv[k]*indices[k]);
            std::cout << cutoffs[k][0] + dv[k] * indices[k] << " ";
        }
        std::cout << '\n';
    }

    auto pdf2 = shifted_maxwellian2(vth, vd);
    std::vector<double> f_max(num_bins);
    // std::transform(vertices.begin(), vertices.end(), f_max.begin(), [](std::vector<double> v) 
    // {   double s = 0.0;
    //     for (auto i = v.begin(); i <v.end();++i){
    //         s += (*i)*(*i);
    //     }
    //     return s;});
    std::transform(vertices.begin(), vertices.end(), f_max.begin(), pdf2);
    for (auto i = 0; i < num_bins; ++i)
    {
        std::cout << f_max[i] << " ";
        // std::cout << '\n';
    }
    

    exit(EXIT_FAILURE);
    // auto t0 = timer.elapsed();
    // std::cout << "Precalculated stuff: " << t0 << '\n';
    // timer.reset();
    // //--------------------------------------------------------------------------
    // // Create proposal pdf
    // //--------------------------------------------------------------------------

    // std::vector<double> vert(dim);
    // double max, vv, vn, value;
    // double coeff = (1.0 / (pow(sqrt(2. * M_PI * vth * vth), dim)));
    // std::vector<double> f_max(num_bins), integrand(num_bins);
    
    // for (auto i = 0; i < num_bins; ++i)
    // {
    //     max = 0.0;
    //     for (auto j = 0; j < num_edges; ++j)
    //     {
    //         vv = 0.0;
    //         vn = 0.0;
    //         for (auto k = 0; k < dim; ++k)
    //         {
    //             vv += (vertices[i * num_edges + j][k] - vd[k]) * (vertices[i * num_edges + j][k] - vd[k]);
    //             vn += vertices[i * num_edges + j][k]*normal[k];
    //         }
    //         value = (vn > 0) * vn * coeff * expf(-0.5 * vv / (vth * vth));
    //         max = std::max(max, value);
    //     }
    //     f_max[i] = max;
    //     integrand[i] = volume*max;
    // }
    // t0 = timer.elapsed();
    // std::cout << "pdf: " << t0 << '\n';
    // timer.reset();

    // //--------------------------------------------------------------------------
    // // Create the cdf
    // //--------------------------------------------------------------------------
    // auto integral = std::accumulate(integrand.begin(), integrand.end(), 0.0);

    // std::vector<double> w_integral(num_bins), weights(num_bins);
    
    // for (auto i = 0; i < num_bins; ++i)
    // {
    //     w_integral[i] = integrand[i] / integral;
    // }

    // std::partial_sum(w_integral.begin(), w_integral.end(), weights.begin());    
    // t0 = timer.elapsed();
    // std::cout << "CDF: " << t0 << '\n';
    // timer.reset();

    // //--------------------------------------------------------------------------
    // // Sample velocities
    // //--------------------------------------------------------------------------

    // int N = 10000000;
    // std::vector<double> vs(N * dim), vs_new(dim);
    
    // int n = 0;
    // double p_vs;
    // int rej=0;
    // while (n < N)
    // {
    //     auto ind = std::distance(weights.begin(), std::lower_bound(weights.begin(), weights.end(), dist(rng)));
    //     vv = 0.0;
    //     vn = 0.0;
    //     indices[0] = ind / (cols * cells); 
    //     indices[1] = (ind / cells) % cols; 
    //     indices[2] = ind - indices[0] * cols * cells - indices[1] * cells;
    //     for (int j = 0; j < dim; j++)
    //     {
    //         vs_new[j] = cutoffs[j][0] + dv[j] * (dist(rng) + indices[j]);
    //         // vs_new[j] = vertices[ind/num_edges][j] + dv[j]*dist(rng);
    //         // vv += (vs_new[j]) * (vs_new[j]);
    //         // vn += vs_new[j]*normal[j];
    //     }
    //     // value = (vn > 0) * vn * coeff * expf(-0.5 * vv / (vth * vth));
    //     value = vdf_flux(vs_new);
    //     p_vs = f_max[ind] * dist(rng);
    //     if (p_vs < value)
    //     {
    //         for (int i = n * dim; i < (n + 1) * dim; ++i)
    //         {
    //             vs[i] = vs_new[i % dim];
    //         }
    //         n += 1;
    //     }else{
    //         rej +=1;
    //     }
    // }
    // t0 = timer.elapsed();
    // std::cout << "Sampling time: " << t0 << '\n'; 
    // std::cout<<"Number of rejections: "<<rej<<'\n'; 
    // std::cout << "Number of samples: " << vs.size() << '\n';
    // std::ofstream file;
    // file.open("vs.txt");
    // for (const auto &e : vs)
    //     file << e << "\n";
    // file.close();
    return 0;
}