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

class ORS
{
public:
  std::function<double(std::vector<double> &)> pdf;
  std::vector<std::vector<double>> sp;
  std::vector<double> dv;
  int dim, nsp;

  std::vector<double> pdf_max;
  std::vector<double> cdf;

  typedef std::mt19937_64 random_source;
  typedef std::uniform_real_distribution<double> distribution;
  random_source rng;
  distribution dist;

  ORS(std::function<double(std::vector<double> &)> pdf,
      std::vector<std::vector<double>> &sp,
      std::vector<double> dv, int dim,
      int nsp, int nbins) : pdf(pdf), sp(sp), dv(dv), dim(dim), nsp(nsp),
      dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
  {
      Timer timer;
      timer.reset();

      pdf_max.resize(nbins);
      cdf.resize(nbins);
      auto t0 = timer.elapsed();
      std::cout << "vectors:  " << t0 << '\n';
      timer.reset();
      double max, value;

      for (auto i = 0; i < nbins; ++i)
      {
          max = 0.0;
          for (auto j = 0; j < nsp; ++j)
          {
              value = pdf(sp[i * nsp + j]);
              max = std::max(max, value);
          }
          pdf_max[i] = max;
      }
      t0 = timer.elapsed();
      std::cout << "The loop:  " << t0 << '\n';
      timer.reset();
      auto normalization_factor = std::accumulate(pdf_max.begin(), pdf_max.end(), 0.0);
      std::partial_sum(pdf_max.begin(), pdf_max.end(), cdf.begin());

      std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                     std::bind1st(std::multiplies<double>(), 1./normalization_factor));
      t0 = timer.elapsed();
      std::cout << "cdf:  " << t0 << '\n';
    //   timer.reset();
  }

    std::vector<double> sample(const int N)
    {
        std::vector<double> vs(N * dim), vs_new(dim);
        int index, n = 0;
        double p_vs, value;
        while (n < N)
        {
            index = std::distance(cdf.begin(),
                        std::lower_bound(cdf.begin(), cdf.end(), dist(rng)));

            for (int i = n * dim; i < (n + 1) * dim; ++i)
            {
                vs_new[i%dim] = sp[index*nsp][i%dim] + dv[i % dim] * dist(rng);
                vs[i] = vs_new[i%dim];
            }
            value = pdf(vs_new);
            p_vs = pdf_max[index] * dist(rng);
            n = n + (p_vs<value);
        }
        return vs;
    }
};

int main()
{
    Timer timer;

    int num_sp = 50;
    std::vector<std::vector<double>> cutoffs{{-6., 6.}, {-6., 6.},{-6.,6.}};

    double vth = 1.0;
    std::vector<double> vd{0.0,0.0,0.0};
    std::vector<double> normal{1.0,0.0,0.0};

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
        // std::cout << nsp[i] << "  ";
    }
    // std::cout << '\n';
    for (auto i = 0; i < dim; ++i)
    {
        dv[i] = diff[i] / nsp[i];
        // std::cout << "dv: " << dv[i] << " nsp: " << nsp[i] << " ";
    }
    // std::cout << '\n';
    int nbins = std::accumulate(nsp.begin(), nsp.end(), 1, std::multiplies<int>());
    // std::cout<<"nbins: "<<nbins<<'\n';
    auto volume = std::accumulate(dv.begin(), dv.begin() + dim, 1.0, std::multiplies<double>());
    // std::cout << "volume: "<<volume <<'\n';
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
        for (auto j = 0; j < num_edges; ++j)
        {
            for (auto k = 0; k < dim; ++k)
            {
                vertices[i*num_edges+j].push_back(cutoffs[k][0] + dv[k]*indices[k]+edges[j][k]);
                // std::cout << cutoffs[k][0] + dv[k] * indices[k] + edges[j][k] << " ";
            }
            // std::cout<<'\n';
        }
        // std::cout<<"-------------"<<'\n';
    }
    auto t0 = timer.elapsed();
    std::cout << "Precalculated stuff: " << t0 << '\n';
    timer.reset();

    //--------------------------------------------------------------------------
    // Create proposal pdf
    //--------------------------------------------------------------------------
    ORS ors(vdf_flux, vertices, dv, dim, num_edges, nbins);
    t0 = timer.elapsed();
    std::cout << "Creating cdf: " << t0 << '\n';
    timer.reset();

    //--------------------------------------------------------------------------
    // Sample velocities
    //--------------------------------------------------------------------------
    int N = 10000000;
    auto vs  = ors.sample(N);
    t0 = timer.elapsed();
    std::cout << "Sampling time: " << t0 << '\n';
    std::cout << "Number of samples: " << vs.size() << '\n';

    std::ofstream file;
    file.open("vs.txt");
    for (const auto &e : vs)
        file << e << "\n";
    file.close();
    return 0;
}
