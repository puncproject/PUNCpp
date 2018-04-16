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

std::function<double(double)> shifted_maxwellian(double vth, double vdn)
{
    double coeff = (1.0 / (sqrt(2. * M_PI * vth * vth)));
    auto pdf = [vth, vdn, coeff](double v) {
        return  coeff * exp(-0.5 * (v-vdn)*(v-vdn) / (vth * vth));
    };

    return pdf;
}

class ORS
{
public:
    std::function<double(double)> pdf;
    double v0, dv; 
    int nsp;

    std::vector<double> pdf_max;
    std::vector<double> cdf;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;

    ORS(std::function<double(double)> pdf,
        double v0, double dv, int nsp) : pdf(pdf), v0(v0), dv(dv), nsp(nsp), 
        dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
    {
        Timer timer;

        timer.reset();
        std::vector<double> pdfv(nsp);
        cdf.resize(nsp-1);
        auto t0 = timer.elapsed();
        std::cout << "vectors:  " << t0 << '\n';
        timer.reset();

        for (auto i = 0; i < nsp; ++i)
        {
            pdfv[i] = pdf(v0 + i*dv);
        }

        std::transform(pdfv.begin(), pdfv.end()-1, pdfv.begin()+1,
        std::back_inserter(pdf_max), [](double a, double b) { return std::max(a, b); });

        t0 = timer.elapsed();
        std::cout << "The loop:  " << t0 << '\n';
        timer.reset();
        auto normalization_factor = std::accumulate(pdf_max.begin(), pdf_max.end(), 0.0);
        std::partial_sum(pdf_max.begin(), pdf_max.end(), cdf.begin());

        std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                        std::bind1st(std::multiplies<double>(), 1./normalization_factor));
        t0 = timer.elapsed();
        std::cout << "cdf:  " << t0 << '\n';
    }

    std::vector<double> sample(const int N)
    {
        std::vector<double> vs(N); 
        int i, index, n = 0;
        double p_vs, value, vs_new;
        while (n < N)
        {
            index = std::distance(cdf.begin(), 
                        std::lower_bound(cdf.begin(), cdf.end(), dist(rng)));

            vs[n] = v0 + dv*(index + dist(rng));
            value = pdf(vs[n]);
            p_vs = pdf_max[index] * dist(rng);
            n = n + (p_vs<value);
        }
        return vs;
    }
};

int main()
{
    Timer timer;

    std::vector<double> cutoffs{-6., 6.};

    double vth = 1.0;
    std::vector<double> vd{0.0};
    std::vector<double> normal{1.0};
    double vdn = vd[0]*normal[0];
    auto pdf = shifted_maxwellian(vth, vdn);

    auto vdf_flux = [pdf, normal](double v) {
        auto vdn = normal[0]*v;
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
    int N = 10000000;
    int dim = 1;
    int nsp = 50;
    double v0 = cutoffs[0];
    double dv = (cutoffs[1]-cutoffs[0])/nsp;    

    //--------------------------------------------------------------------------
    // Create proposal pdf
    //--------------------------------------------------------------------------
    timer.reset();
    ORS ors(vdf_flux, v0, dv, nsp);
    auto t0 = timer.elapsed();
    std::cout << "Creating cdf: " << t0 << '\n';
    timer.reset();

    //--------------------------------------------------------------------------
    // Sample velocities
    //--------------------------------------------------------------------------
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