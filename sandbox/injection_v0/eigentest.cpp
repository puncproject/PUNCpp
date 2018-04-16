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
#include <Eigen/Dense>
using Eigen::MatrixXd;


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

std::vector<std::vector<double>> cartesian(std::vector<std::vector<double>> &v)
{
    auto product = [](long long a, std::vector<double> &b) { return a * b.size(); };
    const long long N = std::accumulate(v.begin(), v.end(), 1LL, product);
    std::vector<std::vector<double>> u(N, std::vector<double>(v.size()));
    for (long long n = 0; n < N; ++n)
    {
        std::lldiv_t q{n, 0};
        for (long long i = v.size() - 1; 0 <= i; --i)
        {
            q = std::div(q.quot, v[i].size());
            u[n][i] = v[i][q.rem];
        }
    }
    return u;
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
    auto pdf = [vth, vd, dim](std::vector<double> &v) -> double {
        double v_sqrt = 0.0;
        for (auto i = 0; i <dim; ++i)
        {
            v_sqrt += (v[i] - vd[i]) * (v[i] - vd[i]);
        }
        return (1.0/(pow(sqrt(2.*M_PI*vth*vth), dim)))*exp(-0.5 * v_sqrt / (vth * vth));
    };

    return pdf;
}

int main()
{
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    return 0;
}