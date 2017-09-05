#ifndef INJECTOR_H
#define INJECTOR_H

#include <iostream>
#include <math.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <assert.h>
#include <limits>

class Particle;
class Population;

double erfinv(const double &P);

double erfcinv(const double &P);

struct random_seed_seq
{
    template<typename It>
    void generate(It begin, It end)
    {
        for (; begin != end; ++begin)
        {
            *begin = device();
        }
    }
 
    static random_seed_seq & get_instance()
    {
        static thread_local random_seed_seq result;
        return result;
    }
 
private:
    std::random_device device;
};

template<typename T>
std::vector<double> linspace(const T &start_in, const T &end_in, const int &num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1) 
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); 

    return linspaced;
}

class RS
{
public:    
    virtual void generate_v(std::vector<double>&){};
    virtual void generate_vs(int&, std::vector<double>&, std::vector<double>&){};  
    virtual std::vector<double> sample(const int&){};  
};

class RSND: public RS
{
private:
    std::vector<double> v;
    double u;
    int i, dim;

public:
    std::function<double (std::vector<double>&)> pdf;
    const double &pdf_max;
    const std::vector<double> &Ld;

    std::vector<std::uniform_real_distribution<double>> dists;  

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;

    RSND(std::function<double (std::vector<double>&)> f, 
         const double &f_max, const std::vector<double>& L);
    
    void generate_v(std::vector<double>& ) override;
    void generate_vs(int&, std::vector<double>&, std::vector<double>&) override;
    std::vector<double> sample(const int&) override;

protected:
      random_source rng;
      distribution dist;
};

class RS1D: public RS
{
private:
    std::vector<double> v;
    int i, n, dim;

public:
    std::function<double (double&)> pdf;
    double pdf_max;
    std::function<double (double&)> transform;
    double lb, ub;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;

    RS1D();

    RS1D(std::function<double (double&)> f, double f_max, 
         std::function<double (double&)> g, double lb=0.0, double ub=1.0);

    void generate_v(std::vector<double>&) override;
    void generate_vs(int&, std::vector<double>&, std::vector<double>&) override;
    std::vector<double> sample(const int &N) override;

protected:
      random_source rng;
      distribution dist;
      distribution dists;
};

class SRS
{
    std::unique_ptr<RS> rs_strategy;
    public:
        SRS(std::function<double (double&)> pdf, double pdf_max, 
            std::function<double (double&)> transform, double lb=0.0, double ub=1.0)
        {
            this->rs_strategy = std::make_unique<RS1D>(pdf, pdf_max, transform, 
                                                       lb, ub);
        };

        SRS(std::function<double (std::vector<double>&)> pdf, double pdf_max, 
            const std::vector<double> &Ld)
        {
            this->rs_strategy = std::make_unique<RSND>(pdf, pdf_max, Ld);
        };

        std::function<std::vector<double> (const int&)> sample = \
        [this](const int &N)->std::vector<double>{
                                                return rs_strategy->sample(N);};
};

class ORS
{
private:
    std::vector<double> u, v, y, dy, z, Yt, integrand, exp_cdf;
    int i, n, nsp;
    double c_i;

public:
    std::function<double (double&)> Vt, dVdt;
    std::vector<double> roots;
    std::function<double (double&)> transform;
    std::vector<double> interval, sp;
    double lb, ub;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;

    ORS();

    ORS(std::function<double (double&)> f, std::function<double (double&)> df, 
        std::vector<double>& roots, std::function<double (double&)> g, 
        std::vector<double> interval={0.0,1.0}, int nsp=50);

    void tangent_intersections();
    void lower_hull();
    void exponentiated_lower_hull();
    std::vector<double> sample_exp(const int &N);
    std::vector<double> sample(const int &N);

protected:
      random_source rng;
      distribution dist;
};

class Maxwellian
{
public:
    std::vector<double> vd, vd_nonzero;
    double vth, r, vd_ratio;
    std::vector<bool> periodic, normal_vec;
    int dim, dim_drift, dim_nonperiodic, dim2, len_cdf;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;

    std::vector<ORS> rs;
    std::vector<std::function<std::vector<double> (int)>> generator, loader;

    Maxwellian();

    Maxwellian(std::vector<double> v_drift, double v_thermal, 
               std::vector<bool> periodic, double vd_ratio=16.);
    void initialize_loading();
    void initialize_injection();
    void initialize_ors();

    double pdf_max_drifting(const double vd_n, const int s);
    std::function<double(double&)> pdf_flux_drifting_transformed(const double vd_n, 
                                                            const int s);
    std::function<double(double&)> pdf_maxwellian(const int index);
    std::function<double(double&)> pdf_flux_nondrifting(const int s);
    std::function<double(double&)> pdf_flux_drifting(const int index, const int s);
    std::function<double(double&)> cdf_inv_nondrifting();
    std::function<double(double&)> cdf_inv_drifting(const int k);
    std::function<double(double&)> cdf_inv_flux_drifting(const int index);    
    std::function<double(double&)> cdf_inv_inward_flux_nondrifting();
    std::function<double(double&)> cdf_inv_backward_flux_nondrifting();

    void inject_nondrifting();
    void inject_drifting();

    std::vector<double> generate(std::function<double(double&)> cdf_inv, const int N);
    std::vector<double> sample(const int &N, int k);
    std::vector<double> load(const int &N);
    std::vector<double> flux_number(const double&, const std::vector<double>&, 
                                    const double&);

protected:
      random_source rng;
      distribution dist;
};

class MoveParticles
{
public:    
    virtual void move(){}; 
};

class MovePeriodic: public MoveParticles
{
public:
    Population& pop;
    std::vector<double> Ld;
    int dim;
    double dt;

    MovePeriodic(Population& pop, double dt);
    void move() override;
};

class MoveNonPeriodic: public MoveParticles
{
public:
    Population &pop;
    std::vector<double> Ld;
    int dim;
    double dt;
    std::vector<int> count;

    MoveNonPeriodic(Population &pop, double dt);
    void move() override;
};

class MoveMixedBnd: public MoveParticles
{
public:
    Population& pop;
    std::vector<double> Ld;
    std::vector<bool> periodic;
    int dim;
    double dt;
    std::vector<int> periodic_indices, nonperiodic_indices;

    MoveMixedBnd(Population& pop, double dt, std::vector<int> periodic_indices, 
                std::vector<int> nonperiodic_indices);
    void move() override;
};

class Move
{
    std::unique_ptr<MoveParticles> mv_strategy;
    public:
        Move(Population& pop, std::vector<bool> periodic, double dt)
        {
            int dim = periodic.size();
            std::vector<int> periodic_indices, nonperiodic_indices;
            for (int i = 0; i < dim; ++i)
            {
                if(periodic[i])
                {
                    periodic_indices.push_back(i);
                }else{
                    nonperiodic_indices.push_back(i);
                }
            }

            if (periodic_indices.size()==dim)
            {
                this->mv_strategy = std::make_unique<MovePeriodic>(pop, dt);
            }else if(nonperiodic_indices.size()==dim){
                this->mv_strategy = std::make_unique<MoveNonPeriodic>(pop, dt);
            }else{
                this->mv_strategy = std::make_unique<MoveMixedBnd>(pop, dt, 
                                         periodic_indices, nonperiodic_indices); 
            }
        }

        std::function<void ()> move = [this]()->void{return mv_strategy->move();};
};

class Injector
{
public:
    Population& pop;
    double dt;
    std::vector<bool> periodic;
    std::vector<double> Ld, L, surface_area, num_particles;
    std::vector<int> index, slices;
    double plasma_density;
    int dim, d;

    std::unique_ptr<Maxwellian> mv;
    std::vector<std::uniform_real_distribution<double>> dists; 

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;

    Injector(Population& pop, double dt);
    
    void inject();
    std::vector<double> sample_positions(int, int);
    void initialize_injection();

protected:
      random_source rng;
      distribution dist;
};

#endif