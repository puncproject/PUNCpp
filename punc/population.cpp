#include "population.h"
#include "injector.h"

Population::Population(int N, std::vector<double> Ld, 
                       std::vector<bool> periodic, double vth, 
                       std::vector<double> vd): tot_num(N), Ld(Ld), 
                       periodic(periodic), vth(vth), vd(vd), dim(Ld.size())
{

    double volume = 1;
    for (auto & l: Ld)
    {
        volume *= l;
    }
    this->plasma_density = tot_num/volume;

    std::cout <<"-----Population------"<<'\n';
    std::cout << "tot_num: "<<tot_num <<", volume: "<<volume <<", density:  "<<plasma_density <<'\n';

    this->mv = std::make_unique<Maxwellian>(vd, vth, periodic);
    load_particles();

}

void Population::load_particles()
{

    auto pdf = [this](std::vector<double> x)->double{return 1.0;};

    double pdf_max = 1.0;
    SRS rs(pdf, pdf_max, Ld);

    std::vector<double> xs, vs;
    xs = rs.sample(tot_num);
    vs = mv->load(tot_num);

    this->xs = xs;
    this->vs = vs;  

    std::vector<Particle> ids(tot_num);
    for (int i = 0; i < tot_num; ++i)
    {
        Particle pr = {i};
        ids[i] = pr;
    }
    this->ids = ids;
}

void Population::add_particles(std::vector<double>& xs_new, std::vector<double>& vs_new)
{
    int i, j;
    int N = xs_new.size()/dim;
    int len = N*dim;
    int new_tot = tot_num*dim + len;
    std::cout << "N: " << N <<", old tot:  "<<tot_num;
    std::cout <<", new total:   "<<tot_num+N<<'\n';
    std::cout << "Test of xs: " <<xs[0]<<'\n';
    xs.resize(new_tot);
    vs.resize(new_tot);
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < dim; ++j)
        {
            xs[tot_num*dim + i*dim + j] = xs_new[i*dim + j];
            vs[tot_num*dim + i*dim + j] = vs_new[i*dim + j];
        }
    }
    this->tot_num = tot_num + N;
    ids.resize(tot_num);
    for (i = (tot_num-N); i < tot_num; ++i)
    {
        Particle pr = {i};
        ids[i] = pr;
    }
    this->ids = ids;
}