#ifndef POPULATION_H
#define POPULATION_H

#include <iostream>
#include <dolfin.h>
#include <algorithm>
#include <math.h>
#include <memory>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <assert.h>
#include <limits>

namespace df = dolfin;
typedef std::uint32_t uint32;

class Maxwellian;

struct Particle
{
    int id;
};

class Population
{
public:

    std::vector<double> xs, vs, Ld;
    std::vector<bool> periodic;
    int dim, tot_num;
    double plasma_density;
    double vth;
    std::vector<double> vd;
    std::vector<Particle> ids;
    std::unique_ptr<Maxwellian> mv;

    Population(int N, std::vector<double> Ld, std::vector<bool> periodic, 
               double vth, std::vector<double> vd);

    void load_particles();
    void add_particles(std::vector<double> &xs_new, std::vector<double> &vs_new);
};

#endif