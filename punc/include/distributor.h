#ifndef DISTRIBUTOR_H
#define DISTRIBUTOR_H

//#include <dolfin.h>
#include "population.h"

namespace punc
{

namespace df = dolfin;

std::vector<double> element_volume(const df::FunctionSpace &V, bool voronoi = false);

df::Function distribute(const df::FunctionSpace &V,
                        Population &pop,
                        const std::vector<double> &dv_inv);

df::Function distribute_dg0(const df::FunctionSpace &Q, Population &pop);
}

#endif