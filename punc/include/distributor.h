#ifndef DISTRIBUTOR_H
#define DISTRIBUTOR_H

#include <dolfin.h>
#include "population.h"

namespace punc
{

namespace df = dolfin;

std::vector<double> voronoi_volume_approx(const df::FunctionSpace &V);

df::Function distribute(const df::FunctionSpace &V,
                        Population &pop,
                        const std::vector<double> &dv_inv);
}

#endif