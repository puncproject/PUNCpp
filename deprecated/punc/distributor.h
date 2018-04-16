#ifndef DISTRIBUTOR_H
#define DISTRIBUTOR_H

#include <dolfin.h>
#include "population.h"

namespace punc
{

namespace df = dolfin;

std::vector<double> voronoi_volume_approx(std::shared_ptr<df::FunctionSpace> &V);

std::shared_ptr<df::Function> distribute(std::shared_ptr<df::FunctionSpace> &V,
                                         Population &pop,
                                         const std::vector<double> &dv_inv);

}

#endif
