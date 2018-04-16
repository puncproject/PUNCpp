#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include <dolfin.h>
#include "population.h"
#include "Energy.h"

namespace punc
{

namespace df = dolfin;

double kinetic_energy(Population &pop);

double mesh_potential_energy(std::shared_ptr<df::Function> &phi,
                             std::shared_ptr<df::Function> &rho);

double particle_potential_energy(Population &pop,
                                 std::shared_ptr<df::Function> &phi);
}

#endif
