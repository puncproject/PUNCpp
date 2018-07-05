#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "population.h"
#include "../ufl/Energy.h"

namespace punc
{

namespace df = dolfin;

double kinetic_energy(Population &pop);

double mesh_potential_energy(df::Function &phi, df::Function &rho);

double particle_potential_energy(Population &pop, const df::Function &phi);
}

#endif
