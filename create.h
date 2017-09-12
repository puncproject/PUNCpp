#ifndef CREATE_H
#define CREATE_H

#include <iostream>
#include <dolfin.h>
#include <algorithm>
#include "punc/object.h"
#include "punc/Potential.h"
#include "punc/capacitance.h"

using namespace punc;

class Source : public df::Expression
{
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
};

std::vector<Object> create_objects(
    const std::shared_ptr<Potential::FunctionSpace> &V,
    const std::vector<std::shared_ptr<ObjectBoundary>> &objects,
    const std::vector<double> &potentials);

std::vector<Object> create_objects(
    const std::shared_ptr<Potential::FunctionSpace> &V,
    const std::vector<std::shared_ptr<ObjectBoundary>> &objects,
    const std::vector<std::shared_ptr<df::Function>> &potentials);

std::vector<std::shared_ptr<ObjectBoundary>> circle_objects();

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

void get_circuit(std::map <int, std::vector<int> > &circuits_info,
                 std::map <int, std::vector<double> > &bias_potential);

std::vector<Circuit> create_circuits(
                          std::vector<Object> &obj,
                          boost_matrix &inv_capacity,
                          std::map <int, std::vector<int> > &circuits_info,
                          std::map <int, std::vector<double> > &bias_potential);

#endif
