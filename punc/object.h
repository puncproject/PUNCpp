#ifndef OBJECT_H
#define OBJECT_H

#include <iostream>
#include <dolfin.h>
#include "Potential.h"
#include "EField.h"
#include "Flux.h"
#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace punc
{

namespace df = dolfin;

class ObjectBoundary: public df::SubDomain
{
public:
    std::function<bool(const df::Array<double>&)> func;

    ObjectBoundary(std::function<bool(const df::Array<double>&)> func);

    bool inside(const df::Array<double>& x, bool on_boundary) const;
};

class Object : public df::DirichletBC
{
public:
    const std::shared_ptr<Potential::FunctionSpace> &V;
    const std::shared_ptr<ObjectBoundary> &sub_domain;
    double potential;
    double charge;
    std::string method;
    double interpolated_charge;
    std::vector<int> dofs;
    std::size_t size_dofs;

    Object(const std::shared_ptr<Potential::FunctionSpace> &V,
           const std::shared_ptr<ObjectBoundary> &sub_domain,
           double init_uniform_potential = 0.0,
           double init_charge = 0.0,
           std::string method = "topological");

    Object(const std::shared_ptr<Potential::FunctionSpace> &V,
           const std::shared_ptr<ObjectBoundary> &sub_domain,
           const std::shared_ptr<df::Function> &init_potential,
           double init_uniform_potential = 0.0,
           double init_charge = 0.0,
           std::string method = "topological");

    void get_dofs();
    void add_charge(const double &q);
    void set_potential(const double &pot);
    void compute_interpolated_charge(const std::shared_ptr<df::Function> &q_rho);
    std::vector<double> vertices();
    std::vector<std::size_t> cells(const df::FacetFunction<std::size_t> &facet_func,
                                   int &id);
    void mark_facets(df::FacetFunction<std::size_t> &facet_func,
                     int id);
    void mark_cells(df::CellFunction<std::size_t> &cell_func,
                    df::FacetFunction<std::size_t> &facet_func,
                    int id);
};

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

void compute_object_potentials(const std::shared_ptr<df::Function> &q,
                               std::vector<Object> &objects,
                               const boost_matrix &inv_capacity);

class Circuit
{
public:
    std::vector<Object> &objects;
    const boost_vector &precomputed_charge;
    const boost_matrix &inv_bias;
    double charge;

    Circuit(std::vector<Object> &objects,
            const boost_vector &precomputed_charge,
            const boost_matrix &inv_bias,
            double charge = 0.0);

    void circuit_charge();
    void redistribute_charge(const std::vector<double> &tot_charge);
};

void redistribute_circuit_charge(std::vector<Circuit> &circuits);

}

#endif
