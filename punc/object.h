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
    const std::shared_ptr<df::Constant> &init_potential;
    std::string method;
    double charge{0.0};
    double potential_{0.0};
    double interpolated_charge;
    std::vector<int> dofs;

    Object(const std::shared_ptr<Potential::FunctionSpace> &V,
           const std::shared_ptr<ObjectBoundary> &sub_domain,
           const std::shared_ptr<df::Constant> &potential0,
           std::string method = "topological");

    void add_charge(const double &q);
    void set_potential(const double &pot);
    void compute_interpolated_charge(const std::shared_ptr<df::Function> &q_rho);
    std::vector<double> vertices();
    std::vector<std::size_t> cells(const std::shared_ptr<df::FacetFunction<std::size_t> > &facet_func, 
                                   int &id);
    void mark_facets(std::shared_ptr<df::FacetFunction<std::size_t> > &facet_func, 
                     int id);
    void mark_cells(std::shared_ptr<df::CellFunction<std::size_t> > &cell_func,
                    std::shared_ptr<df::FacetFunction<std::size_t> > &facet_func,
                    int id);
};

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

void compute_object_potentials(const std::shared_ptr<df::Function> &q,
                               const std::vector<std::shared_ptr<Object> > &objects,
                               const boost_matrix &inv_capacity);

class Circuit
{
public:
    const std::vector<std::shared_ptr<Object>> &objects;
    const boost_vector &precomputed_charge;
    const boost_matrix &inv_bias;
    double charge;

    Circuit(const std::vector<std::shared_ptr<Object>> &objects,
            const boost_vector &precomputed_charge,
            const boost_matrix &inv_bias,
            double charge = 0.0);

    void circuit_charge();
    void redistribute_charge(const std::vector<double> &tot_charge);
};

void redistribute_circuit_charge(const std::vector<std::shared_ptr<Circuit> > &circuits);

}

#endif
