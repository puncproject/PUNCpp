#ifndef CAPACITANCE_H
#define CAPACITANCE_H

#include <iostream>
#include <dolfin.h>
#include "poisson.h"
#include "Flux.h"
#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace punc
{

namespace df = dolfin;

template <class T>
bool inv(const boost::numeric::ublas::matrix<T> &input,
         boost::numeric::ublas::matrix<T> &inverse)
{
    typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;

    // create a working copy of the input
    boost::numeric::ublas::matrix<T> A(input);

    // create a permutation matrix for the LU-factorization
    pmatrix pm(A.size1());

    // perform LU-factorization
    int res = boost::numeric::ublas::lu_factorize(A, pm);
    if (res != 0)
        return false;

    // create identity matrix of "inverse"
    inverse.assign(boost::numeric::ublas::identity_matrix<T>(A.size1()));

    // backsubstitute to get the inverse
    boost::numeric::ublas::lu_substitute(A, pm, inverse);

    return true;
}

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

std::shared_ptr<df::FacetFunction<std::size_t>> markers(
                        std::shared_ptr<const df::Mesh> &mesh,
                        const std::vector<std::shared_ptr<Object>> &objects);


std::vector<std::shared_ptr<df::Function>> solve_laplace(
                const std::shared_ptr<Potential::FunctionSpace> &V,
                const std::shared_ptr<PoissonSolver> &poisson,
                const std::shared_ptr<NonPeriodicBoundary> &non_periodic_bnd,
                const std::vector<std::shared_ptr<Object>> &objects);


boost_matrix capacitance_matrix(
                const std::shared_ptr<Potential::FunctionSpace> &V,
                const std::shared_ptr<PoissonSolver> &poisson,
                const std::shared_ptr<NonPeriodicBoundary> &non_periodic_bnd,
                const std::vector<std::shared_ptr<Object>> &objects);

boost_matrix bias_matrix(const boost_matrix &inv_capacity,
                         const std::map<int, std::vector<int>> &circuits_info);

}

#endif
