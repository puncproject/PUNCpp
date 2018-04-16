#ifndef OBJECT_H
#define OBJECT_H

#include <dolfin.h>
#include "../ufl/Potential1D.h"
#include "../ufl/Potential2D.h"
#include "../ufl/Potential3D.h"
#include "../ufl/EField1D.h"
#include "../ufl/EField2D.h"
#include "../ufl/EField3D.h"
#include "../ufl/Flux.h"
#include "../ufl/Charge.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace punc
{

namespace df = dolfin;

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

bool inv(const boost_matrix &input, boost_matrix &inverse);

class Object : public df::DirichletBC
{
public:
    double potential;
    double charge;
    bool floating;
    std::size_t id;
    df::MeshFunction<std::size_t> bnd;

    double interpolated_charge;
    std::vector<std::size_t> dofs;
    std::size_t size_dofs;

    Object(const df::FunctionSpace &V,
           const df::MeshFunction<std::size_t> &boundaries,
           std::size_t bnd_id,
           double potential = 0.0,
           double charge = 0.0,
           bool floating = true,
           std::string method = "topological");

    void get_dofs();
    void add_charge(const double q);
    void set_potential(const double voltage);
    void compute_interpolated_charge(const df::Function &q_rho);
};

void reset_objects(std::vector<Object> &objcets);

void compute_object_potentials(std::vector<Object> &objects,
                               df::Function &E,
                               const boost_matrix &inv_capacity,
                               std::shared_ptr<const df::Mesh> &mesh);

class VObject : public df::DirichletBC
{
  public:
    double potential;
    double charge;
    bool floating;
    std::size_t id;
    df::MeshFunction<std::size_t> bnd;
    std::vector<std::size_t> dofs;
    std::size_t num_dofs;
    std::shared_ptr<df::Form> charge_form;

    VObject(const df::FunctionSpace &V,
            df::MeshFunction<std::size_t> &boundaries,
            std::size_t bnd_id,
            double potential = 0.0,
            double charge = 0.0,
            bool floating = true,
            std::string method = "topological");
    void get_dofs();
    void add_charge(const double &q);
    double calculate_charge(df::Function &phi);
    void set_potential(double voltage);
    void apply(df::GenericVector &b);
    void apply(df::GenericMatrix &A);
};

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

std::vector<std::shared_ptr<df::Function>> solve_laplace(
    std::shared_ptr<df::FunctionSpace> &V,
    std::vector<Object> &objects,
    std::shared_ptr<df::MeshFunction<std::size_t>> &boundaries,
    std::size_t ext_bnd_id);

std::vector<df::Function> solve_laplace(const df::FunctionSpace &V,
                                        std::vector<Object> &objects,
                                        df::MeshFunction<std::size_t> boundaries,
                                        std::size_t ext_bnd_id);

boost_matrix capacitance_matrix(const df::FunctionSpace &V,
                                std::vector<Object> &objects,
                                const df::MeshFunction<std::size_t> &boundaries,
                                std::size_t ext_bnd_id);

boost_matrix bias_matrix(const boost_matrix &inv_capacity,
                         const std::map<int, std::vector<int>> &circuits_info);

}

#endif
