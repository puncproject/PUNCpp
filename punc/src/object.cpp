#include "../include/poisson.h"

namespace punc
{

bool inv(const boost_matrix &input, boost_matrix &inverse)
{
    typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;
    boost::numeric::ublas::matrix<double> A(input);

    pmatrix pm(A.size1());

    int res = boost::numeric::ublas::lu_factorize(A, pm);
    if (res != 0)
        return false;

    inverse.assign(boost::numeric::ublas::identity_matrix<double>(A.size1()));

    boost::numeric::ublas::lu_substitute(A, pm, inverse);

    return true;
}

Object::Object(const df::FunctionSpace &V,
               const df::MeshFunction<std::size_t> &boundaries,
               std::size_t bnd_id,
               double potential,
               double charge,
               bool floating,
               std::string method):
               df::DirichletBC(std::make_shared<df::FunctionSpace>(V),
               std::make_shared<df::Constant>(potential),
               std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
               bnd_id, method), potential(potential),
               charge(charge), floating(floating), id(bnd_id)
{
    auto tags = boundaries.values();
    auto size = boundaries.size();
    bnd = boundaries;
    bnd.set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
    	if(tags[i] == id)
    	{
    		bnd.set_value(i, 9999);
    	}
    }
    get_dofs();
}

void Object::get_dofs()
{
    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);

    for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
    {
        dofs.emplace_back(itr->first);
    }
    size_dofs = dofs.size();
}

void Object::add_charge(const double q)
{
    charge += q;
}

void Object::set_potential(const double voltage)
{
    potential = voltage;
    set_value(std::make_shared<df::Constant>(voltage));
}

void Object::compute_interpolated_charge(const df::Function &q_rho)
{
    interpolated_charge = 0.0;
    for (std::size_t i = 0; i < size_dofs; ++i)
    {
        interpolated_charge += q_rho.vector()->getitem(dofs[i]);
    }
}

void reset_objects(std::vector<Object> &objcets)
{
    for (auto& obj: objcets)
    {
        obj.set_potential(0.0);
    }
}

void compute_object_potentials(std::vector<Object> &objects,
                               df::Function &E,
                               const boost_matrix &inv_capacity,
                               std::shared_ptr<const df::Mesh> &mesh)
{
    auto dim = mesh->geometry().dim();
    int num_objects = objects.size();
    std::vector<double> image_charge(num_objects);

    std::shared_ptr<df::Form> flux;
    if (dim == 1)
    {
        flux = std::make_shared<Flux::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        flux = std::make_shared<Flux::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        flux = std::make_shared<Flux::Form_2>(mesh);
    }
    for (unsigned j = 0; j < num_objects; ++j)
    {
        flux->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[j].bnd));
        flux->set_coefficient("w0", std::make_shared<df::Function>(E));
        image_charge[j] = df::assemble(*flux);
    }

    double potential;
    for (auto i = 0; i < num_objects; ++i)
    {
        potential = 0.0;
        for (auto j = 0; j < num_objects; ++j)
        {
            potential += (objects[j].charge -\
                          image_charge[j]) * inv_capacity(i, j);
        }
        objects[i].set_potential(potential);
    }
}

VObject::VObject(const df::FunctionSpace &V,
                 df::MeshFunction<std::size_t> &boundaries,
                 std::size_t bnd_id,
                 double potential,
                 double charge,
                 bool floating,
                 std::string method)
                 : df::DirichletBC(V.sub(0),
                                   std::make_shared<df::Constant>(potential),
                                   std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
                                   bnd_id, method),
                 potential(potential), charge(charge), floating(floating),
                 id(bnd_id)
{
    auto tags = boundaries.values();
    auto size = boundaries.size();
    bnd = boundaries;
    bnd.set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
        if (tags[i] == id)
        {
            bnd.set_value(i, 9999);
        }
    }

    auto mesh = V.mesh();
    auto dim = mesh->geometry().dim();
    if (dim == 1)
    {
        charge_form = std::make_shared<Charge::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        charge_form = std::make_shared<Charge::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        charge_form = std::make_shared<Charge::Form_2>(mesh);
    }
    charge_form->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(bnd));
    get_dofs();
}
void VObject::get_dofs()
{
    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);

    for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
    {
        dofs.emplace_back(itr->first);
    }
    num_dofs = dofs.size();
}

void VObject::add_charge(const double &q)
{
    charge += q;
}

double VObject::calculate_charge(df::Function &phi)
{
    charge_form->set_coefficient("w0", std::make_shared<df::Function>(phi));
    return df::assemble(*charge_form);
}

void VObject::set_potential(double voltage)
{
    this->potential = voltage;
    this->set_value(std::make_shared<df::Constant>(voltage));
}

void VObject::apply(df::GenericVector &b)
{
    df::DirichletBC::apply(b);
}

void VObject::apply(df::GenericMatrix &A)
{
    if (!floating)
    {
        df::DirichletBC::apply(A);
    }
    else
    {
        for (auto i = 1; i < num_dofs; ++i)
        {
            std::vector<double> neighbor_values;
            std::vector<std::size_t> neighbor_ids, surface_neighbors;
            A.getrow(dofs[i], neighbor_ids, neighbor_values);
            std::size_t num_neighbors = neighbor_ids.size();
            std::fill(neighbor_values.begin(), neighbor_values.end(), 0.0);

            std::size_t num_surface_neighbors = 0;
            std::size_t self_index;
            for (auto j = 0; j < num_neighbors; ++j)
            {
                if (std::find(dofs.begin(), dofs.end(), neighbor_ids[j]) != dofs.end())
                {
                    neighbor_values[j] = -1.0;
                    num_surface_neighbors += 1;
                    if (neighbor_ids[j] == dofs[i])
                    {
                        self_index = j;
                    }
                }
            }
            neighbor_values[self_index] = num_surface_neighbors - 1;
            A.setrow(dofs[i], neighbor_ids, neighbor_values);
            A.apply("insert");
        }
    }
}

Circuit::Circuit(std::vector<Object> &objects,
                 const boost_vector &precomputed_charge,
                 const boost_matrix &inv_bias,
                 double charge):
                 objects(objects),
                 precomputed_charge(precomputed_charge),
                 inv_bias(inv_bias), charge(charge) {}

void Circuit::circuit_charge()
{
    double c_charge = 0.0;
    for (auto obj: objects)
    {
        c_charge += obj.charge - obj.interpolated_charge;
    }
    this->charge = c_charge;
}

void Circuit::redistribute_charge(const std::vector<double> &tot_charge)
{
    std::size_t num_objects = objects.size();
    std::size_t num_rows = inv_bias.size1();
    std::size_t num_cols = inv_bias.size2();
    std::vector<double> redistr_charge(num_rows);
    for (std::size_t i = 0; i < num_rows; ++i)
    {
        redistr_charge[i] = 0.0;
        for (std::size_t j = 0; j < num_cols; ++j)
        {
            redistr_charge[i] += inv_bias(i, j) * tot_charge[j];
        }
    }

    for (std::size_t i = 0; i < num_objects; ++i)
    {
        objects[i].charge = precomputed_charge(i) + redistr_charge[i] +\
                            objects[i].interpolated_charge;
    }
}

void redistribute_circuit_charge(std::vector<Circuit> &circuits)
{
    std::size_t num_circuits = circuits.size();
    std::vector<double> tot_charge(num_circuits);
    for (std::size_t i = 0; i < num_circuits; ++i)
    {
        circuits[i].circuit_charge();
        tot_charge[i] = circuits[i].charge;
    }
    for(auto circ: circuits)
    {
        circ.redistribute_charge(tot_charge);
    }
}

std::vector<df::Function> solve_laplace(const df::FunctionSpace &V,
                                        std::vector<Object> &objects,
                                        df::MeshFunction<std::size_t> boundaries,
                                        std::size_t ext_bnd_id)
{
    auto phi_bnd = std::make_shared<df::Constant>(0.0);
    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V),
                       phi_bnd,
                       std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
                       ext_bnd_id);
    std::vector<df::DirichletBC> ext_bc = {bc};
    PoissonSolver poisson(V, ext_bc);
    ESolver esolver(V);
    auto num_objects = objects.size();

    std::vector<df::Function> object_e_field;
    auto shared_V = std::make_shared<df::FunctionSpace>(V);
    for (std::size_t i = 0; i < num_objects; ++i)
    {
        for (std::size_t j = 0; j < num_objects; ++j)
        {
            if (i == j)
            {
                objects[j].set_potential(1.0);
            }
            else
            {
                objects[j].set_potential(0.0);
            }
        }
        df::Function rho(shared_V);
        auto phi = poisson.solve(rho, objects);
        object_e_field.emplace_back(esolver.solve(phi));
    }
    return object_e_field;
}

boost_matrix capacitance_matrix(const df::FunctionSpace &V,
                                std::vector<Object> &objects,
                                const df::MeshFunction<std::size_t> &boundaries,
                                std::size_t ext_bnd_id)
{
    auto mesh = V.mesh();
    auto dim = mesh->geometry().dim();
    auto num_objects = objects.size();
    std::shared_ptr<df::Form> flux;
    if (dim == 1)
    {
        flux = std::make_shared<Flux::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        flux = std::make_shared<Flux::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        flux = std::make_shared<Flux::Form_2>(mesh);
    }
    boost_matrix capacitance(num_objects, num_objects);
    boost_matrix inv_capacity(num_objects, num_objects);
    auto object_e_field = solve_laplace(V, objects, boundaries, ext_bnd_id);
    for (unsigned i = 0; i < num_objects; ++i)
    {
        flux->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[i].bnd));
        for (unsigned j = 0; j < num_objects; ++j)
        {
            flux->set_coefficient("w0", std::make_shared<df::Function>(object_e_field[j]));
            capacitance(i, j) = df::assemble(*flux);
        }
    }
    inv(capacitance, inv_capacity);
    return inv_capacity;
}

boost_matrix bias_matrix(const boost_matrix &inv_capacity,
                         const std::map<int, std::vector<int>> &circuits_info)
{
    std::size_t num_components = inv_capacity.size1();
    std::size_t num_circuits = circuits_info.size();
    boost_matrix bias_matrix(num_components, num_components, 0.0);
    boost_matrix inv_bias(num_components, num_components);

    std::vector<int> circuit;
    int i, s = 0;
    for (auto const &c_map : circuits_info)
    {
        i = c_map.first;
        circuit = c_map.second;
        for (unsigned ii = 0; ii < circuit.size(); ++ii)
        {
            bias_matrix(num_circuits + i, circuit[ii]) = 1.0;
        }
        for (unsigned j = 1; j < circuit.size(); ++j)
        {
            for (unsigned k = 0; k < num_components; ++k)
            {
                bias_matrix(j - 1 + s, k) = inv_capacity(circuit[j], k) -
                                            inv_capacity(circuit[0], k);
            }
        }
        s += circuit.size() - 1;
    }
    inv(bias_matrix, inv_bias);
    return inv_bias;
}

}
