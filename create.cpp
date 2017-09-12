#include "create.h"

void Source::eval(df::Array<double> &values, const df::Array<double> &x) const
{
    double dx = x[0] - 3.14;
    double dy = x[1] - 3.14;
    values[0] = 10 * exp(-(dx * dx + dy * dy) / 0.02);
}

std::vector<Object> create_objects(
    const std::shared_ptr<Potential::FunctionSpace> &V,
    const std::vector<std::shared_ptr<ObjectBoundary>> &objects,
    const std::vector<double> &potentials)
{

    std::size_t num_objects = objects.size();
    std::vector<Object> object_vec;
	object_vec.reserve(num_objects);
	for(std::size_t i = 0; i < num_objects; ++i)
	{
		object_vec.emplace_back(Object(V, objects[i], potentials[i]));
    }
    return object_vec;
}

std::vector<Object> create_objects(
    const std::shared_ptr<Potential::FunctionSpace> &V,
    const std::vector<std::shared_ptr<ObjectBoundary>> &objects,
    const std::vector<std::shared_ptr<df::Function>> &potentials)
{
    std::size_t num_objects = objects.size();
    std::vector<Object> object_vec;
	object_vec.reserve(num_objects);
	for(std::size_t i = 0; i < num_objects; ++i)
	{
		object_vec.emplace_back(Object(V, objects[i], potentials[i]));
    }
    return object_vec;
}

std::vector<std::shared_ptr<ObjectBoundary>> circle_objects()
{
    int num_objs = 4;
	double r = 0.5;
	double tol=1e-4;
	std::vector<double> s0{M_PI, M_PI};
	std::vector<double> s1{M_PI, M_PI+3*r};
	std::vector<double> s2{M_PI, M_PI-3*r};
	std::vector<double> s3{M_PI+3*r, M_PI};

	auto func0 = \
	[r,s0,tol](const df::Array<double>& x)->bool
	{
		double dot = 0.0;
		for(std::size_t i = 0; i<s0.size(); ++i)
		{
			dot += (x[i]-s0[i])*(x[i]-s0[i]);
		}
		return dot <= r*r+tol;
	};

	auto func1 = \
	[r,s1,tol](const df::Array<double>& x)->bool
	{
		double dot = 0.0;
		for(std::size_t i = 0; i<s1.size(); ++i)
		{
			dot += (x[i]-s1[i])*(x[i]-s1[i]);
		}
		return dot <= r*r+tol;
	};

	auto func2 = \
	[r,s2,tol](const df::Array<double>& x)->bool
	{
		double dot = 0.0;
		for(std::size_t i = 0; i<s2.size(); ++i)
		{
			dot += (x[i]-s2[i])*(x[i]-s2[i]);
		}
		return dot <= r*r+tol;
	};

	auto func3 = \
	[r,s3,tol](const df::Array<double>& x)->bool
	{
		double dot = 0.0;
		for(std::size_t i = 0; i<s3.size(); ++i)
		{
			dot += (x[i]-s3[i])*(x[i]-s3[i]);
		}
		return dot <= r*r+tol;
	};

	std::vector<std::shared_ptr<ObjectBoundary>> circles(num_objs);

	auto circle0 = std::make_shared<ObjectBoundary>(func0);
	auto circle1 = std::make_shared<ObjectBoundary>(func1);
	auto circle2 = std::make_shared<ObjectBoundary>(func2);
	auto circle3 = std::make_shared<ObjectBoundary>(func3);

    circles = {circle0, circle1, circle2, circle3};
    return circles;
}

void get_circuit(std::map <int, std::vector<int>> &circuits_info,
				 std::map <int, std::vector<double>> &bias_potential)
{
	circuits_info[0] = std::vector<int>({0,2});
	circuits_info[1] = std::vector<int>({1,3});

	bias_potential[0] = std::vector<double>({0.1});
	bias_potential[1] = std::vector<double>({0.2});
}

std::vector<Circuit> create_circuits(
							std::vector<Object> &obj,
							boost_matrix &inv_capacity,
							std::map <int, std::vector<int>> &circuits_info,
							std::map <int, std::vector<double>> &bias_potential)
{

	std::size_t num_circuits = circuits_info.size();

	std::size_t len_bias_potential = 0;
	for(auto const& imap: bias_potential)
	{
		len_bias_potential += imap.second.size();
	}
	boost_vector biases(len_bias_potential);
	auto i = 0;
	for(auto const& imap: bias_potential)
	{
		for(auto &bias_pot: imap.second)
		{
			biases(i) = bias_pot;
			i++;
		}
	}

	boost_matrix inv_bias = bias_matrix(inv_capacity, circuits_info);

	std::size_t num_rows = inv_bias.size1();
	std::size_t num_cols = inv_bias.size2();
	boost_vector bias_0(num_rows, 0.0);

	for(std::size_t i = 0; i < num_rows; ++i)
	{
		for(std::size_t j = 0; j < len_bias_potential; ++j)
		{
			bias_0(i) += inv_bias(i,j)*biases(j);
		}
	}

	std::vector<Circuit> circuits;
	circuits.reserve(num_circuits);
	std::vector<int> circuit;
	for(auto const& c_map: circuits_info)
	{
		circuit = c_map.second;
		std::size_t size_circuit = circuit.size();
		std::vector<Object> circuit_comps;
		circuit_comps.reserve(size_circuit);
		boost_vector bias_0_comp(size_circuit, 0.0);
		boost_matrix inv_bias_matrix(size_circuit, num_cols-len_bias_potential, 0.0);
		int j = 0;
		for(auto &x: c_map.second)
		{
			circuit_comps.emplace_back(obj[x]);
			bias_0_comp[j] = bias_0(x);
			j++;
		}
		for(std::size_t k = 0; k<circuit.size(); ++k)
		{
			int m = 0;
			for(std::size_t l = len_bias_potential; l < num_cols; ++l)
			{
				inv_bias_matrix(k,m) = inv_bias(circuit[k], l);
				m++;
			}
		}
		circuits.emplace_back(Circuit(circuit_comps, bias_0_comp, inv_bias_matrix));
	}
	return circuits;
}
