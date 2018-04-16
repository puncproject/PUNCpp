#include <iostream>
#include <dolfin.h>
#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "../../punc/punc.h"

using namespace punc;

class Phi : public df::Expression
{
  public:
	double phi_1 = 1.0, phi_2 = 0.0;
	double r_1 = 0.02, r_2 = 0.2;
	void eval(df::Array<double> &values, const df::Array<double> &x) const
	{
		values[0] =  ((phi_1-phi_2)*r_1*r_2/( (r_2-r_1) * pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], 0.5) )) + phi_1-r_2*(phi_1-phi_2)/(r_2-r_1);
	}
};

// class EField : public df::Expression
// {
//   public:
// 	double phi_1 = 1.0, phi_2 = 0.0;
// 	double r_1 = 0.02, r_2 = 0.2;
// 	void eval(df::Array<double> &values, const df::Array<double> &x) const
// 	{
// 		values[0] = (phi_1-phi_2)*r_1*r_2*x[0] /((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5));
// 		value[1] = (phi_1-phi_2)*r_1*r_2*x[1] /((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5));
// 		value[2] = (phi_1-phi_2)*r_1*r_2*x[2] /((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5));
// 	}
// };

int main()
{
	
	std::string fname1{"/home/diako/Documents/cpp/punc_experimental/demo/capacitance/mesh/sphere_in_sphere_res16"};
	std::string fname2{"/home/diako/Documents/cpp/punc_experimental/demo/capacitance/mesh/sphere_in_sphere_res32"};
	std::string fname3{"/home/diako/Documents/cpp/punc_experimental/demo/capacitance/mesh/sphere_in_sphere_res64"};
	// std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/capacitance/mesh/main02"};
	std::vector<std::string> fnames = {fname1};
	auto num_cases = fnames.size();
	std::vector<double> error(num_cases);
	std::vector<double> h(num_cases);
	for(auto i = 0; i < num_cases; ++i)
	{
		//----------- Read the mesh --------------------------------------------------
		auto mesh = load_mesh(fnames[i]);
		auto boundaries = load_boundaries(mesh, fnames[i]);
		auto tags = get_mesh_ids(boundaries);
		std::size_t ext_bnd_id = tags[1];

		auto facet_vec = exterior_boundaries(boundaries, ext_bnd_id);
		//__________________________________________________________________________

		//--------------Create the function space-----------------------------------
		auto V = function_space(mesh);
		auto phi_exp = std::make_shared<Phi>();
		auto phi_a = std::make_shared<df::Function>(V);
		phi_a->interpolate(*phi_exp);
		auto rho = std::make_shared<df::Function>(V);
		//__________________________________________________________________________

		//-------------------Define boundary condition------------------------------
		auto u0 = std::make_shared<df::Constant>(0.0);
		df::DirichletBC bc(V, u0, boundaries, ext_bnd_id);
		std::vector<df::DirichletBC> ext_bc = {bc};
		//__________________________________________________________________________

		//----------Create the object-----------------------------------------------
		Object object(V, boundaries, tags[2]);
		object.set_potential(1.0);
		std::vector<Object> obj = {object};

		PoissonSolver poisson(V, ext_bc);
		// ESolver esolver(V);
		df::File file("phi.pvd");
		auto phi = poisson.solve(rho, obj);
		file<<*phi;
		// auto E = esolver.solve(phi);
		auto err = errornorm(phi, phi_a);

		error[i] = err;
		h[i] = mesh->hmin();
	}
	std::ofstream file;
	file.open("error.txt");
	for (const auto &e : error)
		file << e << "\n";
	file.close();

	file.open("h.txt");
	for (const auto &e : h)
		file << e << "\n";
	file.close();
	return 0;
}
