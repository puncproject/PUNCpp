#include <dolfin.h>
#include "../../punc/include/punc.h"

using namespace punc;

class Phi : public df::Expression
{
  public:
	double phi_1 = 1.0, phi_2 = 0.0;
	double r_1 = 0.2, r_2 = 1.0;
	void eval(df::Array<double> &values, const df::Array<double> &x) const
	{
		values[0] =  ((phi_1-phi_2)*r_1*r_2/( (r_2-r_1) * pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], 0.5) )) + phi_1-r_2*(phi_1-phi_2)/(r_2-r_1);
	}
};

class EField : public df::Expression
{
  public:
	double phi_1 = 1.0, phi_2 = 0.0;
	double r_1 = 0.2, r_2 = 1.0;
	EField(const int dim) : df::Expression(dim) {}
	void eval(df::Array<double> &values, const df::Array<double> &x) const
	{
		values[0] = (phi_1-phi_2)*r_1*r_2*x[0] /((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5));
		values[1] = (phi_1-phi_2)*r_1*r_2*x[1] /((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5));
		values[2] = (phi_1-phi_2)*r_1*r_2*x[2] /((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5));
	}
};

int main()
{
	df::set_log_level(df::WARNING);

	std::string fname1{"../../mesh/3D/sphere_in_sphere_res1"};
	std::string fname2{"../../mesh/3D/sphere_in_sphere_res2"};
	std::string fname3{"../../mesh/3D/sphere_in_sphere_res3"};

	std::vector<std::string> fnames = {fname1, fname2, fname3};
	auto num_cases = fnames.size()-1;
	std::vector<double> phi_error(num_cases);
	std::vector<double> E_error(num_cases);
	std::vector<double> h(num_cases);
	for(auto i = 0; i < num_cases; ++i)
	{
		//----------- Read the mesh --------------------------------------------
		auto mesh = load_h5_mesh(fnames[i]);
		auto boundaries = load_h5_boundaries(mesh, fnames[i]);
		auto tags = get_mesh_ids(boundaries);
		std::size_t ext_bnd_id = tags[1];
		auto comm = mesh->mpi_comm();
		//______________________________________________________________________
		//--------------Create the function space-------------------------------
		auto V = function_space(mesh);
		//______________________________________________________________________
		Phi phi_exp;
		df::Function phi_a(std::make_shared<df::FunctionSpace>(V));
		phi_a.interpolate(phi_exp);

		df::Function rho(std::make_shared<df::FunctionSpace>(V));
		//______________________________________________________________________
		//-------------------Define boundary condition--------------------------
		auto u0 = std::make_shared<df::Constant>(0.0);
		df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V), u0,
		std::make_shared<df::MeshFunction<std::size_t>>(boundaries), ext_bnd_id);
		std::vector<df::DirichletBC> ext_bc = {bc};
		//______________________________________________________________________

		//----------Create the object-------------------------------------------
		Object object(V, boundaries, tags[2]);
		object.set_potential(1.0);
		std::vector<Object> int_bc = {object};
		//______________________________________________________________________
		//----------Solvers-----------------------------------------------------
		PoissonSolver poisson(V, ext_bc);
		ESolver esolver(V);

		auto phi = poisson.solve(rho, int_bc);
		auto E = esolver.solve(phi);

		EField E_exp(mesh->geometry().dim());
		auto W = E.function_space();
		df::Function E_a(W);
		E_a.interpolate(E_exp);
		//______________________________________________________________________
		//----------Compute the error-------------------------------------------
		auto err_f = errornorm(phi, phi_a);
		auto err_e = errornorm(E, E_a);
		//______________________________________________________________________
		phi_error[i] = err_f;
		E_error[i] = err_e;
		h[i] = df::MPI::min(comm, mesh->hmin());
	}

	auto rank = df::MPI::rank(MPI_COMM_WORLD);
	if(rank == 0)
	{
		std::ofstream ofile;
		ofile.open("phi.txt");
		for (const auto &e : phi_error)
			ofile << e << "\n";
		ofile.close();
		ofile.open("E.txt");
		for (const auto &e : E_error)
			ofile << e << "\n";
		ofile.close();
		ofile.open("h.txt");
		for (const auto &e : h)
			ofile << e << "\n";
		ofile.close();
	}
	return 0;
}
