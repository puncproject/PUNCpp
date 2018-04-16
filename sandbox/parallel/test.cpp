#include <dolfin.h>
#include "../../punc/include/punc.h"

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

class EField : public df::Expression
{
  public:
	double phi_1 = 1.0, phi_2 = 0.0;
	double r_1 = 0.02, r_2 = 0.2;
	EField(const int dim) : df::Expression(dim){}
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

    // std::string fname{"/home/diako/Documents/cpp/punc/mesh/2D/circle_in_square_res1"};
	std::string fname{"/home/diako/Documents/cpp/punc/mesh/3D/sphere_in_sphere_res1"};

	//----------- Read the mesh --------------------------------------------
	auto mesh = load_h5_mesh(fname);
	auto boundaries = load_h5_boundaries(mesh, fname);

	auto tags = get_mesh_ids(boundaries);
	std::size_t ext_bnd_id = tags[1];

	auto comm = mesh->mpi_comm();
	auto my_rank = df::MPI::rank(comm);
	auto num_proc = df::MPI::size(comm);

	if(my_rank==0)
	{
		printf("size: %d, rank: %d\n", num_proc, my_rank);
		for(auto &e:tags)
		{
			std::cout<<e<<"  ";
		}
		std::cout<<'\n';
		// df::plot(mesh);
		// df::interactive();
	}

	
	//______________________________________________________________________
	//--------------Create the function space-------------------------------
	auto V = function_space(mesh);
	//______________________________________________________________________
	Phi phi_exp;
	df::Function phi_e(std::make_shared<df::FunctionSpace>(V));
	phi_e.interpolate(phi_exp);
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
	auto phi_val = phi.value_rank();

	auto E = esolver.solve(phi);
	auto E_val = E.value_rank();
	EField E_exp(mesh->geometry().dim());
	df::Function E_a(E.function_space());
	E_a.interpolate(E_exp);

	auto err = errornorm(phi, phi_e);
	auto err_e = errornorm(E, E_a);
	auto h = df::MPI::min(comm, mesh->hmin());
	std::cout << "hmin: " << h << '\n';
	if(my_rank==0)
	{
		std::cout << "error phi: " << err << ", error E: " << err_e << '\n';
		std::cout << "phi: " << phi_val << ", E: " << E_val << '\n';
	}
	// df::plot(phi_e);
	// df::interactive();
	return 0;
}
