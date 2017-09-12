#include <iostream>
#include <dolfin.h>
#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "punc/Potential.h"
#include "punc/Flux.h"
#include "punc/EField.h"
#include "punc/poisson.h"
#include "punc/object.h"
#include "punc/capacitance.h"
#include "create.h"

using namespace punc;
using namespace dolfin;

void object_dofs(std::vector<Object> &obj)
{
	std::size_t num_objs = obj.size();
	for(auto i=0; i<num_objs; ++i)
	{
		std::unordered_map<std::size_t, double> boundary_values;
		obj[i].get_boundary_values(boundary_values);

		std::cout << "Object nr. "<<i<<", boundary_values:"<<'\n';
		for (auto it = boundary_values.begin(); it != boundary_values.end(); ++it)
			std::cout << " " << it->first << ":" << it->second;
		std::cout << '\n';
	}
}

void coordinates(std::shared_ptr<df::Function> &phi,
				 std::shared_ptr<const Mesh> &mesh)
{
	std::cout<<"---geometry-----"<<'\n';
	std::size_t count = mesh->num_vertices();
	std::vector<double> vertex_values(count);
	phi->compute_vertex_values(vertex_values);
	for(std::size_t i = 0; i < count; ++i)
	{
		double X = mesh->geometry().point(i).x();
		double Y = mesh->geometry().point(i).y();
		double f = vertex_values[i];
		if (std::abs(f-10.0)<1e-3)
		{
			std::cout<<"i: "<<i<<", x: "<<X<<", y: "<<Y<<", phi: "<<f<<'\n';
		}
	}
}

int main()
{
	bool show_plot = true;
	bool write_to_file = false;
	std::vector<bool> periodic{false, false};
	int num_objs = 4;

	//----------- Create mesh --------------------------------------------------
	std::string file_name{"/home/diako/Documents/cpp/punc/mesh/circuit.xml"};
	auto mesh = std::make_shared<const Mesh>(file_name);
	std::vector<double> Ld = get_mesh_size(mesh);
	//__________________________________________________________________________

	//--------------Create the function space-----------------------------------
	auto V = std::make_shared<Potential::FunctionSpace>(mesh);
	//__________________________________________________________________________

	//-------------------Define boundary condition------------------------------
	auto u0 = std::make_shared<Constant>(0.0);
	auto boundary = std::make_shared<NonPeriodicBoundary>(Ld, periodic);
	auto bc = std::make_shared<DirichletBC>(V, u0, boundary);
	//__________________________________________________________________________

	//----------Create the object-----------------------------------------------
	auto circles = circle_objects();
	class Phi : public Expression
    {
    public:

        void eval(Array<double>& values, const Array<double>& x) const
        {
            values[0] = 10.0;
        }

	};
	Phi phi_exp;
	auto init_potential = std::make_shared<Function>(V);
	init_potential->interpolate(phi_exp);
	std::vector<double> potential = {1.0, 2.0, 3.0, 4.0};
	auto obj = create_objects(V, circles, potential);
	//__________________________________________________________________________

	//-------------------Create the source and the solution functions-----------
	auto rho_exp = std::make_shared<Source>();
	auto rho = std::make_shared<Function>(V);
	rho->interpolate(*rho_exp);
	auto phi = std::make_shared<Function>(V);
	//__________________________________________________________________________

	//--------------Create Poisson solver---------------------------------------
	PoissonSolver poisson(V, bc);
	poisson.solve(phi, rho, obj);
	auto E = electric_field(phi);
	//__________________________________________________________________________

	//---------------------Capacitance------------------------------------------
	typedef boost::numeric::ublas::matrix<double> boost_matrix;
	typedef boost::numeric::ublas::vector<double> boost_vector;

	boost_matrix inv_capacity = capacitance_matrix(V, poisson, boundary, obj);
	//__________________________________________________________________________

	//---------------------Circuits---------------------------------------------
	std::map <int, std::vector<int> > circuits_info;
	std::map <int, std::vector<double> > bias_potential;
	get_circuit(circuits_info, bias_potential);
	auto circuits = create_circuits(obj, inv_capacity, circuits_info,
		                            bias_potential);
	//__________________________________________________________________________

	//---------------Object proparties------------------------------------------
	for(auto i=0; i<num_objs; ++i)
	{
		obj[i].compute_interpolated_charge(phi);
		obj[i].vertices();
	}
	//__________________________________________________________________________

	//----------------------create facet function-------------------------------
	//             mark facets and mark cells adjacent to object
	//--------------------------------------------------------------------------
	auto num_objects = obj.size();
	auto facet_func = markers(mesh, obj);
	CellFunction<std::size_t> cell_func(mesh);
	cell_func.set_all(num_objects);
	for (int i = 0; i < num_objects; ++i)
	{
		obj[i].mark_cells(cell_func, facet_func, i);
	}
	//__________________________________________________________________________

	//-------------------------Plot solution------------------------------------
	if(show_plot)
	{
		plot(facet_func);
		plot(cell_func);
		plot(phi);
		plot(E);
		interactive();
	}
	//__________________________________________________________________________

	//------------------ Save solution in VTK format----------------------------
	if(write_to_file)
	{
		File file1("phi.pvd");
		file1 << *phi;
		XDMFFile("phi.xdmf").write(*phi);
		File file2("E.pvd");
		file2 << *E;
		XDMFFile("E.xdmf").write(*E);

	}
	//__________________________________________________________________________

	return 0;
}
