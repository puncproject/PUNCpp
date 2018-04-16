#include <iostream>
#include <dolfin.h>
#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "../../punc/punc.h"

using namespace punc;

int main()
{
	auto r = 0.02;
	auto R = 0.2;
	auto cap = 4.*M_PI*r*R/(R-r);

	//----------- Create mesh --------------------------------------------------
	std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/capacitance/mesh/main02"};
	auto mesh = load_mesh(fname);

	auto boundaries = load_boundaries(mesh, fname);
	auto tags = get_mesh_ids(boundaries);
	std::size_t ext_bnd_id = tags[1];

	//__________________________________________________________________________

	//--------------Create the function space-----------------------------------
	auto V = function_space(mesh);
	//__________________________________________________________________________

	//-------------------Define boundary condition------------------------------
	auto u0 = std::make_shared<df::Constant>(0.0);
	auto bc = std::make_shared<df::DirichletBC>(V, u0, boundaries, ext_bnd_id);
	//__________________________________________________________________________

	//----------Create the object-----------------------------------------------
	Object object(V, boundaries, tags[2]);
	object.set_potential(1.0);
	std::vector<Object> obj = {object};
	//__________________________________________________________________________

	//__________________________________________________________________________

	//---------------------Capacitance------------------------------------------
	typedef boost::numeric::ublas::matrix<double> boost_matrix;

	boost_matrix inv_capacity = capacitance_matrix(V, obj, boundaries, ext_bnd_id);

	std::cout <<"Analytical value: "<< cap << '\n';
	//__________________________________________________________________________

	return 0;
}
