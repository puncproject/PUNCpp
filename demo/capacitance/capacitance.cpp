#include <dolfin.h>
#include "../../punc/include/punc.h"

using namespace punc;

int main()
{
	df::set_log_level(df::WARNING);
	auto r = 0.2;
	auto R = 1.0;
	auto cap = 4.*M_PI*r*R/(R-r);

	//----------- Create mesh --------------------------------------------------
	std::string fname{"../../mesh/3D/sphere_in_sphere_res1"};
	auto mesh = load_mesh(fname);

	auto boundaries = load_boundaries(mesh, fname);
	auto tags = get_mesh_ids(boundaries);
	std::size_t ext_bnd_id = tags[1];
	//__________________________________________________________________________

	//--------------Create the function space-----------------------------------
	auto V = function_space(mesh);
	//__________________________________________________________________________

	//----------Create the object-----------------------------------------------
	Object object(V, boundaries, tags[2]);
	object.set_potential(1.0);
	std::vector<Object> obj = {object};
	//__________________________________________________________________________

	//---------------------Capacitance------------------------------------------
	boost_matrix inv_capacity = capacitance_matrix(V, obj, boundaries, ext_bnd_id);

	auto error = (cap-1./inv_capacity(0,0) )*inv_capacity(0,0);
	printf("Analatical value: %f, numerical value: %f, error: %f\n", cap, 1./inv_capacity(0,0), error);
	//__________________________________________________________________________

	return 0;
}
