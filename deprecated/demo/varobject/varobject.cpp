#include <dolfin.h>
#include "../../punc/punc.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <string>
#include <fstream>
#include <cstdio>
#include <limits>
#include <cmath>
#include <stdlib.h>

using namespace punc;

class Rho : public df::Expression
{
public:

    void eval(df::Array<double>& values, const df::Array<double>& x) const
    {
        values[0] = 8*M_PI*M_PI*sin(2*M_PI*x[0])*sin(2*M_PI*x[1]);
    }

};

class Phi : public df::Expression
{
public:

    void eval(df::Array<double>& values, const df::Array<double>& x) const
    {
        values[0] = sin(2*M_PI*x[0])*sin(2*M_PI*x[1]);
    }

};

class ObjectPhi : public df::Expression
{
    void eval(df::Array<double>& values, const df::Array<double>& x) const
    {
        values[0] = sin(2*M_PI*x[0])*sin(2*M_PI*x[1]);
    }
};

int main()
{
	//----------- Create mesh --------------------------------------------------
	// std::string fname{"/home/diako/Documents/Software/punc/mesh/2D/circle_in_square_res1"};
    std::string fname{"/home/diako/Documents/Software/punc/mesh/3D/sphere_in_sphere_res1"};
	auto mesh = load_mesh(fname);

	auto boundaries = load_boundaries(mesh, fname);
	auto tags = get_mesh_ids(boundaries);
	std::size_t ext_bnd_id = tags[1];
	std::size_t int_bnd_id = tags[2];
    std::cout<<ext_bnd_id<<"  "<<int_bnd_id<<'\n';

    auto Q =  10.0;
	//__________________________________________________________________________

	//--------------Create the function space-----------------------------------
    auto W = var_function_space(mesh);
    std::size_t component = 0;
    auto W0 = W->sub(component)->collapse();
	//__________________________________________________________________________

	auto rho_exp = std::make_shared<Rho>();
	auto rho = std::make_shared<df::Function>(W0);
	rho->interpolate(*rho_exp);
	// auto phi_exp = std::make_shared<Phi>();
	// auto phi_a = std::make_shared<Function>(V);
	// phi_a->interpolate(*phi_exp);
	// auto phi = std::make_shared<Function>(V);

	//-------------------Define boundary condition------------------------------
    df::DirichletBC ext_bc(W->sub(0), rho, boundaries, ext_bnd_id);
    std::vector<df::DirichletBC> ext_bcs{ext_bc};
	//__________________________________________________________________________

	//----------Create the object-----------------------------------------------
    VObject vobject(W, boundaries, int_bnd_id);

    VarPoissonSolver poisson(W, ext_bcs, vobject);

    auto phi = poisson.solve(rho, Q, vobject);
    auto Qm = vobject.calculate_charge(phi);
    std::cout<<"Qm: "<<Qm<<'\n';

    // plot(phi);
    // interactive();
	return 0;
}