#include <dolfin.h>
#include "../../punc/include/punc.h"

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
    df::set_log_level(df::WARNING);

    //----------- Create mesh --------------------------------------------------
    std::string fname{"/home/diako/Documents/cpp/punc/mesh/3D/sphere_in_sphere_res1"};
    auto mesh = load_mesh(fname);
	auto boundaries = load_boundaries(mesh, fname);
	auto tags = get_mesh_ids(boundaries);
	std::size_t ext_bnd_id = tags[1];
	std::size_t int_bnd_id = tags[2];

    auto Q =  10.0;
	//__________________________________________________________________________

	//--------------Create the function space-----------------------------------
    auto W = var_function_space(mesh);
    std::size_t component = 0;
    auto W0 = W.sub(component)->collapse();
	//__________________________________________________________________________

    Rho rho_exp;
    df::Function rho(W0);
    rho.interpolate(rho_exp);

	//-------------------Define boundary condition------------------------------
    df::DirichletBC ext_bc(W.sub(0), std::make_shared<df::Function>(rho),
                           std::make_shared<df::MeshFunction<std::size_t>>(boundaries), ext_bnd_id);
    std::vector<df::DirichletBC> ext_bcs{ext_bc};
	//__________________________________________________________________________

	//----------Create the object-----------------------------------------------
    VObject vobject(W, boundaries, int_bnd_id);

    VarPoissonSolver poisson(W, ext_bcs, vobject);

    auto phi = poisson.solve(rho, Q, vobject);
    auto Qm = vobject.calculate_charge(phi);
    std::cout<<"Qm: "<<Qm<<'\n';
	return 0;
}