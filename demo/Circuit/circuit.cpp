/*
*	Tests circuitry for both BC and capacitance methods. The test is described 
*   in the PUNC paper. 
*/

#include <punc.h>
#include <dolfin.h>

using namespace punc;

using std::cout;
using std::endl;
using std::size_t;
using std::string;
using std::vector;

class Rho : public df::Expression
{
public:
	void eval(df::Array<double> &values, const df::Array<double> &x) const
	{
		values[0] = 100*x[1];
	}
};

int main()
{
	df::set_log_level(df::WARNING);

	string object_method{"CM"};
	PhysicalConstants constants;
	double eps0 = 1.0; //constants.eps0;
	double dt = 0.1;

	string fname{"mesh"};
	Mesh mesh(fname);

	/***************************************************************************
	 * PART I: TWO GROUNDED OBJECTS. THE IMPOSED VOLTAGES ARE 1V AND 2V, RESPECTIVELY
	 **************************************************************************/
	vector<Source> isources_r;
	vector<Source> vsources_r;

	vsources_r.push_back(Source{-1, 0, 1.0});
	vsources_r.push_back(Source{-1, 1, 2.0});

	auto V = CG1_space(mesh);
	auto W = CG1_vector_space(mesh);

	// The electric potential and electric field
	df::Function phi(std::make_shared<const df::FunctionSpace>(V));
	Rho rho_exp;
	df::Function rho(std::make_shared<const df::FunctionSpace>(V));
	rho.interpolate(rho_exp);

	auto u0 = std::make_shared<df::Constant>(0.0);
	df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V), u0,
					   std::make_shared<df::MeshFunction<size_t>>(mesh.bnd), mesh.ext_bnd_id);
	vector<df::DirichletBC> ext_bc = {bc};

	vector<std::shared_ptr<Object>> objects_r;
	std::shared_ptr<Circuit> circuit_r;
	if (object_method == "BC")
	{
		objects_r.push_back(std::make_shared<ObjectBC>(V, mesh, 2));
		objects_r.push_back(std::make_shared<ObjectBC>(V, mesh, 3));
		circuit_r = std::make_shared<CircuitBC>(V, objects_r, vsources_r, isources_r, dt, eps0);
	}
	else if (object_method == "CM")
	{
		objects_r.push_back(std::make_shared<ObjectCM>(V, mesh, 2));
		objects_r.push_back(std::make_shared<ObjectCM>(V, mesh, 3));
		circuit_r = std::make_shared<CircuitCM>(V, objects_r, vsources_r, isources_r, mesh, dt, eps0);
	}

	PoissonSolver poisson_r(V, ext_bc, circuit_r, eps0);

	if (object_method == "BC")
	{
		// SOLVE POISSON
		poisson_r.solve(phi, rho, objects_r, circuit_r);

		// UPDATE OBJECT CHARGE AND POTENTIAL
		for (auto &o : objects_r)
		{
			o->update(phi);
		}
	}
	else if (object_method == "CM")
	{
		// RESET OBJECT POTENTIAL TO 0
		reset_objects(objects_r);

		// SOLVE POISSON WITH POTENTIAL EQUAL 0 ON OBJECTS
		poisson_r.solve(phi, rho, objects_r);

		// APPLY CIRCUITRY
		circuit_r->apply(phi, mesh);

		// SOLVE POISSON WITH CORRECT POTENTIAL ON OBJECTS
		poisson_r.solve(phi, rho, objects_r, circuit_r);
	}

	cout << "----------------------------------" << '\n';
	cout << "              PART I              " << '\n';
	cout << "----------------------------------" << '\n';
	for(auto& o:objects_r)
	{
		cout << "charge = " << o->charge;
		cout << ", voltage = " << o->potential << '\n';
	}
	cout << "----------------------------------" << '\n';
	cout << '\n';
	
	/***************************************************************************
	 * PART II: TWO COFLOATING OBJECTS. THE VOLTAGE BETWEEN THE OBJECTS IS 1V
	 **************************************************************************/
	vector<double> charges = {objects_r[0]->charge, objects_r[1]->charge};

	vector<Source> isources;
	vector<Source> vsources;

	vsources.push_back(Source{0, 1, 1.0});

	vector<std::shared_ptr<Object>> objects;
	std::shared_ptr<Circuit> circuit;
	if (object_method == "BC")
	{
		objects.push_back(std::make_shared<ObjectBC>(V, mesh, 2));
		objects.push_back(std::make_shared<ObjectBC>(V, mesh, 3));
		circuit = std::make_shared<CircuitBC>(V, objects, vsources, isources, dt, eps0);
	}
	else if (object_method == "CM")
	{
		objects.push_back(std::make_shared<ObjectCM>(V, mesh, 2));
		objects.push_back(std::make_shared<ObjectCM>(V, mesh, 3));
		for (size_t i = 0; i < objects.size(); ++i)
		{
			objects[i]->charge = charges[i];
		}
		circuit = std::make_shared<CircuitCM>(V, objects, vsources, isources, mesh, dt, eps0);
	}

	PoissonSolver poisson(V, ext_bc, circuit, eps0);

	if (object_method == "BC")
	{
		// SOLVE POISSON
		poisson.solve(phi, rho, objects, circuit);

		// UPDATE OBJECT CHARGE AND POTENTIAL
		for (auto &o : objects)
		{
			o->update(phi);
		}
	}
	else if (object_method == "CM")
	{
		// RESET OBJECT POTENTIAL TO 0
		reset_objects(objects);

		// SOLVE POISSON WITH POTENTIAL EQUAL 0 ON OBJECTS
		poisson.solve(phi, rho, objects);

		// APPLY CIRCUITRY
		circuit->apply(phi, mesh);

		// SOLVE POISSON WITH CORRECT POTENTIAL ON OBJECTS
		poisson.solve(phi, rho, objects, circuit);
	}

	std::cout<<'\n';
	cout << "----------------------------------" << '\n';
	cout << "              PART II             " << '\n';
	cout << "----------------------------------" << '\n';
	for (auto &o : objects)
	{
		cout << "charge = " << o->charge;
		cout << ", voltage = " << o->potential << '\n';
	}
	cout << "----------------------------------" << '\n';

	return 0;
}
