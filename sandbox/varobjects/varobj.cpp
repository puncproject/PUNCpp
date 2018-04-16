#include <iostream>
#include <dolfin.h>
#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
//#include "../../punc/Potential.h"
//#include "../../punc/Flux.h"
//#include "../../punc/Surface.h"
//#include "../../punc/EField.h"
#include "../../punc/poisson.h"
#include "VarPotential1D.h"
#include "VarPotential2D.h"
#include "VarPotential3D.h"
#include "Charge.h"
#include <stdio.h>

using namespace punc;
using namespace dolfin;

class Rho : public Expression
{
public:

    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = 8*M_PI*M_PI*sin(2*M_PI*x[0])*sin(2*M_PI*x[1]);
    }

};

class Phi : public Expression
{
public:

    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = sin(2*M_PI*x[0])*sin(2*M_PI*x[1]);
    }

};

class ObjectPhi : public Expression
{
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = sin(2*M_PI*x[0])*sin(2*M_PI*x[1]);
    }
};

class VObjects : public df::DirichletBC
{
public:
    double potential;
    double charge;
    bool floating;
    std::size_t id;
    std::shared_ptr<df::MeshFunction<std::size_t>> bnd;
    std::vector<std::size_t> dofs;
    std::size_t num_dofs;
    std::shared_ptr<df::Form> charge_form;

    VObjects(const std::shared_ptr<df::FunctionSpace> &V,
           std::shared_ptr<df::MeshFunction<std::size_t>> boundaries,
           std::size_t bnd_id,
           double potential = 0.0,
           double charge = 0.0,
           bool floating = true,
           std::string method = "topological"):
           df::DirichletBC(V->sub(0), std::make_shared<df::Constant>(potential),
           boundaries, bnd_id, method), potential(potential),
           charge(charge), floating(floating), id(bnd_id){

               auto tags = boundaries->values();
               auto size = boundaries->size();
               bnd = std::make_shared<df::MeshFunction<std::size_t>>();
               *bnd = *boundaries;
               bnd->set_all(0);
               for (std::size_t i = 0; i < size; ++i)
               {
                    if(tags[i] == id)
                    {
                        bnd->set_value(i, 9999);
                    }
               }
               auto mesh = V->mesh();
               auto dim = mesh->geometry().dim();
               if (dim == 1)
               {
                   charge_form = std::make_shared<Charge::Form_0>(mesh);
               }
               else if (dim == 2)
               {
                   charge_form = std::make_shared<Charge::Form_1>(mesh);
               }
               else if (dim == 3)
               {
                   charge_form = std::make_shared<Charge::Form_2>(mesh);
               }
               charge_form->set_exterior_facet_domains(bnd);
               get_dofs();
           }
void get_dofs()
{
   std::unordered_map<std::size_t, double> dof_map;
   get_boundary_values(dof_map);

   for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
   {
       dofs.emplace_back(itr->first);
   }
   num_dofs = dofs.size();
}

void add_charge(const double &q)
{
   charge += q;
}
double calculate_charge(df::Function &phi)
{
     charge_form->set_coefficient("w0", std::make_shared<df::Function>(phi));
     return df::assemble(*charge_form);
}
void set_potential(double voltage)
{
   this->potential = voltage;
   this->set_value(std::make_shared<df::Constant>(voltage));
}

void apply(df::GenericVector &b)
{
    df::DirichletBC::apply(b);
}

void apply(df::GenericMatrix &A)
{
    if(!floating){
        df::DirichletBC::apply(A);
    }else{
        for(auto i = 1; i < num_dofs; ++i)
        {
            std::vector<double> neighbor_values;
            std::vector<std::size_t> neighbor_ids, surface_neighbors;
            A.getrow(dofs[i], neighbor_ids, neighbor_values);
            std::size_t num_neighbors = neighbor_ids.size();
            std::fill(neighbor_values.begin(), neighbor_values.end(), 0.0);

            std::size_t num_surface_neighbors = 0;
            std::size_t self_index;
            for(auto j = 0; j <num_neighbors; ++j)
            {
                if(std::find(dofs.begin(), dofs.end(), neighbor_ids[j]) != dofs.end())
                {
                    neighbor_values[j] = -1.0;
                    num_surface_neighbors += 1;
                    if(neighbor_ids[j]==dofs[i]){
                        self_index = j;
                    }
                }
            }
            neighbor_values[self_index] = num_surface_neighbors - 1;
            A.setrow(dofs[i], neighbor_ids, neighbor_values);
            A.apply("insert");
        }
    }
}
};

class LagrangeMultiplierSolver
{
public:
    std::shared_ptr<df::FunctionSpace> &V;
    std::vector<df::DirichletBC> ext_bc;
    df::PETScKrylovSolver solver;
    std::shared_ptr<df::Form> a, L;
    df::PETScMatrix A;
    df::PETScVector b;

    std::shared_ptr<df::Constant> S;
    LagrangeMultiplierSolver(std::shared_ptr<df::FunctionSpace> &V,
                             std::vector<df::DirichletBC> &ext_bc,
                             VObjects &vobject,
                             std::string method = "tfqmr",
                             std::string preconditioner = "none"): V(V), 
                             ext_bc(ext_bc),
                             solver(method, preconditioner)
   {
       auto dim = V->mesh()->geometry().dim();
       if (dim == 1)
       {
           a = std::make_shared<VarPotential1D::BilinearForm>(V, V);
           L = std::make_shared<VarPotential1D::LinearForm>(V);
       }
       else if (dim == 2)
       {
           a = std::make_shared<VarPotential2D::BilinearForm>(V, V);
           L = std::make_shared<VarPotential2D::LinearForm>(V);
       }
       else if (dim == 3)
       {
           a = std::make_shared<VarPotential3D::BilinearForm>(V, V);
           L = std::make_shared<VarPotential3D::LinearForm>(V);
       }
       a->set_exterior_facet_domains(vobject.bnd);
       L->set_exterior_facet_domains(vobject.bnd);

       df::assemble(A, *a);
       for(auto& bc: ext_bc)
       {
           bc.apply(A);
       }
        auto mesh = V->mesh();
        auto surface = surface_area(mesh, vobject.bnd);
        S = std::make_shared<df::Constant>(surface);
        L->set_coefficient("S", S);

        solver.parameters["absolute_tolerance"] = 1e-14;
        solver.parameters["relative_tolerance"] = 1e-12;
        solver.parameters["maximum_iterations"] = 100000;
        solver.set_reuse_preconditioner(true);
   }

   df::Function solve(std::shared_ptr<df::Function> &rho, double Q,
                      VObjects& int_bc)
   {
        L->set_coefficient("rho", rho);
        L->set_coefficient("Q", std::make_shared<df::Constant>(Q));

        df::assemble(b, *L);
        for (auto &bc : ext_bc)
        {
            bc.apply(b);
        }

        int_bc.apply(b);
        int_bc.apply(A);

        df::Function wh(V);
        solver.solve(A, *wh.vector(), b);
        return wh[0];
   }
};

std::shared_ptr<df::FunctionSpace> function_space_(std::shared_ptr<const df::Mesh> &mesh)
{
    std::shared_ptr<df::FunctionSpace> V;
    if (mesh->geometry().dim() == 1)
    {
        V = std::make_shared<VarPotential1D::FunctionSpace>(mesh);
    }
    else if (mesh->geometry().dim() == 2)
    {
        V = std::make_shared<VarPotential2D::FunctionSpace>(mesh);
    }
    else if (mesh->geometry().dim() == 3)
    {
        V = std::make_shared<VarPotential3D::FunctionSpace>(mesh);
    }
    else
        df::error("PUNC is programmed for dimensions up to 3D only.");

    return V;
}

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
    auto W = function_space_(mesh);
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
    VObjects int_bc(W, boundaries, int_bnd_id);

    LagrangeMultiplierSolver LSolver(W, ext_bcs, int_bc);

    auto phi = LSolver.solve(rho, Q, int_bc);
    auto Qm = int_bc.calculate_charge(phi);
    std::cout<<"Qm: "<<Qm<<'\n';

    // plot(phi);
    // interactive();
	return 0;
}