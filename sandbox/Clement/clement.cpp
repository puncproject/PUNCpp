#include <dolfin.h>
#include "../../punc/include/punc.h"
#include "Clement1D.h"
#include "Clement2D.h"
#include "Clement3D.h"
#include "Mean1D.h"
#include "Mean2D.h"
#include "Mean3D.h"
#include "EFieldDG01D.h"
#include "EFieldDG02D.h"
#include "EFieldDG03D.h"
#include <petscvec.h>

using namespace punc;

class EFieldMean
{
public:
	std::shared_ptr<df::FunctionSpace> V;
	EFieldMean(std::shared_ptr<const df::Mesh> mesh, bool arithmetic_mean=false);
	df::Function mean(const df::Function &phi);

private:
	std::shared_ptr<df::FunctionSpace> Q, W;
	std::shared_ptr<df::Form> a, b, c, d;
	df::PETScMatrix A;
	df::PETScVector ones, Av, e_dg0;
};

EFieldMean::EFieldMean(std::shared_ptr<const df::Mesh> mesh, bool arithmetic_mean)
{
	auto gdim = mesh->geometry().dim();
	auto tdim = mesh->topology().dim();
	std::vector<double> one_vec(gdim, 1.0);
	if (gdim == 1){
		V = std::make_shared<Mean1D::Form_a_FunctionSpace_0>(mesh); 
		Q = std::make_shared<Mean1D::Form_a_FunctionSpace_1>(mesh);
		W = std::make_shared<Mean1D::Form_d_FunctionSpace_0>(mesh);
		a = std::make_shared<Mean1D::Form_a>(Q, V);
		c = std::make_shared<Mean1D::Form_c>(Q, V);
		b = std::make_shared<Mean1D::Form_b>(Q);
		d = std::make_shared<Mean1D::Form_d>(W);
	}else if (gdim==2){
		V = std::make_shared<Mean2D::Form_a_FunctionSpace_0>(mesh);
		Q = std::make_shared<Mean2D::Form_a_FunctionSpace_1>(mesh);
		W = std::make_shared<Mean2D::Form_d_FunctionSpace_0>(mesh);
		a = std::make_shared<Mean2D::Form_a>(Q, V);
		c = std::make_shared<Mean2D::Form_c>(Q, V);
		b = std::make_shared<Mean2D::Form_b>(Q);
		d = std::make_shared<Mean2D::Form_d>(W);
	} else if (gdim==3){
		V = std::make_shared<Mean3D::Form_a_FunctionSpace_0>(mesh); 
		Q = std::make_shared<Mean3D::Form_a_FunctionSpace_1>(mesh);
		W = std::make_shared<Mean3D::Form_d_FunctionSpace_0>(mesh);
		a = std::make_shared<Mean3D::Form_a>(Q, V);
		c = std::make_shared<Mean3D::Form_c>(Q, V);
		b = std::make_shared<Mean3D::Form_b>(Q);
		d = std::make_shared<Mean3D::Form_d>(W);
	}
	a->set_coefficient("c1", std::make_shared<df::Constant>(tdim+1.0));
	b->set_coefficient("c2", std::make_shared<df::Constant>(one_vec));

	if(arithmetic_mean)
	{
		df::assemble(A, *c);
	}else{
		df::assemble(A, *a);
	}
	
	df::assemble(ones, *b);
	A.mult(ones, Av);
	auto A_vec = Av.vec();
	auto A_mat = A.mat();

	VecReciprocal(A_vec);
	MatDiagonalScale(A_mat, A_vec, NULL);
}

df::Function EFieldMean::mean(const df::Function &phi)
{
	// df::parameters["linear_algebra_backend"] = "PETSc";
	d->set_coefficient("phi", std::make_shared<df::Function>(phi));
	df::assemble(e_dg0, *d);

	df::Function E(V);
	A.mult(e_dg0, *E.vector());
	// E.vector()->update_ghost_values();
	return E;
}

class ClementInterpolant
{
public:
    std::shared_ptr<df::FunctionSpace> V;
	std::shared_ptr<df::FunctionSpace> Q;
    ClementInterpolant(std::shared_ptr<const df::Mesh> mesh);
    df::Function interpolate(const df::Function &u);

private:
	std::shared_ptr<df::Form> a, b;
	df::PETScMatrix A;
	df::PETScVector ones, Av;
};

ClementInterpolant::ClementInterpolant(std::shared_ptr<const df::Mesh> mesh)
{
	auto gdim = mesh->geometry().dim();
	auto tdim = mesh->topology().dim();
	if (gdim == 1){
		V = std::make_shared<Clement1D::Form_a_FunctionSpace_0>(mesh); 
		Q = std::make_shared<Clement1D::Form_a_FunctionSpace_1>(mesh);
		a = std::make_shared<Clement1D::BilinearForm>(Q, V);
		b = std::make_shared<Clement1D::LinearForm>(Q);
	}else if (gdim==2){
		V = std::make_shared<Clement2D::Form_a_FunctionSpace_0>(mesh);
		Q = std::make_shared<Clement2D::Form_a_FunctionSpace_1>(mesh);
		a = std::make_shared<Clement2D::BilinearForm>(Q, V);
		b = std::make_shared<Clement2D::LinearForm>(Q);
	} else if (gdim==3){
		V = std::make_shared<Clement3D::Form_a_FunctionSpace_0>(mesh); 
		Q = std::make_shared<Clement3D::Form_a_FunctionSpace_1>(mesh);
		a = std::make_shared<Clement3D::BilinearForm>(Q, V);
		b = std::make_shared<Clement3D::LinearForm>(Q);
	}
	a->set_coefficient("c1", std::make_shared<df::Constant>(tdim+1.0));
	b->set_coefficient("c2", std::make_shared<df::Constant>(1.0));

	df::assemble(A, *a);
	df::assemble(ones, *b);
	A.mult(ones, Av);
	auto A_vec = Av.vec();
	auto A_mat = A.mat();

	VecReciprocal(A_vec);
	MatDiagonalScale(A_mat, A_vec, NULL);
}

df::Function ClementInterpolant::interpolate(const df::Function &u)
{
	df::Function ui(V);
	auto u_vec = u.vector();
	auto ui_vec = ui.vector();
	A.mult(*u_vec, *ui_vec);
	return ui;
}

class EFieldDG0
{
public:
	std::shared_ptr<df::FunctionSpace> Q;
	EFieldDG0(std::shared_ptr<const df::Mesh> mesh);
	df::Function solve(const df::Function &phi);

private:
	std::shared_ptr<df::Form> M;
};

EFieldDG0::EFieldDG0(std::shared_ptr<const df::Mesh> mesh)
{
	auto gdim = mesh->geometry().dim();
	if (gdim == 1)
	{
		Q = std::make_shared<EFieldDG01D::FunctionSpace>(mesh);
		M = std::make_shared<EFieldDG01D::LinearForm>(Q);
	}
	else if (gdim == 2)
	{
		Q = std::make_shared<EFieldDG02D::FunctionSpace>(mesh);
		M = std::make_shared<EFieldDG02D::LinearForm>(Q);
	}
	else if (gdim == 3)
	{
		Q = std::make_shared<EFieldDG03D::FunctionSpace>(mesh);
		M = std::make_shared<EFieldDG03D::LinearForm>(Q);
	}
}

df::Function EFieldDG0::solve(const df::Function &phi)
{
	M->set_coefficient("phi", std::make_shared<df::Function>(phi));
	
	df::Function E(Q);
	df::assemble(*E.vector(), *M);
	return E;
}

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
	// std::string fname{"/home/diako/Documents/cpp/punc/mesh/2D/circle_in_square_res1"};
	std::string fname{"/home/diako/Documents/Software/punc/mesh/3D/laframboise_sphere_in_sphere_res1b"};

	//----------- Read the mesh --------------------------------------------
	auto mesh = load_mesh(fname);
	// auto boundaries = load_boundaries(mesh, fname);

	// auto tags = get_mesh_ids(boundaries);
	// std::size_t ext_bnd_id = tags[1];
	// auto gdim = mesh->geometry().dim();
	// auto tdim = mesh->topology().dim();

	// Clement2D::Form_a_FunctionSpace_0 V(mesh); // CG1
	// Clement2D::Form_a_FunctionSpace_1 Q(mesh); // DG0
	// Clement2D::Form_b_FunctionSpace_0 W(mesh); // DG0

	// auto V_shared = std::make_shared<df::FunctionSpace>(V);
	// auto Q_shared = std::make_shared<df::FunctionSpace>(Q);
	// auto W_shared = std::make_shared<df::FunctionSpace>(W);

    // std::shared_ptr<df::Form> a, b;
    // df::PETScMatrix A;
	// df::PETScVector ones, Avec;

	// a = std::make_shared<Clement2D::BilinearForm>(Q_shared, V_shared);
	// b = std::make_shared<Clement2D::LinearForm>(Q_shared);

	// a->set_coefficient("c1", std::make_shared<df::Constant>(tdim+1.0));
	// b->set_coefficient("c2", std::make_shared<df::Constant>(1.0));

	// df::assemble(A, *a);
	// df::assemble(ones, *b);

	// // df::Function Av(Q_shared);
	// // auto Avec = Av.vector();

	// A.mult(ones, Avec);
	// std::cout<<Avec.size()<<'\n';
	// std::cout<<A.size(0)<<'\n';
	// std::cout<<A.size(1)<<'\n';
	// std::cout << ones.size() << '\n';

	// auto A_vec = Avec.vec();
	// auto A_mat = A.mat();

	// VecReciprocal(A_vec);
	// MatDiagonalScale(A_mat, A_vec, NULL);


	// auto sign0 = V.element()->signature();
	// auto sign1 = Q.element()->signature();
	// auto sign2 = W.element()->signature();
	// std::cout<<sign0<<'\n';
	// std::cout<<sign1<<'\n';
	// std::cout<<sign2<<'\n';


	ClementInterpolant ci(mesh);	
	auto Q = ci.Q;

	Phi phi_exp;
	df::Function phi_e(Q);
	phi_e.interpolate(phi_exp);
	
	auto phi_cg = ci.interpolate(phi_e);

	EFieldMean cl(mesh);
	auto E = cl.mean(phi_cg);
	// auto W = E.function_space();

	df::Function E_a(E.function_space());
	EField E_exp(mesh->geometry().dim());
	E_a.interpolate(E_exp);

	EFieldDG0 esolve_dg0(mesh);
	auto E_dg0 = esolve_dg0.solve(phi_cg);
	// df::Function phi_cg(V_shared);
	// auto phi_vec = phi_cg.vector();
	// auto phi_e_vec = phi_e.vector();
	// A.mult(*phi_e_vec, *phi_vec);

	df::File out("phi.pvd");
	out<<phi_cg;
	df::File out2("phiDG.pvd");
	out2<<phi_e;
	df::File out3("E.pvd");
	out3<<E;
	df::File out4("E_a.pvd");
	out4<<E_a;
	df::File out5("E_dg0.pvd");
	out5 << E_dg0;
	// auto comm = mesh->mpi_comm();
	// auto my_rank = df::MPI::rank(comm);
	// auto num_proc = df::MPI::size(comm);

	// if(my_rank==0)
	// {
	// 	printf("size: %d, rank: %d\n", num_proc, my_rank);
	// 	for(auto &e:tags)
	// 	{
	// 		std::cout<<e<<"  ";
	// 	}
	// 	std::cout<<'\n';
	// 	// df::plot(mesh);
	// 	// df::interactive();
	// }

	
	// //______________________________________________________________________
	// //--------------Create the function space-------------------------------
	// auto V = function_space(mesh);
	// //______________________________________________________________________
	// Phi phi_exp;
	// df::Function phi_e(std::make_shared<df::FunctionSpace>(V));
	// phi_e.interpolate(phi_exp);
	// df::Function rho(std::make_shared<df::FunctionSpace>(V));
	// //______________________________________________________________________
	// //-------------------Define boundary condition--------------------------
	// auto u0 = std::make_shared<df::Constant>(0.0);
	// df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V), u0, 
	// std::make_shared<df::MeshFunction<std::size_t>>(boundaries), ext_bnd_id);
	// std::vector<df::DirichletBC> ext_bc = {bc};
	// //______________________________________________________________________

	// //----------Create the object-------------------------------------------
	// Object object(V, boundaries, tags[2]);
	// object.set_potential(1.0);
	// std::vector<Object> int_bc = {object};
	// //______________________________________________________________________
	// //----------Solvers-----------------------------------------------------
	// PoissonSolver poisson(V, ext_bc);
	// ESolver esolver(V);

	// auto phi = poisson.solve(rho, int_bc);
	// auto phi_val = phi.value_rank();

	// auto E = esolver.solve(phi);
	// auto E_val = E.value_rank();
	// EField E_exp(mesh->geometry().dim());
	// df::Function E_a(E.function_space());
	// E_a.interpolate(E_exp);

	// auto err = errornorm(phi, phi_e);
	// auto err_e = errornorm(E, E_a);
	// auto h = df::MPI::min(comm, mesh->hmin());
	// std::cout << "hmin: " << h << '\n';
	// if(my_rank==0)
	// {
	// 	std::cout << "error phi: " << err << ", error E: " << err_e << '\n';
	// 	std::cout << "phi: " << phi_val << ", E: " << E_val << '\n';
	// }
	// df::plot(phi_e);
	// df::interactive();
	return 0;
}
