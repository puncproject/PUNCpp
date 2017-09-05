#include <iostream>
#include <fstream>
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

using namespace punc;
using namespace dolfin;


int main()
{
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

    bool show_plot = false;
    std::vector<int> N{4,8,16,32,64,128};
    bool remove_null_space = false;
    auto dim = 2;

    std::vector<bool> periodic(dim);
    if(remove_null_space)
    {
        for (int i = 0; i<dim; ++i)
        {
            periodic[i] = true;
        }
    }else{
        for (int i = 0; i<dim; ++i)
        {
            periodic[i] = false;
        }
    }

    std::vector<double> error(N.size());
    std::vector<double> errorv2(N.size());
    std::vector<double> h(N.size());
    for (int i = 0; i<N.size(); ++i)
    {
        std::shared_ptr<const UnitSquareMesh> mesh;
        if (dim == 2)
        {
            mesh = std::make_shared<const UnitSquareMesh>(N[i],N[i]);
        }else if(dim == 3){
            mesh = std::make_shared<const UnitSquareMesh>(N[i],N[i],N[i]);
        }

        std::vector<double> Ld = get_mesh_size(mesh);

        auto constr = std::make_shared<PeriodicBoundary>(Ld, periodic);
        auto boundary = std::make_shared<NonPeriodicBoundary>(Ld, periodic);

        auto V = std::make_shared<Potential::FunctionSpace>(mesh, constr);
        // std::cout<<"degree: "<<V->element()->ufc_element()->degree()<<'\n';

        auto rho_exp = std::make_shared<Rho>();
        auto rho = std::make_shared<Function>(V);
        rho->interpolate(*rho_exp);

        std::vector<std::shared_ptr<DirichletBC> > bc;
        std::shared_ptr<DirichletBC> bce;
        if (!remove_null_space)
        {
            bce = std::make_shared<DirichletBC>(V, rho, boundary);
            bc.push_back(bce);
        }
        
        auto phi_exp = std::make_shared<Phi>();
        auto phi_a = std::make_shared<Function>(V);
        phi_a->interpolate(*phi_exp);
        auto phi = std::make_shared<Function>(V);

        std::shared_ptr<PoissonSolver> poisson;
        if(remove_null_space)
        {
            poisson = std::make_shared<PoissonSolver>(V, remove_null_space);
        }else{
            poisson = std::make_shared<PoissonSolver>(V, bc, remove_null_space);
        }

        poisson->solve(phi, rho);

        auto res = poisson->residual(phi);
        auto err = poisson->errornormv1(phi, phi_a);
        error[i] = err;
        auto errv2 = poisson->errornormv2(phi, phi_a);
        errorv2[i] = errv2;        
        h[i] = mesh->hmin();

        if (show_plot)
        {
            plot(rho);
            plot(phi_a);
            plot(phi);
            interactive();
        }

    }

    std::ofstream file;
    file.open("error.txt");
    for (const auto &e:error)
        file << e <<"\n";
    file.close();
    
    file.open("error2.txt");
    for (const auto &e:errorv2)
        file << e <<"\n";
    file.close();

    file.open("h.txt");
    for (const auto &e:h)
        file << e <<"\n";
    file.close();

    return 0;
}