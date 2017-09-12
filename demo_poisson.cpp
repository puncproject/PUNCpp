#include <iostream>
#include <fstream>
#include <assert.h>
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

int main()
{
    bool show_plot = false;
    bool remove_null_space = true;
    bool with_object = false;
    std::size_t dim = 2;
    std::vector<int> N{4,8,16,32,64,128};

    std::vector<double> origin;
    std::vector<double> Ld;
    if (dim==2)
    {
        origin = {0.0, 0.0};
        Ld = {1.0, 1.0};
    }else if(dim==3)
    {
        origin = {0.0, 0.0, 0.0};
        Ld = {1.0, 1.0, 1.0};
    }

    std::vector<bool> periodic(dim);
    if(remove_null_space)
    {
        for (std::size_t i = 0; i<dim; ++i)
        {
            periodic[i] = true;
        }
    }else{
        for (std::size_t i = 0; i<dim; ++i)
        {
            periodic[i] = false;
        }
    }

    std::vector<double> error_v1(N.size());
    std::vector<double> error_v2(N.size());
    std::vector<double> h(N.size());

    std::vector<std::string> file_names{
                             "/home/diako/Documents/cpp/punc/mesh/circle.xml",
                             "/home/diako/Documents/cpp/punc/mesh/circle1.xml",
                             "/home/diako/Documents/cpp/punc/mesh/circle2.xml",
                             "/home/diako/Documents/cpp/punc/mesh/circle3.xml",
                             "/home/diako/Documents/cpp/punc/mesh/circle4.xml",
                             "/home/diako/Documents/cpp/punc/mesh/circle5.xml"};

    double radius = 0.5; // mshr stuff, not finished!
    std::vector<double> center = {M_PI, M_PI}; // mshr stuff, not finished!
    for (std::size_t i = 0; i<N.size(); ++i)
    {
        std::shared_ptr<const Mesh> mesh;
        if(with_object)
        {
            assert(dim == 2 && " Only 2D meshes are currently available! ");
            mesh = std::make_shared<const Mesh>(file_names[i]);
        }else{

            if (dim == 2)
            {
                Point p0(dim, origin.data());
                Point p1(dim, Ld.data());
                mesh = std::make_shared<const RectangleMesh>(p0, p1, N[i],N[i]);
            }else if(dim == 3){
                Point p0(dim, origin.data());
                Point p1(dim, Ld.data());
                mesh = std::make_shared<const BoxMesh>(p0, p1, N[i],N[i],N[i]);
            }
        }

        std::vector<double> Ld = get_mesh_size(mesh);

        std::shared_ptr<Potential::FunctionSpace> V;
        if(remove_null_space)
        {
            auto constr = std::make_shared<PeriodicBoundary>(Ld, periodic);
            V = std::make_shared<Potential::FunctionSpace>(mesh, constr);

        }else{
            V = std::make_shared<Potential::FunctionSpace>(mesh);
        }
        // std::cout<<"degree: "<<V->element()->ufc_element()->degree()<<'\n';

        auto rho_exp = std::make_shared<Rho>();
        auto rho = std::make_shared<Function>(V);
        rho->interpolate(*rho_exp);

        std::vector<std::shared_ptr<DirichletBC> > bc;
        std::shared_ptr<DirichletBC> bce;
        if (!remove_null_space)
        {
            auto boundary = std::make_shared<NonPeriodicBoundary>(Ld, periodic);
            bce = std::make_shared<DirichletBC>(V, rho, boundary);
            bc.emplace_back(bce);
        }

        auto phi_exp = std::make_shared<Phi>();
        auto phi_a = std::make_shared<Function>(V);
        phi_a->interpolate(*phi_exp);
        auto phi = std::make_shared<Function>(V);

        // Object
        double r = 0.25;
        double tol=1e-4;
        std::vector<double> s0{0.5, 0.5};

        auto func0 = \
        [r,s0,tol](const Array<double>& x)->bool
        {
            double dot = 0.0;
            for(std::size_t i = 0; i<s0.size(); ++i)
            {
                dot += (x[i]-s0[i])*(x[i]-s0[i]);
            }
            return dot <= r*r+tol;
        };

        auto obj_phi_exp = std::make_shared<ObjectPhi>();
        auto obj_phi = std::make_shared<Function>(V);
        obj_phi->interpolate(*obj_phi_exp);

        auto circle0 = std::make_shared<ObjectBoundary>(func0);
        std::shared_ptr<Object> object;

        std::shared_ptr<PoissonSolver> poisson;
        if(remove_null_space)
        {
            poisson = std::make_shared<PoissonSolver>(V, remove_null_space);
        }else
        {
            object = std::make_shared<Object>(V, circle0, obj_phi);
            poisson = std::make_shared<PoissonSolver>(V, bc);
        }

        if (with_object)
        {
            poisson->solve(phi, rho, object);
        }else{
            poisson->solve(phi, rho);
        }

        auto res = poisson->residual(phi);
        auto err = poisson->errornormv1(phi, phi_a);
        error_v1[i] = err;
        auto errv2 = poisson->errornormv2(phi, phi_a);
        error_v2[i] = errv2;
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
    file.open("error_v1.txt");
    for (const auto &e:error_v1)
        file << e <<"\n";
    file.close();

    file.open("error_v2.txt");
    for (const auto &e:error_v2)
        file << e <<"\n";
    file.close();

    file.open("h.txt");
    for (const auto &e:h)
        file << e <<"\n";
    file.close();

    return 0;
}
