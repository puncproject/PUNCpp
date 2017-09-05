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

int main()
{

  std::vector<bool> periodic{false, false};
  int num_objs = 4;
  //----------- Create mesh ----------------------------------------------------
  auto mesh = std::make_shared<const Mesh>("/home/diako/Documents/cpp/punc/mesh/circuit.xml");
  // auto mesh = std::make_shared<UnitSquareMesh>(5,5);
  std::vector<double> Ld = get_mesh_size(mesh);
  std::cout<<"----Ld----"<<'\n';
  for(auto & c:Ld)
  {
    std::cout <<c<<"  ";
  }
  std::cout <<'\n';
  //____________________________________________________________________________

  //--------------Create the function space-------------------------------------
  auto V = std::make_shared<Potential::FunctionSpace>(mesh);
  //____________________________________________________________________________

  //-------------------Define boundary condition--------------------------------
  auto u0 = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<NonPeriodicBoundary>(Ld, periodic);
  auto bc = std::make_shared<DirichletBC>(V, u0, boundary);
  //____________________________________________________________________________

  //----------Create the object-------------------------------------------------
  double potential = 0.0;
  auto init_potential = std::make_shared<Constant>(potential);
  double r = 0.5;
  double tol=1e-4;
  std::vector<double> s0{M_PI, M_PI};
  std::vector<double> s1{M_PI, M_PI+3*r};
  std::vector<double> s2{M_PI, M_PI-3*r};                                      
  std::vector<double> s3{M_PI+3*r, M_PI};  

  std::vector<std::shared_ptr<Object>> obj(num_objs);

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

  auto func1 = \
  [r,s1,tol](const Array<double>& x)->bool
  {
      double dot = 0.0;
      for(std::size_t i = 0; i<s1.size(); ++i)
      {
          dot += (x[i]-s1[i])*(x[i]-s1[i]);
      }
      return dot <= r*r+tol;
  };
  
  auto func2 = \
  [r,s2,tol](const Array<double>& x)->bool
  {
      double dot = 0.0;
      for(std::size_t i = 0; i<s2.size(); ++i)
      {
          dot += (x[i]-s2[i])*(x[i]-s2[i]);
      }
      return dot <= r*r+tol;
  };
  
  auto func3 = \
  [r,s3,tol](const Array<double>& x)->bool
  {
      double dot = 0.0;
      for(std::size_t i = 0; i<s3.size(); ++i)
      {
          dot += (x[i]-s3[i])*(x[i]-s3[i]);
      }
      return dot <= r*r+tol;
  };

  std::vector<std::shared_ptr<ObjectBoundary>> circles(num_objs);

  auto circle0 = std::make_shared<ObjectBoundary>(func0);
  auto circle1 = std::make_shared<ObjectBoundary>(func1);
  auto circle2 = std::make_shared<ObjectBoundary>(func2);
  auto circle3 = std::make_shared<ObjectBoundary>(func3);
  circles = {circle0, circle1, circle2, circle3};

  for(int i = 0; i < num_objs; ++i)
  {
    // auto object = std::make_shared<Object>(V, circles[i], init_potential);
    auto object = std::make_shared<Object>(V, circles[i], init_potential);
    object->set_potential(10.0);
    obj[i] = object;
  }
  //____________________________________________________________________________

  //--------------Dofs of object------------------------------------------------
  for(auto i=0; i<num_objs; ++i)
  {
    std::unordered_map<std::size_t, double> boundary_values;
    obj[i]->get_boundary_values(boundary_values);

    std::cout << "Object nr. "<<i<<", boundary_values:"<<'\n';
    for (auto it = boundary_values.begin(); it != boundary_values.end(); ++it)
      std::cout << " " << it->first << ":" << it->second;
    std::cout << '\n';
  }
  //____________________________________________________________________________

  //-------------------Create the source and the solution functions-------------
  auto rho_exp = std::make_shared<Source>();
  auto rho = std::make_shared<Function>(V);
  rho->interpolate(*rho_exp);
  auto phi = std::make_shared<Function>(V);
  //____________________________________________________________________________

  //--------------Create Poisson solver-----------------------------------------
  auto poisson = std::make_shared<PoissonSolver>(V, bc);
  poisson->solve(phi, rho, obj);
  auto E = electric_field(phi);
  //____________________________________________________________________________

  //---------------------Capacitance--------------------------------------------
  typedef boost::numeric::ublas::matrix<double> boost_matrix;
  typedef boost::numeric::ublas::vector<double> boost_vector;

  boost_matrix inv_capacity = capacitance_matrix(V, poisson, boundary, obj);
  std::map <int, std::vector<int> > circuits_info;
  circuits_info[0] = std::vector<int>({0,2});
  circuits_info[1] = std::vector<int>({1,3});

  std::size_t num_circuits = circuits_info.size();

  std::map <int, std::vector<double> > bias_potential;
  bias_potential[0] = std::vector<double>({0.1});
  bias_potential[1] = std::vector<double>({0.2});


  for(auto const& imap: circuits_info)
  {
    std::cout<<"first: "<<imap.first<<'\n';
    for(auto &x: imap.second){
      std::cout<<"x: "<<x<<'\n';
    }
  }

  std::size_t len_bias_potential = 0;
  for(auto const& imap: bias_potential)
  {
    len_bias_potential += imap.second.size();
  }
  boost_vector biases(len_bias_potential);
  auto i = 0;
  for(auto const& imap: bias_potential)
  {
    for(auto &x: imap.second)
    {
      biases(i) = x;
      i++;
    }
  }
  std::cout<<"biases: "<<biases<<'\n';

  boost_matrix inv_bias = bias_matrix(inv_capacity, circuits_info);

  boost_vector bias_0(inv_bias.size1(), 0.0);

  for(std::size_t i = 0; i<inv_bias.size1(); ++i)
  {
    for(std::size_t j = 0; j<len_bias_potential; ++j)
    {
      bias_0(i) += inv_bias(i,j)*biases(j);
    }
  }
  std::cout<<"bias_0: "<<bias_0<<'\n';

  std::vector<std::shared_ptr<Circuit> > circuits(num_circuits);
  std::vector<int> circuit;
  i = 0;
  for(auto const& c_map: circuits_info)
  {
    circuit = c_map.second;
    std::vector<std::shared_ptr<Object> > circuit_comps(circuit.size());
    boost_vector bias_0_comp(circuit.size(), 0.0);
    boost_matrix inv_bias_matrix(circuit.size(), inv_bias.size2()-len_bias_potential, 0.0);
    int j = 0;
    for(auto &x: c_map.second)
    {
      circuit_comps[j] = obj[x];
      bias_0_comp[j] = bias_0(x);
      j++;
    }
    for(std::size_t k = 0; k<circuit.size(); ++k)
    {
      int m = 0;
      for(std::size_t l = len_bias_potential; l<inv_bias.size2(); ++l)
      {
        inv_bias_matrix(k,m) = inv_bias(circuit[k], l);
        m++;
      }
    }
    std::cout<<"inv_bias_matrix: "<<inv_bias_matrix<<'\n';

    circuits[i] = std::make_shared<Circuit>(circuit_comps, bias_0_comp, inv_bias_matrix);
    i++;
  }

  //____________________________________________________________________________

  //---------------Object proparties--------------------------------------------
  for(auto i=0; i<num_objs; ++i)
  {
    obj[i]->compute_interpolated_charge(phi);
    obj[i]->vertices();
  }
  //____________________________________________________________________________

  //-------------------------Coordinates----------------------------------------
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
  //___________________________Coordinates______________________________________

  //----------------------create facet function---------------------------------
  //             mark facets and mark cells adjacent to object
  //----------------------------------------------------------------------------
  auto num_objects = obj.size();
  auto facet_func = markers(mesh, obj);
  auto cell_func = std::make_shared<CellFunction<std::size_t>>(mesh);
  cell_func->set_all(num_objects);
  for (int i = 0; i < num_objects; ++i)
  {
    obj[i]->mark_cells(cell_func, facet_func, i);
  }
  //____________________________________________________________________________

  //-------------------------Plot solution--------------------------------------
  plot(facet_func);
  plot(cell_func);
  plot(phi);
  plot(E);
  interactive();
  //____________________________________________________________________________

  //------------------ Save solution in VTK format------------------------------
  // File file("poisson.pvd");
  // file << u;
  //____________________________________________________________________________

  return 0;
}

// Calculations for the capacitance matrix for circuits demo:
// Python version:
// [[ 6.03274429 -1.83956002 -1.84411232 -1.65082635]
//  [-1.85130499  4.78829561 -0.04243981 -0.55819532]
//  [-1.85152875 -0.04246471  4.78898192 -0.55827198]
//  [-1.66794035 -0.56310385 -0.55801109  4.89051702]]

// inverse:
// [[ 0.2721993   0.11965458  0.11976778  0.11921181]
//  [ 0.12034576  0.26469355  0.05770902  0.07742289]
//  [ 0.1203438   0.05768568  0.26468632  0.07742199]
//  [ 0.12042337  0.07786823  0.07769313  0.26288379]]

// C++ version:
// capacitance: [[ 6.03274 -1.83956 -1.84411 -1.65083]
//               [-1.8513 4.7883 -0.0424398 -0.558195]
//               [-1.85153 -0.0424647 4.78898 -0.558272]
//               [ -1.66794 -0.563104 -0.558011 4.89052]]

// invverse:
// ((0.272199,0.119655,0.119768,0.119212),
// (0.120346,0.264694,0.057709,0.0774229),
// (0.120344,0.0576857,0.264686,0.077422),
// (0.120423,0.0778682,0.0776931,0.262884))
