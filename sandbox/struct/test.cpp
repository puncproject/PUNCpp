#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/systems/si/codata/electron_constants.hpp>
#include <boost/units/systems/si/codata/physico-chemical_constants.hpp>
#include <boost/units/systems/si/codata/universal_constants.hpp>

struct PhysicalConstants
{
	double e = boost::units::si::constants::codata::e / boost::units::si::coulomb;
	double m_e = boost::units::si::constants::codata::m_e / boost::units::si::kilograms;
	double ratio = boost::units::si::constants::codata::m_e_over_m_p / boost::units::si::dimensionless();
	double m_i = m_e / ratio;

	double k_B = boost::units::si::constants::codata::k_B * boost::units::si::kelvin / boost::units::si::joules;
	double eps0 = boost::units::si::constants::codata::epsilon_0 * boost::units::si::meter / boost::units::si::farad;
} constants;

struct Particle
{
	std::vector<double> x;
	std::vector<double> v;
	double q;
	double m;
};

int main(){

	std::vector<double> vs = {0.0,0.0};
	std::vector<double> xs = {1.0,1.0};
	double qs = 1.0;
	double ms = 1.0;

	std::vector<Particle> particles;
	particles.push_back(Particle{xs,vs,qs,ms});
	// Particle particle{xs,vs,qs,ms};

	for(auto& e:particles[0].x)
	{
		std::cout<<e<<"  ";
	}
	std::cout<<'\n';
	std::cout<<"qs: "<<particles[0].q<<'\n';
	std::cout<<"ms: "<<particles[0].m<<'\n';
	std::cout << "kB: " << constants.k_B << '\n';
	return 0;
}
