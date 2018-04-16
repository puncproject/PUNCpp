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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <stdio.h>
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
};

int main()
{
    // double e = boost::units::si::constants::codata::e / boost::units::si::coulomb;
    // double m_e = boost::units::si::constants::codata::m_e/ boost::units::si::kilograms;
    // double ratio = boost::units::si::constants::codata::m_e_over_m_p/ boost::units::si::dimensionless();
    // double m_i = m_e/ratio;

    // double k_B = boost::units::si::constants::codata::k_B*boost::units::si::kelvin/boost::units::si::joules;
    // double eps0 = boost::units::si::constants::codata::epsilon_0*boost::units::si::meter/boost::units::si::farad;

    PhysicalConstants constants;
    printf("e: %e\n",constants.e);
    printf("m_e: %e\n", constants.m_e);
    printf("m_e_over_m_p: %e\n", constants.ratio);
    printf("ratio: %e\n", 1. / constants.ratio);
    printf("m_i: %e\n", constants.m_i);
    printf("kB: %e\n", constants.k_B);
    printf("eps0: %e\n", constants.eps0);

    return 0;
}
