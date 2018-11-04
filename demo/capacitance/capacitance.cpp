// Copyright (C) 2018, Diako Darian and Sigvald Marholm
//
// This file is part of PUNC++.
//
// PUNC++ is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// PUNC++ is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// PUNC++. If not, see <http://www.gnu.org/licenses/>.

/**
 * @file		capacitance.cpp
 * @brief		Demo for calculating the capacitance of two concentric spheres
 *
 * This demo shows how to obtain the capacitance of two concentric spheres. The
 * space between the spheres is assumed to be empty. 
 */

#include <punc.h>

using namespace punc;

int main()
{
	df::set_log_level(df::WARNING);

	PhysicalConstants constants;
	double eps0 = 1.0; //constants.eps0;

	/***************************************************************************
	 * ANALYTICAL EXPRESSION FOR THE CAPACITANCE OF TWO CONCENTRIC SPHERES
	 **************************************************************************/
	auto r = 0.2;
	auto R = 1.0;
	auto cap = 4. * M_PI * eps0 * r * R / (R - r);

	/***************************************************************************
	 * CREATE THE MESH
	 **************************************************************************/
	std::string fname{"sphere_in_sphere"};
	Mesh mesh(fname);

	/***************************************************************************
	 * CREATE THE FUNCTION SPACE
	 **************************************************************************/
	auto V = CG1_space(mesh);

	/***************************************************************************
	 * CREATE THE OBJECT
	 **************************************************************************/
	std::vector<std::shared_ptr<Object>> objects;
	objects.push_back(std::make_shared<ObjectCM>(V, mesh, 2));

	/***************************************************************************
	 * CALCULATE THE CAPACITANCE AND PRINT THE RESULT
	 **************************************************************************/
	boost_matrix inv_capacity = inv_capacitance(V, objects, mesh, eps0);

	auto error = (cap - 1. / inv_capacity(0, 0)) * inv_capacity(0, 0);
	printf("Analatical value: %f, numerical value: %f, error: %f\n", cap, 1. / inv_capacity(0, 0), error);

	return 0;
}
