all: empty_square.xml empty_cube.xml

empty_square.xml: empty_square.msh
	dolfin-convert empty_square.msh empty_square.xml

empty_square.msh: empty_square.geo
	gmsh -2 empty_square.geo

empty_cube.xml: empty_cube.msh
	dolfin-convert empty_cube.msh empty_cube.xml

empty_cube.msh: empty_cube.geo
	gmsh -3 empty_cube.geo
