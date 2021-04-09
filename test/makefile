sphere.xml: sphere.msh
	dolfin-convert sphere.msh sphere.xml

sphere.msh: sphere.geo
	gmsh -3 -format msh2 sphere.geo

clean:
	rm -f sphere.msh sphere*.xml
