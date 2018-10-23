Tutorial: Basic
===============

(Sigvald Marholm)

In this tutorial, we will carry out a 2D simulation of the current collected by a cylindrical Langmuir probe in non-drifting, non-magnetized, Maxwellian proton-electron plasma. The physical parameters of the problem is given below:

=====================     =========================================================
Parameter                 Value
=====================     =========================================================
Ion/electron density      :math:`10\cdot 10^{10}\,\mathrm{m^{-3}}`
Ion/electron temperature  :math:`0.26\,\mathrm{eV}` (:math:`\sim 3014\,\mathrm{K}`)
Electron Debye length     :math:`\sim 12\,\mathrm{mm}`
Probe radius              :math:`2\,\mathrm{mm}`
Probe voltage             :math:`3\,\mathrm{V}`
=====================     =========================================================

We start by making a new directory for the project, and make a symbolic link of the ``interaction`` executable in this folder (modify paths as necessary)::

    cd ~
    mkdir -p Projects/tutorials/basic
    cd Projects/tutorials/basic
    ln -s ~/punc++/interaction/build/interaction

Mesh generation
---------------
The first thing we need is a mesh of the simulation domain including any objects. 
For a 2D simulation of a cylindrical probe this is just two concentric circles.
The inner circle, or interior boundary, is the probe itself.
The outer circle is the exterior boundary of the simulation domain.
It is an assumption of the underlying numerical methods that the exterior boundary is in the background plasma and is not affected by any local perturbations in the electric potential.
This makes it necessary for the outer boundary to be outside the sheath of the probe.
The closer the outer boundary is to the probe, the less valid the assumption becomes, and the more it limits the accuracy of the simulation.
On the other hand, increasing the domain size increases the cost of the simulation.
The mesh gets bigger, and you need more simulation particles to maintain the same density of particles in the domain.
You may in fact also need to run the simulation for a longer physical time in order to reach a sufficiently steady state.
This is because the domain is uniformly filled with particles before the simulation starts, and you get rather violent, unphysical transient behaviour where the plasma is perturbed, e.g. where the sheath should be.
These transients may cause waves which takes a long time to leave the domain.
Moreover, in improving the accuracy of a simulation, the clue is to improve the limiting factor, which may or may not be the radius of the outer boundary.
Thus, finding the right set of simulation parameters requires some experimentation.
We will use an outer boundary of 100mm radius.

This geometry can be created either using the Gmsh *Graphical User Interface* (GUI), or it can be specified directly in a text file for Gmsh to read.
We shall call our geometry file ``cylinder.geo`` and it looks as follows:

.. literalinclude:: ../../../tutorials/basic/cylinder.geo

On the first four lines we define variables for the inner and outer radii, as well as the resolution of the mesh on the inner and outer boundaries.
The resolution must be sufficiently fine to resolve the characteristic lengths of the physical processes involved.
This means that it must be fine enough to resolve the probe on the inner boundary, and not much bigger than the electron Debye length on the outer boundary.
Insufficiently resolving the electron Debye length is known to cause numerical heating of the plasma on rectangular meshes (see Birdsall&Langdon), and it is reasonable to expect simlar effects for unstructured meshes.
Since we are most interested in what happens close to the probe, we can often get away with a resolution that is actually somewhat bigger than the Debye length at the outer boundary.

Next, we add a point in the center of the domain, as well as north, east, south and west of the center on each boundary, and connect them by circle arcs. (It may be convenient to define the variables and add the points in text, and use the GUI to do the rest).

It is important that the proper *Physical Groups* are created in Gmsh. PUNC++ needs one physical group for each boundary (*Physical Lines* in 2D or *Physical Surfaces* in 3D).
Note that it is *mandatory* that the exterior boundary has the lowest id number of all boundaries since this is how PUNC++ knows which boundary is the exterior one.
It is also possible to define a physical group for the domain (*Physical Surface* in 2D or *Physical Volume* in 3D), but this is optional.

To generate the mesh from the geometry, run ``Mesh -> 2D`` from the GUI or from the terminal::

    $ gmsh -2 cylinder.geo 

The mesh is named ``cylinder.msh`` and should look something like this when opened in Gmsh:

.. image:: mesh.png

Mesh files must be in either DOLFIN XML or HDF5 formats.
To convert the Gmsh mesh, use FEniCS's own conversion tool::

    $ dolfin-convert cylinder.msh cylinder.xml

You should now have the files ``cylinder.xml``, ``cylinder_facet_region.xml`` and possibly also ``cylinder_physical_region.xml``. The first file is the mesh itself, whereas the latter two contain the physical groups for the boundaries and the domain, respectively.

Running the simulation
----------------------

Inspecting the results
----------------------
