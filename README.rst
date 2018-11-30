PUNC++
======

*Particles-in-UNstructured-Cells in C++* (PUNC++) is the C++ version of PUNC.

Contributors
------------

Principal authors:

- `Sigvald Marholm`_
- `Diako Darian`_

Contributors and mentors:

- `Mikael Mortensen`_
- `Richard Marchand`_
- `Wojciech J. Miloch`_

.. _`Sigvald Marholm`: mailto:sigvald@marebakken.com
.. _`Diako Darian`: mailto:diakod@math.uio.no
.. _`Mikael Mortensen`: mailto:mikael.mortensen@gmail.com
.. _`Richard Marchand`: mailto:rmarchan@ualberta.ca
.. _`Wojciech J. Miloch`: mailto:w.j.miloch@fys.uio.no

Installation
------------

The following dependencies must be installed prior to using PUNC++:

- FEniCS_ 2018.1.0 (stable)
- CMake_ (for installing)
- Git_ (optional, for installing)

In addition, FEniCS must be compiled with *at least* the following optional dependencies:

- hdf5_
- hypre_
- PETSc_

It is crucial that the correct version of FEniCS is used. For more on installing these dependencies, see their official pages. For Arch Linux, the arch-fenics-packages_ repository can also be used. It contains full installation instructions for FEniCS and its dependencies.

Prior to installing, the PUNC++ repository must be downloaded. We recommend cloning it using Git::

    cd ~ # Or other parent folder
    git clone --recurse-submodules https://github.com/puncproject/PUNCpp.git punc++

Note that the subfolder ```punc/mesh``` is a Git submodule and will be empty if submodules are not initialized. The ```--recurse-submodules``` flag should take care of this (```git submodule update``` can be used at a later point if it is forgotten).

Next, install PUNC++ as follows::

    cd punc++/punc
    mkdir build
    cd build
    cmake ..
    make
    make install

For developers it is convenient to use the script ``punc/install.sh``.

In addition we recommend the following tools for pre- and post-processing:

- Gmsh_
- ParaView_

.. _FEniCS: https://fenicsproject.org
.. _CMake: https://cmake.org
.. _Git: https://git-scm.com
.. _Python: https://www.python.org
.. _TaskTimer: https://github.com/sigvaldm/TaskTimer
.. _arch-fenics-packages: https://github.com/sigvaldm/arch-fenics-packages
.. _petsc4py: https://bitbucket.org/petsc/petsc4py/src/master/
.. _matplotlib: https://matplotlib.org/
.. _hdf5: https://support.hdfgroup.org/HDF5/
.. _hypre: https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods
.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _Gmsh: http://gmsh.info/
.. _ParaView: https://www.paraview.org/

