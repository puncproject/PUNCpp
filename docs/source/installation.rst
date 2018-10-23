Installation
============

The following dependencies must be installed before installing PUNC++:

- FEniCS_ 2018.1.0
- CMake_
- Doxygen_
- Git_

.. _FEniCS: https://fenicsproject.org
.. _CMake: https://cmake.org
.. _Doxygen: http://www.doxygen.org
.. _Git: https://git-scm.com

In addition, FEniCS must be compiled with *at least* the following optional dependencies:

- hdf5_
- hypre_
- PETSc_

.. _hdf5: https://support.hdfgroup.org/HDF5/
.. _hypre: https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods
.. _PETSc: http://www.mcs.anl.gov/petsc/

While FEniCS is a very powerful problem solving environment for finite elements, it has proven notoriously troublesome to install, and we do not have the capacity to offer much guidance here. Since FEniCS and its dependencies change functionality frequently, we recommend using the version listed on this page. Several installation methods are described on the FEniCS website. We point out that depending on how FEniCS and its dependencies are installed, there may be a tenfold difference in the performance of PUNC++, since not all installations have the same level of optimization. The Anaconda_ version yields good performance, but make sure the folder containing ``libdolfin.so`` is accesible to the compiler, e.g. by exporting it to ``LD_LIBRARY_PATH``.

.. _Anaconda: https://anaconda.org/conda-forge/fenics

Once you have installed all dependencies, download PUNC++::

    cd ~ # or the loaction of your choice
    git clone https://github.com/puncproject/PUNCpp.git punc++
    cd punc++

PUNC++ comes in two parts: a library (in the folder ``punc++/punc``) and an executable  program (in ``punc++/interaction``). The library contains every computational aspects in a modular fashion, and makes it easy to re-use bits and pieces in different ways and to perform custom tests and benchmarks on individual parts of the library. The executable makes use of the library to perform plasma-object interaction simulations. To build the library::

    cd punc
    ./build.sh # or ./install.sh

The library will be installed in a local folder ``punc++/punc/install``. It can also be installed in system-wide folders by running ``./install.sh`` instead of ``./build.sh``. For developers, the documentation of the code can be installed by running::

    ./doc.sh

and launching ``doc.html``. Next, to install ``interaction``::

    cd ../interaction
    ./build.sh

That's it! In addition we recommend installing the following tools for pre- and post-processing:

- Gmsh_
- ParaView_
- Metaplot_

.. _Gmsh: http://gmsh.info/
.. _ParaView: https://www.paraview.org/
.. _Metaplot: https://metaplot.readthedocs.io

The tutorials in this user guide will make use of them.
