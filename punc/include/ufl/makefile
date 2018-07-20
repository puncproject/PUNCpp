# To generate a .h file for every .ufl file in this folder:
#
#   make
#
# Already existing .h files will not be created again unless the corresponding
# .ufl file is newer. To delete all .h files:
#
#   make clean
#

UFL_FILES=$(shell ls *.ufl)
H_FILES = $(patsubst %.ufl,%.h,$(UFL_FILES))

all: $(H_FILES)

%.h: %.ufl
	ffc -O -l dolfin $<

clean:
	rm -f *.h
