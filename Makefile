# ==============================================================================
# Walicxe 3D Makefile
# ==============================================================================

# Name of the compiled binary
PROGRAM= raytracing_test

# Non-MPI Compiler
# Supported options: gfortran, ifort
# For MPI, mpfi90 is used
COMPILER= ifort
#COMPILER= gfortran

# Additional user compiler flags
## ifort-compatible flags
#USER_FLAGS= -O2 -warn nounused -nogen-interfaces
#USER_FLAGS= -traceback -warn all -check all,noarg_temp_created -nogen-interfaces
## gfortran-compatible flags
#USER_FLAGS= -Wall -pedantic -fbounds-check -g -fbacktrace
#USER_FLAGS= -g -fbacktrace
## Generic flags
USER_FLAGS= -O3

# ============================================= #
# Compilation Time Parameters (Y=on, N=off)     #
# Warning: the following is all case sensitive! #
# ============================================= #

# Use MPI parallelization?
MPI= Y

# Use double precision for all real variables?
DOUBLEP= Y

# Enable passive magnetic field?
PASB= N

# ==============================================================================

# Additional USER modules to compile
MODULES_USER= \


# Independently compilable modules in the MAIN CODE
MODULES_MAIN= \
./source/constants.o \
./source/parameters.o \
./source/globals.o \
./source/snr.o \
./source/winds.o \
./source/orbits.o \
./source/user.o

# Dependent source files and modules in the MAIN CODE
OBJECTS_MAIN= \
./source/initflow.o \
./source/tictoc.o \
./source/admesh.o \
./source/initmain.o \
./source/basegrid.o \
./source/boundary.o \
./source/loadbalance.o \
./source/hilbert.o \
./source/deinit.o \
./source/output.o \
./source/utils.o \
./source/cut.o \
./source/prims.o \
./source/hydro.o \
./source/godunov.o \
./source/lax.o \
./source/hll.o \
./source/hllc.o \
./source/cooling.o \
./source/warmstart.o \
./source/report.o \
./source/main.o

# List of modules and objects to compile the Column Density facility
OBJECTS_COLDENS= \
./source/constants.o \
./source/utils.o \
./source/orbits.o \
./source/coldens.o

OBJECTS_RAYTRACING= \
./source/constants.o \
./source/utils.o \
./source/orbits.o \
./source/raytracing.o \
#./source/Ray_tracing.o \
# ./source/parameters.o \
# ./source/utils.o \
# ./source/globals.o \
# ./source/tictoc.o \
# ./source/deinit.o \
# ./source/prims.o \
# ./source/hilbert.o \
# ./source/loadbalance.o \
# ./source/admesh.o \
# ./source/coldens.o \




# ./source/basegrid.o \

# List of modules and objects to compile the Data Extractor facility
OBJECTS_EXTRACT= \
./source/constants.o \
./source/utils.o \
./source/parameters.o \
./source/extract.o \

# ==============================================================================
# THERE SHOULD BE NO NEED TO MODIFY BELOW THIS LINE
# ==============================================================================

# Build compiler flags
CFLAGS = $(USER_FLAGS) -cpp

# MPI
ifeq ($(MPI),Y)
CFLAGS += -DMPIP
endif

# Double precision (compiler dependent)
ifeq ($(DOUBLEP),Y)
CFLAGS += -DDOUBLEP
ifeq ($(COMPILER),ifort)
CFLAGS += -r8
endif
ifeq ($(COMPILER),gfortran)
CFLAGS += -fdefault-real-8
endif
endif

# MPI
ifeq ($(PASB),Y)
CFLAGS += -DPASBP
endif

# Set mpif90 as compiler if MPI, otherwise use specified
ifeq ($(MPI),Y)
COMPILER = mpif90
endif

# Join object lists
OBJECTS_ALL = ${MODULES_MAIN} ${MODULES_USER} ${OBJECTS_MAIN} 

# ==============================================================================

$(PROGRAM) : prebuild ${OBJECTS_ALL}
	@echo Linking object files ...
	@$(COMPILER) $(CFLAGS) $(OBJECTS_ALL) -o $@
	@echo Cleaning up ...
	@rm -f *.o *.mod source/*.o source/*.mod
	@echo "Done! (`date`)"

raytracing_c : prebuild $(OBJECTS_RAYTRACING)
	@echo Linking object files ...
	@$(COMPILER) $(CFLAGS) $(OBJECTS_RAYTRACING) -o $@
	@echo Cleaning up ...
	@rm -f *.o *.mod source/*.o source/*.mod
	@echo "Done! (`date`)"

coldens : prebuild $(OBJECTS_COLDENS)
	@echo Linking object files ...
	@$(COMPILER) $(CFLAGS) $(OBJECTS_COLDENS) -o $@
	@echo Cleaning up ...
	@rm -f *.o *.mod source/*.o source/*.mod
	@echo "Done! (`date`)"

extract : prebuild $(OBJECTS_EXTRACT)
	@echo Linking object files ...
	@$(COMPILER) $(CFLAGS) $(OBJECTS_EXTRACT) -o $@
	@echo Cleaning up ...
	@rm -f *.o *.mod source/*.o source/*.mod
	@echo "Done! (`date`)"

prebuild :
	@echo "Walicxe3D build started `date`"

%.o : %.f90
	@echo Compiling $^ ...
	@$(COMPILER) $(CFLAGS) -c $^ -o $@

clean : 
	rm -f *.o *.mod source/*.o source/*.mod
	rm -f $(PROGRAM).*

cleanall :
	rm -f *.o *.mod source/*.o source/*.mod
	rm -f $(PROGRAM)
	rm -f $(PROGRAM).*
	rm -f coldens
	rm -f extract
	rm -f data/*.bin
	rm -f data/*.vtk
	rm -f data/*.dat
	rm -f data/*.log
	rm -f data/*.visit
	rm -f data/*.rg
