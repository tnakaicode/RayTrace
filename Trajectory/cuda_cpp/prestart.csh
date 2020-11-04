# Please note that this is meant to work together with TCSH (c-shell) for 
# setenv, and a valid CUDA installation.
# the variable gode_src should be set for each box, uncomment the one you need
#--------------------------------#
# this one is for zorn@pdc
setenv gode_src /afs/pdc.kth.se/home/d/dmitra/research/codes/python/pyoden/cuda_cpp/src 
#--------------------------------#
setenv rhs_source particle_in_mag.cu
setenv rhs_header particle_in_mag.h
echo "setting up and compiling ode solve in gpu ..."
echo " first removing exisiting links if any "
rm -f rhs.h
rm -f rhs.cu
rm -f Makefile
rm -f evolve.cu
rm -f gode.h
rm -f gode.cu
echo " The model header and source files are:"
echo $rhs_header
echo $rhs_source
ln -s $gode_src/RHS/$rhs_header  rhs.h
ln -s $gode_src/RHS/$rhs_source  rhs.cu
ln -s $gode_src/gode.h gode.h
echo "...done, now linking other cu files from src director to here.."
ln -s $gode_src/gode.h gode.h
ln -s $gode_src/evolve.cu evolve.cu
ln -s $gode_src/gode.cu  gode.cu
ln -s $gode_src/Makefile Makefile 
echo "....done. now run 'make g.exe' "
