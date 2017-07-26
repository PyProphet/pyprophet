# Create manylinux packages from current release using the docker image
#
# based on https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh
#
# Execute as:
# 
#   sudo docker run --net=host -v `pwd`:/data quay.io/pypa/manylinux1_x86_64 /bin/bash /data/create-manylinux.sh
#

git clone https://github.com/PyProphet/pyprophet.git
cd pyprophet
# Apply patch that sets dependency to matplotlib 1.5.3 which does not depend on
# subprocess32 (which cannot be properly installed under CentOS 5).
# See https://github.com/matplotlib/matplotlib/issues/8361
# See https://github.com/google/python-subprocess32/issues/12
git apply /data/manylinux.patch

# Compile wheels
for PYBIN in /opt/python/cp27*/bin /opt/python/cp3[4-9]*/bin; do
  "${PYBIN}/pip" install Cython
  "${PYBIN}/pip" install numpy
  "${PYBIN}/pip" install pandas
  "${PYBIN}/pip" wheel . -w wheelhouse_tmp/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse_tmp/pyprophet*.whl; do
    auditwheel repair "$whl" -w wheelhouse/
done

# upload / store result
mv wheelhouse /data/


