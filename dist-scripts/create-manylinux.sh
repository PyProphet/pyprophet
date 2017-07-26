# Create manylinux packages from current release using the docker image

# based on https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh

# sudo docker run --net=host -v `pwd`:/data quay.io/pypa/manylinux1_x86_64 /bin/bash /data/create-manylinux.sh

git clone https://github.com/PyProphet/pyprophet.git
cd pyprophet
git apply /data/manylinux.patch

# Compile wheels
for PYBIN in /opt/python/*27*/bin; do
  "${PYBIN}/pip" install Cython
  "${PYBIN}/pip" install numpy
  "${PYBIN}/pip" install pandas
  "${PYBIN}/pip" wheel . -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/pyprophet*.whl; do
    auditwheel repair "$whl" -w wheelhouse_final/
done

# upload / store result
mv wheelhouse_final /data/
## /opt/python/cp27-cp27mu/bin/pip install twine
## /opt/python/cp27-cp27mu/bin/twine upload wheelhouse_final/* -u hroest


