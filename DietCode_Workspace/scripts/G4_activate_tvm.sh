PYTHONPATH_BAK=${PYTHONPATH}
LD_LIBRARY_PATH_BAK=${LD_LIBRARY_PATH}

TVM_ROOT=$(cd $(dirname $BASH_SOURCE[0])/../../.. && pwd)/tvm

export PYTHONPATH=\
${PYTHONPATH}${TVM_ROOT}/python/build/lib.linux-x86_64-3.8/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}${TVM_ROOT}/build

export PATH=${PATH}:/usr/local/cuda/bin
