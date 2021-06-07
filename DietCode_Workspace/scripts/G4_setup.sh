#!/bin/bash

LLVM_VERSION=12

wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
        apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-${LLVM_VERSION} main"

apt-get update && \
apt-get install -y --no-install-recommends \
        llvm-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev clang-${LLVM_VERSION} && \
ln -s /usr/lib/llvm-${LLVM_VERSION} /usr/lib/llvm

apt-get install -y --no-install-recommends \
        libtinfo-dev libedit-dev libxml2-dev zlib1g-dev \
        python3.8 python3.8-dev

pip3 install numpy scipy decorator attrs psutil typed_ast \
             cython six xgboost tornado networkx pytest synr
