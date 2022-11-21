# setup.sh

if [ ! -d "log" ]; then
    mkdir log
fi
if [ ! -d "model" ]; then
    mkdir model
fi
if [ ! -d "data" ]; then
    mkdir data
fi
if [ ! -d "utils" ]; then
    mkdir utils
fi
cd utils

# install the problem generator
git clone https://github.com/RalfRothenberger/Power-Law-Random-SAT-Generator.git
cd Power-Law-Random-SAT-Generator
make
cd ..

# install baseline MaxSAT solvers
# MaxHS
wget https://maxsat-evaluations.github.io/2021/mse21-solver-src/complete/maxhs.zip
unzip maxhs.zip
rm maxhs.zip
# Loandra
wget https://maxsat-evaluations.github.io/2021/mse21-solver-src/incomplete/Loandra-2020.zip
unzip Loandra-2020.zip
rm Loandra-2020.zip
# SATLike
wget https://lcs.ios.ac.cn/~caisw/Code/maxsat/SATLike_v2018.zip
unzip SATLike_v2018.zip
chmod 755 SATLike_v2018/unweightedPMS/SATLike/bin/*
rm SATLike_v2018.zip
