# GMS

GMS is a graph neural network model which can learn to predict the solution of MaxSAT problem, an optimization variant of the well-known Boolean Satisfiability problem (SAT).

### Usage
We published the sourse code of GMS, as well as the problem generation, training and testing scripts.

1. Clone this repo to your machine
```shell
	git clone https://github.com/minghao-liu/GMS.git
```
We only tested this project on a server with NVIDIA Tesla V100 GPU, and PyTorch 1.5.0.

2. Setup
```shell
	chmod 755 *.sh
	./setup.sh
```
Create the required folders, and download the problem generator [[link]](https://github.com/RalfRothenberger/Power-Law-Random-SAT-Generator), as well as the baseline MaxSAT solvers we use: MaxHS, Loandra and SATLike.

3. Generate raw data
```shell
	 ./generate_raw_data.sh
```
The generated problems are in the standard DIMACS format.

4. Generate data
```shell
	 ./generate_data.sh
```
The data files will be stored in the `data` folder, which can be loaded by GMS.

5. Train the GMS models
```shell
	 ./train.sh
```
The trained model will be stored in the `model` folder.
