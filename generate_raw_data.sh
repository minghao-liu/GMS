# generate_raw_data.sh
# Generate random MaxSAT problems of UF and PL distributions,
# and compute the optimal solution by MaxHS if needed.
# Usage: python src/generate_raw_data.py <UF/PL> <K> <NV> <NC> <COUNT> <NEED_SOL> 

python src/generate_raw_data.py UF 2 60 600 20000 1
