# train.sh

python src/train.py \
  --task-name 'GMS_N_datav1' \
  --model 'GMS_N' \
  --n_vars 60 \
  --epochs 10 \
  --n_rounds 30 \
  --train-file 'trainset_v1_bs20000_nb3.pkl' \
  --val-file 'valset_v1_bs20000_nb1.pkl'
