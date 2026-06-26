# CSVSUF-3stg
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 2 --outf "./experiments/CSVSUF-3stg/" --model "CSVSUF" --num_iter 3

# CSVSUF-6stg
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 2 --outf "./experiments/CSVSUF-6stg/" --model "CSVSUF" --num_iter 6

# CSVSUF-9stg
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 2 --outf "./experiments/CSVSUF-9stg/" --model "CSVSUF" --num_iter 9
