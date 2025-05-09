import os
from pathlib import Path

scene_dense_dir = Path("/home/zodnguy1/datasets/zeiss/machine-dense")
scene_sparse_dir = Path("/home/zodnguy1/datasets/zeiss/machine-sparse")

cmd = f"python train.py -s {scene_sparse_dir} -m output/zeiss/machine-sparse -r 2 --ncc_scale 0.5"
os.system(cmd)

cmd = f"python render.py -m output/zeiss/machine-sparse --num_cluster 1 --voxel_size 0.001 --max_depth 3"
os.system(cmd)

cmd = f"python train.py -s {scene_dense_dir} -m output/zeiss/machine-dense -r 2 --ncc_scale 0.5"
os.system(cmd)

cmd = f"python render.py -m output/zeiss/machine-dense --num_cluster 1 --voxel_size 0.001 --max_depth 3"
os.system(cmd)

cmd = f"python train.py -s {scene_sparse_dir} -m output/zeiss/machine-sparse-mask -r 2 --ncc_scale 0.5 --use_mask"
os.system(cmd)

cmd = f"python render.py -m output/zeiss/machine-sparse-mask --num_cluster 1 --voxel_size 0.001 --max_depth 3"
os.system(cmd)

cmd = f"python train.py -s {scene_dense_dir} -m output/zeiss/machine-dense-mask -r 2 --ncc_scale 0.5 --use_mask"
os.system(cmd)

cmd = f"python render.py -m output/zeiss/machine-dense-mask --num_cluster 1 --voxel_size 0.001 --max_depth 3"
os.system(cmd)