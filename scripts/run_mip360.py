import os

scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']
data_base_path='/home/zodnguy1/datasets/mipnerf360'
out_base_path='output/mipnerf360'

for id, scene in enumerate(scenes):
    cmd = f'cp -rf {data_base_path}/{scene}/sparse/0/* {data_base_path}/{scene}/sparse/'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"-r{factors[id]} --data_device {data_devices[id]} --densify_abs_grad_threshold 0.0002 --eval"
    cmd = f'python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--skip_train"
    cmd = f'python render.py -m {out_base_path}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f'python metrics.py -m {out_base_path}/{scene}'
    print(cmd)
    os.system(cmd)
