import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='/home/zodnguy1/datasets/dtu'

out_base_path='output/dtu_aware_r5_f1_g7k'
max_abs = [50_000, 50_000, 0, 0, 50_000, 50_000, 50_000, 50_000, 50_000, 50_000, 0, 50_000, 50_000, 50_000, 50_000]

for scene, abs_split in zip(scenes, max_abs):
#     print(f"==> Processing scene: scan{scene} <===")
#     cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
#     print("[>] " + cmd)
#     os.system(cmd)

    common_args = f"-r 2 --ncc_scale 0.5 --max_abs_split_points {abs_split}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = "--num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'python render.py -m {out_base_path}/scan{scene}/ {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/scan{scene}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/scan{scene}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")

out_base_path='output/dtu_aware_r5_f1_g5k'
max_abs = [50_000, 50_000, 0, 0, 50_000, 50_000, 50_000, 50_000, 50_000, 50_000, 0, 50_000, 50_000, 50_000, 50_000]

for scene, abs_split in zip(scenes, max_abs):
#     print(f"==> Processing scene: scan{scene} <===")
#     cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
#     print("[>] " + cmd)
#     os.system(cmd)

    common_args = f"-r 2 --ncc_scale 0.5 --max_abs_split_points {abs_split} --multi_view_weight_from_iter 5000"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = "--num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'python render.py -m {out_base_path}/scan{scene}/ {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/scan{scene}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/scan{scene}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")

out_base_path='output/dtu_same_r5_f1_g5k'
for scene in scenes:
#     print(f"==> Processing scene: scan{scene} <===")
#     cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
#     print("[>] " + cmd)
#     os.system(cmd)

    common_args = f"-r 2 --ncc_scale 0.5 --multi_view_weight_from_iter 5000"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = "--num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'python render.py -m {out_base_path}/scan{scene}/ {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/scan{scene}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/scan{scene}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")
