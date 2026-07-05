import argparse
import os
import os.path as osp

import torch
import torchvision
from tqdm import tqdm

# disable built-in parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import build_vae_var
from utils.misc import create_npz_from_sample_folder


HF_HOME = 'https://huggingface.co/FoundationVision/var/resolve/main'


def parse_args():
    parser = argparse.ArgumentParser(description='VAR sampling script')

    # model config
    parser.add_argument('--depth', type=int, default=24, choices=[16, 20, 24, 30],
                        help='VAR model depth. d24 -> FID=2.33, d30 -> FID=1.97')
    parser.add_argument('--vae-ckpt', type=str, default='vae_ch160v4096z32.pth')
    parser.add_argument('--var-ckpt', type=str, default=None,
                        help='path to var_d{depth}.pth. If None, auto set to var_d{depth}.pth')

    # output
    parser.add_argument('--sample-dir', type=str, default='samples',
                        help='directory to save sampled images and class_ids.txt')

    # sampling hyperparameters (defaults match the FID=2.33 recipe for d24)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg', type=float, default=1.5,
                        help='classifier-free guidance ratio. 1.5 is the FID recipe.')
    parser.add_argument('--top-k', type=int, default=900)
    parser.add_argument('--top-p', type=float, default=0.96)
    parser.add_argument('--more-smooth', action='store_true',
                        help='set True for better visual quality but NOT for FID/IS benchmarking')
    parser.add_argument('--bf16', action='store_true',
                        help='use bfloat16 autocast (can be faster on newer GPUs)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='batch size per forward pass')

    # class selection
    parser.add_argument('--class-id-list', type=str, default=None,
                        help='Comma-separated list of ImageNet-1K class IDs, e.g. "4,7,10,25". '
                             'If None, sample ALL 1000 classes (for FID=2.33 evaluation).')
    parser.add_argument('--num-image-per-class', type=int, default=50,
                        help='Number of images to sample per class. 50 is the FID recipe.')

    # npz packing for FID evaluation
    parser.add_argument('--make-npz', action='store_true',
                        help='Pack the sampled PNGs into a single .npz for OpenAI FID eval.')
    parser.add_argument('--npz-expected-count', type=int, default=50_000,
                        help='Expected number of PNGs when packing npz. Set 0 to skip the count check.')

    return parser.parse_args()


def download_checkpoint(ckpt_name: str):
    if not osp.exists(ckpt_name):
        url = f'{HF_HOME}/{ckpt_name}'
        print(f'[download] {ckpt_name} from {url}')
        os.system(f'wget {url} -O {ckpt_name}')


def build_models(args, device):
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.depth, shared_aln=False,
    )
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(args.var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'[model] VAR-d{args.depth} loaded. '
          f'cfg={args.cfg}, top_k={args.top_k}, top_p={args.top_p}, more_smooth={args.more_smooth}')
    return vae, var


def sample_one_batch(var, label_B, seed, cfg, top_k, top_p, more_smooth, bf16, device):
    B = label_B.shape[0]
    dtype = torch.bfloat16 if bf16 else torch.float16
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=dtype, cache_enabled=True):
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B, label_B=label_B, cfg=cfg,
                top_k=top_k, top_p=top_p, g_seed=seed, more_smooth=more_smooth,
            )
    return recon_B3HW


def main():
    args = parse_args()
    os.makedirs(args.sample_dir, exist_ok=True)

    # resolve checkpoints
    if args.var_ckpt is None:
        args.var_ckpt = f'var_d{args.depth}.pth'
    download_checkpoint(args.vae_ckpt)
    download_checkpoint(args.var_ckpt)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda', 'CUDA is required for sampling VAR-d24 at reasonable speed'

    # tf32 for speed
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # build models
    vae, var = build_models(args, device)

    # determine class list
    if args.class_id_list is not None:
        class_ids = [int(x.strip()) for x in args.class_id_list.split(',') if x.strip() != '']
        for cid in class_ids:
            assert 0 <= cid < 1000, f'class id {cid} out of range [0, 1000)'
    else:
        class_ids = list(range(1000))
    num_per_class = args.num_image_per_class
    total_images = len(class_ids) * num_per_class
    print(f'[plan] classes={len(class_ids)}, per_class={num_per_class}, total={total_images}')

    # class_ids.txt path
    class_ids_path = osp.join(args.sample_dir, 'class_ids.txt')

    # resume support: count existing images
    existing_pngs = [f for f in os.listdir(args.sample_dir) if f.endswith('.png')]
    start_idx = len(existing_pngs)
    if start_idx > 0:
        print(f'[resume] found {start_idx} existing pngs in {args.sample_dir}, resuming from index {start_idx}')
        if not osp.exists(class_ids_path):
            print(f'[resume] WARNING: {class_ids_path} not found but {start_idx} pngs exist. '
                  f'class_ids.txt will only record newly sampled images. '
                  f'Recommend cleaning {args.sample_dir} and restarting if you need a complete record.')

    # flatten the work list: each entry is class_id for global_idx
    work_list = []
    for cid in class_ids:
        for _ in range(num_per_class):
            work_list.append(cid)
    assert len(work_list) == total_images

    if start_idx >= total_images:
        print(f'[done] all {total_images} images already exist in {args.sample_dir}')
        return

    # open class_ids.txt: append if resuming, otherwise write fresh
    write_mode = 'a' if start_idx > 0 else 'w'
    class_ids_file = open(class_ids_path, write_mode)

    bs = args.batch_size
    pbar = tqdm(range(start_idx, total_images, bs), desc='Sampling')
    for batch_start in pbar:
        batch_end = min(batch_start + bs, total_images)
        cur_bs = batch_end - batch_start
        batch_classes = work_list[batch_start:batch_end]

        # each image in the batch gets a unique seed = base_seed + global_idx
        # we pass the seed of the first image in the batch to autoregressive_infer_cfg;
        # the per-image diversity comes from different label_B and the multinomial sampler.
        # For stricter per-image reproducibility, sample one image at a time (bs=1).
        seed = args.seed + batch_start
        label_B = torch.tensor(batch_classes, device=device, dtype=torch.long)

        recon_B3HW = sample_one_batch(
            var, label_B=label_B, seed=seed,
            cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
            more_smooth=args.more_smooth, bf16=args.bf16, device=device,
        )

        # save images and record class ids
        for i in range(cur_bs):
            global_idx = batch_start + i
            img = recon_B3HW[i]
            img_name = f'{global_idx:06d}.png'
            img_path = osp.join(args.sample_dir, img_name)
            torchvision.utils.save_image(img, img_path)
            class_ids_file.write(f'{global_idx:06d} {batch_classes[i]}\n')
        class_ids_file.flush()

    class_ids_file.close()
    print(f'[done] saved {total_images} images to {args.sample_dir}')
    print(f'[done] class_ids.txt written to {class_ids_path}')

    # optionally pack into npz for FID evaluation
    if args.make_npz:
        print(f'[npz] packing PNGs in {args.sample_dir} into .npz ...')
        npz_path = create_npz_from_sample_folder(args.sample_dir, expected_count=args.npz_expected_count)
        print(f'[npz] saved to {npz_path}')


if __name__ == '__main__':
    main()
