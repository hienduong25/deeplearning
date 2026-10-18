"""Test verifying configuration consistency:
1. sum(iters) == total_iter == sum(periods) == 60000 across all 5 training option files.
2. Identical common fields (seed=100, val_freq=5000, save_checkpoint_freq=5000, tb_logger=False, data paths).
3. Pairwise ablation verification: each ablation step differs ONLY by the intended keys.
"""
import os
import yaml


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def test_configs_consistency():
    print("Testing config consistency across 5 training configurations...")

    configs = [
        'Shadow_Removal/Options/1_Restormer_baseline.yml',
        'Shadow_Removal/Options/2_Restormer_fft.yml',
        'Shadow_Removal/Options/3_DocShadowNetV2_fp_freqavg.yml',
        'Shadow_Removal/Options/4_DocShadowNetV2_hf.yml',
        'Shadow_Removal/Options/5_DocShadowNetV2_full.yml',
    ]

    loaded_opts = []

    for cfg_path in configs:
        assert os.path.exists(cfg_path), f"File missing: {cfg_path}"
        with open(cfg_path, 'r') as f:
            opt = yaml.safe_load(f)
        loaded_opts.append((cfg_path, opt))

        total_iter = opt['train']['total_iter']
        iters = opt['datasets']['train']['iters']
        periods = opt['train']['scheduler']['periods']
        gt_sizes = opt['datasets']['train']['gt_sizes']
        mini_batch_sizes = opt['datasets']['train']['mini_batch_sizes']

        sum_iters = sum(iters)
        sum_periods = sum(periods)

        print(f"\nChecking: {cfg_path}")
        print(f"  total_iter       : {total_iter}")
        print(f"  iters            : {iters} -> sum = {sum_iters}")
        print(f"  periods          : {periods} -> sum = {sum_periods}")
        print(f"  gt_sizes         : {gt_sizes}")
        print(f"  mini_batch_sizes : {mini_batch_sizes}")

        # Assertions required by user
        assert sum_iters == total_iter, f"sum(iters) {sum_iters} != total_iter {total_iter}"
        assert sum_periods == total_iter, f"sum(periods) {sum_periods} != total_iter {total_iter}"
        assert total_iter == 60000, f"Expected total_iter=60000, got {total_iter}"
        assert gt_sizes == [128, 160, 192, 256], f"gt_sizes mismatch: {gt_sizes}"
        assert mini_batch_sizes == [16, 10, 8, 4], f"mini_batch_sizes mismatch: {mini_batch_sizes}"
        assert iters == [20000, 14000, 14000, 12000], f"iters mismatch: {iters}"
        assert periods == [20000, 40000], f"periods mismatch: {periods}"

        # Common fields assertions
        assert opt['manual_seed'] == 100, f"manual_seed must be 100, got {opt['manual_seed']}"
        assert int(float(opt['val']['val_freq'])) == 5000, f"val_freq must be 5000"
        assert int(float(opt['logger']['save_checkpoint_freq'])) == 5000, f"save_checkpoint_freq must be 5000"
        assert opt['logger']['use_tb_logger'] is False, f"use_tb_logger must be False"
        assert opt['datasets']['train']['dataroot_gt'] == '/content/data/RDD_data/train/gt'
        assert opt['datasets']['train']['dataroot_lq'] == '/content/data/RDD_data/train/img'
        assert opt['datasets']['val']['dataroot_gt'] == '/content/data/RDD_data/val/gt'
        assert opt['datasets']['val']['dataroot_lq'] == '/content/data/RDD_data/val/img'
        assert opt['path']['pretrain_network_g'] is None, "Must train from scratch (no pretrain)"
        assert opt['path']['resume_state'] is None, "Resume state must be null in master configs"

        print(f"  -> Schedule & common fields: OK")

    # Assert fft_opt in configs 2, 3, 4, 5
    for idx in [1, 2, 3, 4]:
        opt = loaded_opts[idx][1]
        assert 'fft_opt' in opt['train'], f"fft_opt missing in config {idx + 1}"
        assert opt['train']['fft_opt']['type'] == 'FFTLoss'
        assert float(opt['train']['fft_opt']['loss_weight']) == 0.1

    print("\n--- Pairwise Ablation Consistency Checks ---")
    opt1 = flatten_dict(loaded_opts[0][1])
    opt2 = flatten_dict(loaded_opts[1][1])
    opt3 = flatten_dict(loaded_opts[2][1])
    opt4 = flatten_dict(loaded_opts[3][1])
    opt5 = flatten_dict(loaded_opts[4][1])

    # 1 -> 2: only differs by name and train.fft_opt.*
    diff_1_2 = set(k for k in set(opt1) | set(opt2) if opt1.get(k) != opt2.get(k))
    expected_diff_1_2 = {'name', 'train.fft_opt.type', 'train.fft_opt.loss_weight', 'train.fft_opt.reduction'}
    assert diff_1_2 == expected_diff_1_2, f"Diff 1->2 mismatch: {diff_1_2} vs {expected_diff_1_2}"
    print("  Config 1 -> Config 2 (+FFT): exactly only added FFTLoss: OK")

    # 2 -> 3: only differs by name and network_g architecture flags (FP and FreqAvgUp)
    diff_2_3 = set(k for k in set(opt2) | set(opt3) if opt2.get(k) != opt3.get(k))
    expected_diff_2_3 = {
        'name', 'network_g.type', 'network_g.down_mode', 'network_g.up_mode',
        'network_g.use_hf', 'network_g.reduce_top', 'network_g.out_mode'
    }
    assert diff_2_3 == expected_diff_2_3, f"Diff 2->3 mismatch: {diff_2_3}"
    print("  Config 2 -> Config 3 (+FP & +FreqAvgUp): exactly only changed down_mode='fp' & up_mode='freqavg': OK")

    # 2 -> 4: only differs by name and network_g architecture flags (HF only)
    diff_2_4 = set(k for k in set(opt2) | set(opt4) if opt2.get(k) != opt4.get(k))
    expected_diff_2_4 = {
        'name', 'network_g.type', 'network_g.down_mode', 'network_g.up_mode',
        'network_g.use_hf', 'network_g.reduce_top', 'network_g.out_mode'
    }
    assert diff_2_4 == expected_diff_2_4, f"Diff 2->4 mismatch: {diff_2_4}"
    print("  Config 2 -> Config 4 (+HF only): exactly only changed use_hf=True: OK")

    # 3 -> 5: only differs by name and network_g.use_hf (adding HF to FP+FreqAvgUp)
    diff_3_5 = set(k for k in set(opt3) | set(opt5) if opt3.get(k) != opt5.get(k))
    expected_diff_3_5 = {'name', 'network_g.use_hf'}
    assert diff_3_5 == expected_diff_3_5, f"Diff 3->5 mismatch: {diff_3_5}"
    print("  Config 3 -> Config 5 (+HF addition): exactly only changed use_hf: OK")

    # 4 -> 5: only differs by name and network_g.down_mode / up_mode (adding FP+FreqAvgUp to HF)
    diff_4_5 = set(k for k in set(opt4) | set(opt5) if opt4.get(k) != opt5.get(k))
    expected_diff_4_5 = {'name', 'network_g.down_mode', 'network_g.up_mode'}
    assert diff_4_5 == expected_diff_4_5, f"Diff 4->5 mismatch: {diff_4_5}"
    print("  Config 4 -> Config 5 (+FP & +FreqAvgUp addition): exactly only changed down_mode & up_mode: OK")

    print("\n" + "=" * 60)
    print("ALL 5 CONFIGURATIONS ARE STRICTLY CONSISTENT AND VERIFIED!")
    print("=" * 60)


if __name__ == '__main__':
    test_configs_consistency()
