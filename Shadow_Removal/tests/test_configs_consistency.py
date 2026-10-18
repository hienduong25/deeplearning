"""Test verifying configuration consistency:
sum(iters) == total_iter == sum(periods)
across all 5 training option files.
"""
import os
import yaml


def test_configs_consistency():
    print("Testing config consistency across 5 training configurations...")

    configs = [
        'Shadow_Removal/Options/1_Restormer_baseline.yml',
        'Shadow_Removal/Options/2_Restormer_fft.yml',
        'Shadow_Removal/Options/3_DocShadowNetV2_fp_freqavg.yml',
        'Shadow_Removal/Options/4_DocShadowNetV2_hf.yml',
        'Shadow_Removal/Options/5_DocShadowNetV2_full.yml',
    ]

    for cfg_path in configs:
        assert os.path.exists(cfg_path), f"File missing: {cfg_path}"
        with open(cfg_path, 'r') as f:
            opt = yaml.safe_load(f)

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

        print(f"  -> sum(iters) == total_iter == sum(periods) == 60000: OK")

    print("\n" + "=" * 60)
    print("ALL 5 CONFIGURATIONS ARE STRICTLY CONSISTENT AND VERIFIED!")
    print("=" * 60)


if __name__ == '__main__':
    test_configs_consistency()
