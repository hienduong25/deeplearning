"""Generate a temporary resume YAML configuration from the latest checkpoint state.

Usage:
    python Shadow_Removal/tools/make_resume_yml.py --opt Shadow_Removal/Options/1_Restormer_baseline.yml --state experiments/1_Restormer_baseline/training_states/20000.state
"""
import os
import argparse
import yaml


def make_resume_yml(opt_path, state_path, out_path=None):
    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)

    exp_name = opt['name']
    assert os.path.exists(state_path), f"Checkpoint state file not found: {state_path}"

    opt['path']['resume_state'] = os.path.abspath(state_path)

    if out_path is None:
        out_path = f"Shadow_Removal/Options/temp_resume_{exp_name}.yml"

    with open(out_path, 'w') as f:
        yaml.dump(opt, f, sort_keys=False)

    print(f"Generated resume config: {out_path}")
    print(f"  resume_state set to  : {opt['path']['resume_state']}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Base option yaml file')
    parser.add_argument('--state', type=str, required=True, help='Path to latest .state file')
    parser.add_argument('--out', type=str, default=None, help='Output path for resume yaml')
    args = parser.parse_args()
    make_resume_yml(args.opt, args.state, args.out)
