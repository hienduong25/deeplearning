"""Test proving train, val, test have ZERO file overlap."""
import os


def test_no_data_leakage(data_dir='./RDD_data'):
    print("Testing data leakage between train, val, test...")
    train_dir = os.path.join(data_dir, 'train', 'img')
    val_dir = os.path.join(data_dir, 'val', 'img')
    test_dir = os.path.join(data_dir, 'test', 'img')

    assert os.path.isdir(train_dir), f"{train_dir} does not exist"
    assert os.path.isdir(val_dir), f"{val_dir} does not exist"
    assert os.path.isdir(test_dir), f"{test_dir} does not exist"

    train_names = set(os.listdir(train_dir))
    val_names = set(os.listdir(val_dir))
    test_names = set(os.listdir(test_dir))

    print(f"  Train images: {len(train_names)}")
    print(f"  Val   images: {len(val_names)}")
    print(f"  Test  images: {len(test_names)}")

    train_val_overlap = train_names & val_names
    train_test_overlap = train_names & test_names
    val_test_overlap = val_names & test_names

    print(f"  Train & Val  overlap: {len(train_val_overlap)}")
    print(f"  Train & Test overlap: {len(train_test_overlap)}")
    print(f"  Val   & Test overlap: {len(val_test_overlap)}")

    assert len(train_val_overlap) == 0, f"DATA LEAK: train & val overlap by {len(train_val_overlap)} files: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"DATA LEAK: train & test overlap by {len(train_test_overlap)} files: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"DATA LEAK: val & test overlap by {len(val_test_overlap)} files: {val_test_overlap}"

    print("ZERO data leakage detected! All splits are strictly disjoint.\n")


if __name__ == '__main__':
    test_no_data_leakage()
