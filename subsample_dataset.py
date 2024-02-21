"""
Subsample a composed moving cable dataset by a factor of ten. The subsampled
dataset will have ca. 60 images per clip instead of ca. 600.
"""

import os
import sys
import pathlib
import shutil



def copy_file(src_path, dst_path, dry_run=False, verbose=False):
    if verbose:
        print(src_path, "-->", dst_path)
    if dry_run:
        return
    pathlib.Path(os.path.dirname(dst_path)).mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dst_path)


def subsample_files(src, dst):
    skip_count = 20
    i = 1
    src_path = os.path.join(src, "{:08d}.png".format(i))
    while os.path.isfile(src_path):
        dst_path = os.path.join(dst, "{:08d}.png".format(i))
        copy_file(src_path, dst_path)
        i += skip_count
        src_path = os.path.join(src, "{:08d}.png".format(i))


def subsample_recursive(src_root, dst_root):
    lst = sorted(os.listdir(src_root))
    if not lst:
        return
    path = os.path.join(src_root, lst[0])
    if not os.path.isdir(path):
        subsample_files(src_root, dst_root)
        return
    folders = lst
    while folders:
        folder = folders.pop()
        folder_path = os.path.join(src_root, folder)
        assert(os.path.isdir(folder_path))
        lst = sorted(os.listdir(folder_path))
        if not lst:
            continue
        path_0 = os.path.join(folder_path, lst[0])
        if not os.path.isdir(path_0):
            print(folder)
            subsample_files(
                os.path.join(src_root, folder),
                os.path.join(dst_root, folder))
            continue
        for name in lst:
            path_n = os.path.join(folder_path, name)
            assert(os.path.isdir(path_n))
            folders.append(os.path.join(folder, name))


def main():
    if len(sys.argv) != 3:
        print("Usage: python subsample_dataset.py /src/dataset/root "
            "/dst/dataset/root")
        return
    src_root = sys.argv[1]
    dst_root = sys.argv[2]
    subsample_recursive(src_root, dst_root)


if __name__ == "__main__":
    main()
