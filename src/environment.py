import os

import toml


def check_dir(dirname: str, abort_on_fail: bool) -> bool:
    """
    Checks if the given directory exists.

    :param dirname: Directory's path
    :param abort_on_fail: True if directory is required, otherwise creates it.
    :return: True if directory exists, otherwise exits.
    """
    if os.path.isdir(dirname):
        return True
    print("The required {} directory could not be found.".format(dirname))
    if abort_on_fail:
        print("Aborting...")
        exit(1)
    else:
        os.mkdir(dirname)
        print("Created directory {}".format(dirname))


def check_file(filename: str) -> bool:
    if os.path.isfile(filename):
        return True
    print(
        "The required {} file could not be found.\nAborting...".format(
            filename
        )
    )
    exit(-1)


def verify_required_files_dirs():
    check_dir("assets", True)
    check_file(os.path.join("assets", "clean_map.jpeg"))
    check_file(os.path.join("assets", "outliers_mask.npy"))
    check_file("env.toml")
    check_dir("CSVs", False)
    check_dir("Visualization", False)


def load_config(filename: str = "env.toml"):
    with open(filename, "r") as fh:
        return toml.load(fh)


CONFIG = load_config()
