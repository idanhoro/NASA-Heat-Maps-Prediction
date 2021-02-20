import os

import toml


def check_dir(dirname: str, abort_on_fail: bool):
    if not os.path.isdir(formatted_dirname := ".{0}{1}{0}".format(os.sep, dirname)):
        print("The required {} directory could not be found.".format(formatted_dirname))
        if abort_on_fail:
            print("Aborting...")
            exit(1)
        else:
            os.mkdir(formatted_dirname)
            print("Created directory {}".format(formatted_dirname))
    return True


def check_file(filename: str):
    if not os.path.isfile(formatted_filename := ".{}{}".format(os.sep, filename)):
        print(
            "The required {} file could not be found.\nAborting...".format(
                formatted_filename
            )
        )
        exit(-1)
    return True


def verify_required_files_dirs():
    check_dir("assets", True)
    check_file("assets{}clean_map.jpeg".format(os.sep))
    check_file("assets{}outliers_mask.npy".format(os.sep))
    check_file("env.toml")
    check_dir("CSVs", False)
    check_dir("Visualization", False)


def load_config(filename: str = ".{}env.toml".format(os.sep)):
    with open(filename, "r") as fh:
        return toml.load(fh)


CONFIG = load_config()
