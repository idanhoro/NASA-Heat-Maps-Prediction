import os

import time

from src.crawler import get_frame_urls
from src.df_creator import create_df
from src.environment import verify_required_files_dirs

if __name__ == "__main__":
    verify_required_files_dirs()

    start_time = time.time()
    print("Start at: {}".format(start_time))

    csv_file = ".{0}CSVs{0}final_df.csv".format(os.sep)
    create_df(get_frame_urls()).to_csv(csv_file, index=False)

    print("Finished at: {}".format(time.time()))
    print("--- {} seconds ---".format(time.time() - start_time))
