import os

import numpy as np
import time
from datetime import datetime

from src.crawler import get_frame_urls
from src.df_creator import create_df
from src.environment import init, load_config

if __name__ == "__main__":
    init()

    start_time = time.time()
    print("Start at: {}".format(start_time))

    csv_file = ".{0}CSVs{0}combined-{1}.csv".format(
        os.sep, datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    )
    create_df(get_frame_urls()).to_csv(csv_file, index=False)

    print("Finished at: {}".format(time.time()))
    print("--- {} seconds ---".format(time.time() - start_time))
