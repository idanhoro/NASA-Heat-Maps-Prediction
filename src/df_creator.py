import os
import time

import numpy as np
import pandas as pd
from PIL import Image

from src.crawler import parse_date_from_url
from src.matrix_analyzer import map_analyzer, color_mapper


def merge_final_df(
    cat_df_list: list[pd.DataFrame], bool_col_list: list[str]
) -> pd.DataFrame:
    print("Merging...")

    final_df: pd.DataFrame = cat_df_list.pop()
    for cat_df in cat_df_list:
        final_df = pd.merge(final_df, cat_df, how="outer")

    for col in bool_col_list:
        final_df[col].fillna(False, inplace=True)

    final_df.fillna(-1, inplace=True)

    return final_df


def create_df(maps: dict[str, tuple[str, np.ndarray]]) -> pd.DataFrame:
    clean_img_arr: np.ndarray = np.asarray(
        Image.open(".{0}assets{0}clean_map.jpeg".format(os.sep)).convert("RGB")
    )
    """
    Example to the structure of the `maps` variable:
    maps = {
        "vegetation": tuple([scale_url, np.array([map_url1, map_url2,...,map_url252])]),
        "land surface temperature": tuple([scale_url, np.array([map_url1, map_url3,...,map_url252])])
    }
    """

    indices_by_category_by_month: dict[
        str, dict[tuple[np.ushort, np.ubyte], np.ndarray]
    ] = {cat: dict() for cat in maps}
    """
    Example to the structure of the `indices_by_category_by_month` variable:
        maps = {
        "vegetation": {
            (2000,3): np.array([index0, index1, ..., index249999], shape=(245000,)],
            (2000,4): np.array([index0, index1, ..., index249999], shape=(245000,)],
            ...
            (2020,12): np.array([index0, index1, ..., index249999], shape=(245000,)]
            },
        "land surface temperature": {
            (2000,2): np.array([index0, index1, ..., index249999], shape=(245000,)],
            (2000,3): np.array([index0, index1, ..., index249999], shape=(245000,)],
            ...
            (2020,12): np.array([index0, index1, ..., index249999], shape=(245000,)]
            },
        ...
        }
    """

    for map_category in maps:
        scale_url, url_arr = maps[map_category]
        scale_arr: np.array = color_mapper(scale_url)
        for map_url in url_arr:
            map_start_time = time.time()

            final_map = map_analyzer(map_url, clean_img_arr, scale_arr)
            indices_by_category_by_month[map_category][
                parse_date_from_url(map_url)
            ] = final_map

            print(
                "--- {} seconds to process map {} ---".format(
                    (time.time() - map_start_time), map_url
                )
            )

    cat_df_list, bool_col_list = monthly_to_categorical_df(indices_by_category_by_month)

    if len(cat_df_list) == 1:
        return cat_df_list[0]

    return merge_final_df(cat_df_list, bool_col_list)


def monthly_to_categorical_df(
    categories_by_month_dict: dict[str, dict[tuple[int, int], np.ndarray]],
) -> tuple[list[pd.DataFrame], list[str]]:
    bool_col_list: list[str] = []
    cat_df_list: list[pd.DataFrame] = []

    for cat in categories_by_month_dict:
        category_df: pd.DataFrame = pd.DataFrame()
        for date in categories_by_month_dict[cat]:
            mdf = create_monthly_df(cat, date, categories_by_month_dict[cat][date])
            bool_col_list.append(mdf.columns[-1])

            category_df = pd.concat([category_df, mdf], ignore_index=True)

        cat_df_list.append(category_df)

    return cat_df_list, bool_col_list


def create_monthly_df(
    category: str, date: tuple[int, int], idx_arr: np.ndarray
) -> pd.DataFrame:
    print("Parsing {} {}".format(category, date))
    monthly_df: pd.DataFrame = pd.DataFrame()
    monthly_df["Y"], monthly_df["X"] = np.indices((350, 700)).reshape(2, -1)
    monthly_df["Year"], monthly_df["Month"] = date
    monthly_df["{} Color Index".format(category)] = idx_arr

    # Add a boolean column to indicate whether the pixel for a given map is valuable.
    monthly_df["{} Is Valuable".format(category)] = (
        monthly_df["{} Color Index".format(category)] != -1
    )

    # Clean the outliers - white background and sea.
    return monthly_df.loc[
        np.load(".{0}assets{0}outliers_mask.npy".format(os.sep)).reshape(-1)
    ]
