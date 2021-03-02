import sys
import time
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from bs4 import BeautifulSoup

from src.environment import CONFIG

BASE_URL = CONFIG["crawler"]["BASE_URL"]
MAX_CRAWL_SLEEP_TIME = CONFIG["crawler"]["MAX_CRAWL_SLEEP_TIME"]


def parse_date_from_url(map_url: str) -> tuple[np.ushort, np.ubyte]:
    year, month = (np.ushort(map_url[-12:-8]), np.ubyte(map_url[-7:-5]))
    return year, month


def image_url_to_mat(map_url: str) -> np.ndarray:
    response = get_by_url(map_url)
    map_mat = np.asarray(Image.open(BytesIO(response.content)).convert("RGB"))

    return map_mat


def get_frame_urls() -> dict[str, tuple[str, np.ndarray]]:
    map_dict = {
        "Land Surface Temperature Anomaly": "MOD_LSTAD_M",
        "Snow Cover": "MOD10C1_M_SNOW",
        "Land Surface Temperature": "MOD_LSTD_M",
        "Vegetation": "MOD_NDVI_M",
        "Fire": "MOD14A1_M_FIRE",
        "Net Primary Productivity": "MOD17A2_M_PSN",
    }

    map_names = {cat: np.array([]) for cat in CONFIG["maps"]["types"]}

    for map_name in map_names.keys():
        req = get_by_url("{}global-maps/{}".format(BASE_URL, map_dict[map_name]))
        html = BeautifulSoup(req.text, "lxml")

        maps_player = html.find("div", class_="panel-slideshow panel-slideshow-primary")
        urls_arr = np.array(
            [img["src"] for img in maps_player.find_all("img", class_="panel-slideshow-image", src=True)]
        )
        scale_url = "{}{}".format(
            BASE_URL,
            html.find("img", class_="panel-slideshow-scale-image", src=True)["src"],
        )

        for idx, img_url in enumerate(urls_arr):
            if r"no_data_available" in img_url:
                # Remove "Missing Data" heatmaps.
                urls_arr = np.delete(urls_arr, idx)

        map_names[map_name] = (scale_url, urls_arr)

    return map_names


def get_by_url(url=BASE_URL, params=None) -> requests.Response:
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36"
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print("Could not reach: {} status code: {} html: {}".format(response.url, response.status_code, response.text))
        sys.exit(1)

    time.sleep(MAX_CRAWL_SLEEP_TIME)

    return response
