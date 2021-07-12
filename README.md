# NASA Heat Maps Prediction

-------------------------------

### Description

In this project we research the correlations between different weather conditions and try to predict future scenarios by
using image processing and traditional machine learning algorithms.

### Data Collection

Using the libraries `requests` and `BeautifulSoup4` we crawl NASA's website searching for heat maps, and their scale
according to the categories which were noted in the `env.toml` file.

```toml
[maps]
types = [
    "Vegetation",
    "Net Primary Productivity",
    "Land Surface Temperature",
    "Snow Cover",
    "Fire",
    "Land Surface Temperature Anomaly",
]
```

#### Preparation

The URLs of the scale and images we found are then inserted into NumPy arrays for each map category which is found in a
dictionary structured likewise:

```python
maps = {
    "vegetation": tuple([scale_url, np.array([map_url1, map_url2, ..., map_url252])]),
    "land surface temperature": tuple([scale_url, np.array([map_url1, map_url3, ..., map_url252])])
}
```

```python
map_names = {cat: np.array([]) for cat in CONFIG["maps"]["types"]}

for map_name in map_names.keys():
    req = get_by_url("{}global-maps/{}".format(BASE_URL, map_dict[map_name]))
    html = BeautifulSoup(req.text, "lxml")

    maps_player = html.find("div", class_="panel-slideshow panel-slideshow-primary")
    urls_arr = np.array([img["src"] for img in maps_player.find_all("img", class_="panel-slideshow-image", src=True)])
    scale_url = "{}{}".format(BASE_URL, html.find("img", class_="panel-slideshow-scale-image", src=True)["src"])

    for idx, img_url in enumerate(urls_arr):
        if r"no_data_available" in img_url:
            # Remove "Missing Data" heatmaps.
            urls_arr = np.delete(urls_arr, idx)

    map_names[map_name] = (scale_url, urls_arr)
```

--------

#### Scraping

Now that we have a scale and a list of maps for each category, we are able to iterate over all of those, which allows us
to continue to the next step - image analysis.

```python
for map_category in maps:
    scale_url, url_arr = maps[map_category]
    scale_arr: np.array = color_mapper(scale_url)
    for map_url in url_arr:
        ...
```

-----

### Image Analysis

For each of those image URLs, we call our `map_analyzer()` function which is responsible to the following parts:

#### Clean Pixel Removal

We decided to remove pixels which were not colored, by using the RGB difference formula:
![RGB Difference Formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/06cdd86ced397bbf6fad505b4c4d91fa2438b567)

As you can see, instead of iterating over the pixels of the heatmap `(350,700,3)*2`
we decided to use a vectorized version of this algorithm which saved us a lot of time per heatmap -- it allowed us to
calculate a single map in ~1.5 seconds instead of ~450 seconds.

```py
difference_array: np.ndarray = np.sqrt(np.sum((img_arr - clean_img_arr) ** 2, axis=2))
mask_array: np.ndarray = difference_array != 0
img_arr_subset: np.ndarray = img_arr[mask_array]
```

#### Scale Index Calculation

Using the aforementioned differences helped us find where a pixel is placed on the scale. For each pixel which we've
deemed "not clean" (AKA potentially valuable) we compared between the differences to the clean map and the differences to
the map by using`scipy.spatial.distance_matrix()`
and adding to it the differences to the scale. We then found out the position of the pixel to the scale, either by an
exact match or by finding the closest color.

```python
dist_matrix: np.ndarray = np.c_[difference_array[mask_array], distance_matrix(img_arr_subset, scale_arr)]
min_dist_indices: np.ndarray = (dist_matrix.argmin(axis=-1) - 1)  # Shift the values (scale indices) by -1
scale_pixels_final: np.ndarray = np.full(shape=(350, 700), fill_value=-1)  # (350,700) index matrix
scale_pixels_final[mask_array] = min_dist_indices
scale_pixels_final_reshaped: np.ndarray = scale_pixels_final.reshape(-1)
```

-------

### Data Categorization

Using the amazing `Pandas` library we were able to easily collect the data in a DataFrame on which we ran multiple other
steps.

```python
    for cat in categories_by_month_dict:
        category_df: pd.DataFrame = pd.DataFrame()
        for date in categories_by_month_dict[cat]:
            mdf = create_monthly_df(cat, date, categories_by_month_dict[cat][date])
            bool_col_list.append(mdf.columns[-1])

            category_df = pd.concat([category_df, mdf], ignore_index=True)

        cat_df_list.append(category_df)
```

-------

### Data Prediction

We ran multiple traditional machine learning algorithms against our data. We found that using the `Linear Pipeline`
method gave us the best results for the `Land Surface Temperature` heatmap standing at about `~90%` accuracy.

We noticed that using algorithms such as `Naive Bayes` or `MLP` didn't work out so well generally, but since our data is
not uniform, it performed best on the `Fire` dataset.

As a result, we can conclude that linear algorithms worked best for our data type.
