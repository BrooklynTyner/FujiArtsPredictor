#one may need to change the path to this folder

import re
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path.home() / "Desktop" / "FujiArtsPredictor"
TRAIN_PATH = FOLDER / "train.csv"      # R: read_csv("Desktop/fuj/train-2.csv")
RAW_PATH = FOLDER / "data_raw.csv"       # same folder as train
OUT_PATH = FOLDER / "train_transformed.csv"

train_2 = pd.read_csv(TRAIN_PATH)
data_raw = pd.read_csv(RAW_PATH)


def matching_rows(artist: str) -> pd.DataFrame:
#checks if artist is missing
    if pd.isna(artist):
        return data_raw.iloc[0:0]
#finds words in artist
    words = re.findall(r"[A-Za-z]+", artist)
    if not words:  
        return data_raw.iloc[0:0]

#searches artistString for words found above and returns if they are found
    pattern = r"\b(?:" + "|".join(map(re.escape, words)) + r")\b"

    mask = data_raw["artistString"].astype("string").str.contains(
        pattern, flags=re.IGNORECASE, regex=True, na=False
    )
    return data_raw[mask.to_numpy(dtype=bool)]


#gets unique artists
unique_artists = train_2["artist_info"].unique()
#creates dictionaries

counts, eras = {}, {}
#loop for each artist, counts matching entries

for artist in unique_artists:
    rows = matching_rows(artist)
    counts[artist] = len(rows)
#counts each unique era

    era_counts = (
        rows["era"].dropna()
        .value_counts()
        .rename_axis("era")
        .reset_index(name="n")
        .sort_values(["n", "era"], ascending=[False, True], kind="stable")
    )
#stores most common eras
    eras[artist] = era_counts["era"].iloc[0] if len(era_counts) else np.nan
#adds era and artist_count to dataframe
train_2["artist_count"] = train_2["artist_info"].map(counts)
train_2["era"] = train_2["artist_info"].map(eras)


#fills missing eras from the year at the end of link_text
#collects unique era strings of the form "(YYYY to YYYY)" to make a range of years for each era
era_ranges = (
    train_2.loc[train_2["era"].notna(), ["era"]]
    .drop_duplicates()
)
era_ranges = era_ranges[
    era_ranges["era"].str.contains(r"\(\d{4} to \d{4}\)", regex=True)
].copy()
era_ranges["start_year"] = era_ranges["era"].str.extract(r"(?<=\()(\d{4})")[0].astype(float)
era_ranges["end_year"] = era_ranges["era"].str.extract(r"(?<=to )(\d{4})")[0].astype(float)
era_ranges = era_ranges.sort_values("start_year", kind="stable").reset_index(drop=True)

#gets the year at the end of link_text to add era to artist not mentioned in raw data
train_2["link_year"] = (
    train_2["link_text"].astype("string").str.extract(r"(\d{4})$")[0].astype(float)
)


#fills in missing eras
def fill_era(current_era, year):
    if pd.notna(current_era):  
        return current_era
    if pd.isna(year):                  
        return np.nan
    match = era_ranges[
        (era_ranges["start_year"] <= year) & (year <= era_ranges["end_year"])
    ]
# era_ranges is already sorted by start_year -> first row = lowest start
    return match["era"].iloc[0] if len(match) else np.nan


train_2["era"] = [
    fill_era(e, y) for e, y in zip(train_2["era"], train_2["link_year"])
]
train_2 = train_2.drop(columns="link_year")

#saves to new .csv
train_2.to_csv(OUT_PATH, index=False)
print(f"Saved {len(train_2):,} rows to {OUT_PATH}")
