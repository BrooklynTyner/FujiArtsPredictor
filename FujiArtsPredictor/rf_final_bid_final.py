
import re
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "train_transformed.csv"
OUT_PATH = "train_with_predictions.csv"
TEST_PATH = "test_transformed.csv"  #new data to predict (same folder you run the script from)
TEST_OUT_PATH = "test_with_predictions.csv"
MIN_COUNT = 10  #artists and series with fewer prints are pooled into "Other"
MAX_FEATURES = .5  #chosen by cross validation (was 0.33); CV R^2 kept rising up to 1.0

#each damage word/group becomes its own 0/1 dummy (dmg_<name>)
# word following "no" or "without" are ignored
_NEG = r"(?<!no )(?<!without )\b(?:"
DAMAGE_TERMS = {
    "repair": r"repair\w*",
    "hole": r"holes?",
    "wormhole": r"wormholes?",
    "tear": r"tears?|torn",
    "stain": r"stain\w*",
    "loss": r"loss(?:es)?",
    "thinning": r"thinning",
    "soiling": r"soiling",
    "crease": r"creases?",
    "centerfold": r"cent(?:er|re)\s?folds?",
    "rubbing": r"rubb\w*",
    "spots": r"spots?",
}
#turns damage terms into dictionary to make dummy vars with
DAMAGE_RES = {
    name: re.compile(_NEG + pat + r")\b", re.I) for name, pat in DAMAGE_TERMS.items()
}
#flags for if on sale
CLEARANCE_RE = re.compile(r"clearance", re.I)
LOW_START_RE = re.compile(r"opening|special offer|starting bid|low start", re.I)
PROMO_RE = re.compile(r"A?\s*Fuji Arts.*$", re.I)  # promo text glued onto series_info

#condition-text 
PANEL_RE = re.compile(r"\bpanels?\b", re.I)
BACKED_RE = re.compile(r"\b(?:backed|backing)\b", re.I)
NO_ISSUES_RE = re.compile(r"\bno issues\b", re.I)

#defining functions to clean data
def clean_text(s: pd.Series) -> pd.Series:
    return s.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

#categories with at least min_count prints (computed on the TRAINING data)
def keep_set(s: pd.Series, min_count: int) -> set:
    counts = s.value_counts()
    return set(counts[counts >= min_count].index)

#creates other catagory for rare series
#keep = categories to keep; if None, they are worked out from s itself
def pool_rare(s: pd.Series, min_count: int, keep=None) -> pd.Series:
    if keep is None:
        keep = keep_set(s, min_count)
    return s.where(s.isin(keep), "Other")


def clean_columns(df: pd.DataFrame):
    title = clean_text(df["link_text"])
    series_raw = clean_text(df["series_info"])
    condition = clean_text(df["condition"])

#removes auction promo text and "Series;" prefix
    series = series_raw.str.replace(PROMO_RE, "", regex=True)
    series = series.str.replace(r"^Series[;:]\s*", "", regex=True).str.strip()
    series = series.replace("", "No series")

    artist = clean_text(df["artist_info"]).replace("", "Unknown")
    return title, series_raw, series, artist, condition


#ref = the training dataframe. For new data, pass ref=train_df so that rare-category
#pooling, medians and counts all come from the training data (not the new data).
#When ref is None the data is its own reference (original behaviour).
def build_features(df: pd.DataFrame, ref: pd.DataFrame = None) -> pd.DataFrame:
    ref_df = df if ref is None else ref

    title, series_raw, series, artist, condition = clean_columns(df)
    _, _, ref_series, ref_artist, _ = clean_columns(ref_df)

#creates the dummy var dataframe for damage
    damage = pd.DataFrame(
        {f"dmg_{name}": condition.str.contains(rx) for name, rx in DAMAGE_RES.items()}
    ).astype(int)

    flags = pd.DataFrame(
        {
            "has_damage": damage.max(axis=1).astype(bool),  # any damage word
            "clearance": (title + " " + series_raw).str.contains(CLEARANCE_RE),
            "low_start_bid": (title + " " + series_raw).str.contains(LOW_START_RE),
            "cond_panel": condition.str.contains(PANEL_RE),
            "cond_backed": condition.str.contains(BACKED_RE),
            "cond_no_issues": condition.str.contains(NO_ISSUES_RE),
        }
    ).astype(int)
#creates dummys for series and artist
    artist_d = pd.get_dummies(
        pool_rare(artist, MIN_COUNT, keep_set(ref_artist, MIN_COUNT)), prefix="artist"
    )
    series_d = pd.get_dummies(
        pool_rare(series, MIN_COUNT, keep_set(ref_series, MIN_COUNT)), prefix="series"
    )

#era can be numeric or text depending on the file; type is decided by the training data
    if pd.api.types.is_numeric_dtype(ref_df["era"]):
        era_median = pd.to_numeric(ref_df["era"], errors="coerce").median()
        era_d = pd.to_numeric(df["era"], errors="coerce").fillna(era_median).to_frame("era")
    else:
        era = clean_text(df["era"]).replace("", "Unknown")
        ref_era = clean_text(ref_df["era"]).replace("", "Unknown")
        era_d = pd.get_dummies(
            pool_rare(era, MIN_COUNT, keep_set(ref_era, MIN_COUNT)), prefix="era"
        )

#adds dummy vars 
    X = pd.concat([flags, damage, artist_d, series_d, era_d], axis=1).astype(int)
    X.insert(0, "original", df["original"].astype(int))  # existing 0/1 column
    X.insert(1, "title_length", title.str.len())  # numb of characters in link_text


#fixes errors that appear due to unclean data with artist_count

    if "artist_count" in df.columns:
        ac = pd.to_numeric(df["artist_count"], errors="coerce")
        ref_ac = pd.to_numeric(ref_df["artist_count"], errors="coerce")
        X.insert(2, "artist_count", ac.fillna(ref_ac.median()))
    else:
        print("NOTE: no 'artist_count' column in CSV; using prints-per-artist instead")
        ref_counts = ref_artist.value_counts()
        X.insert(2, "artist_count", artist.map(ref_counts).fillna(1))
    return X


def main():
    df = pd.read_csv(CSV_PATH)
    X = build_features(df)
    y = np.log(df["final_bid"])


#creates testing and training data at a 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=449)
#defines random forest 
    rf = RandomForestRegressor(
        n_estimators=800,
        min_samples_leaf=5,
        max_features=MAX_FEATURES,
        oob_score=True,
        n_jobs=-1,
        random_state=493,
    )
#traines forest
    rf.fit(X_train, y_train)
#predict values in X_test
    pred = rf.predict(X_test)
#calc. RMSE for log and usd
    rmse_log = np.sqrt(mean_squared_error(y_test, pred))
    rmse_usd = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))
#prints performance
    print(f"OOB R^2 (train): {rf.oob_score_:.3f}")
    print(f"Test R^2 (log bid): {r2_score(y_test, pred):.3f}")
    print(f"Test RMSE (log bid): {rmse_log:.3f}")
    print(f"Test RMSE (dollars): ${rmse_usd:,.2f} (after exp() of predictions)\n")
#prints 
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("top feature importances:")
    print(imp.head(15).round(4).to_string())

#outputing the fitted out of back estimates for training data used for validation and testing 

    final = RandomForestRegressor(
        n_estimators=800,
        min_samples_leaf=5,
        max_features=MAX_FEATURES,
        oob_score=True,
        n_jobs=-1,
        random_state=48,
    )
#fits data for all entries
    final.fit(X, y)
#makes copy of og dataset names out 
    out = df.copy()
#adds out of bag predictions to out
    out["predicted_log_final_bid"] = final.oob_prediction_
    out["predicted_final_bid"] = np.exp(final.oob_prediction_).round(2)
#saves new csv called train_with_predictions
    out.to_csv(OUT_PATH, index=False)
#tells me it completed this process
    print(f"\nSaved {len(out)} rows with predictions to {OUT_PATH}")

#predicts final bids for the new dataset using the model fit on all training data
    new = pd.read_csv(TEST_PATH)
#builds features the same way as training (pooling/medians come from the training data)
#reindex makes the columns match training exactly; any dummy not present in new data = 0
    X_new = build_features(new, ref=df).reindex(columns=X.columns, fill_value=0)
    new_pred_log = final.predict(X_new)
    new_out = new.copy()
    new_out["predicted_log_final_bid"] = new_pred_log
    new_out["predicted_final_bid"] = np.exp(new_pred_log).round(2)
    new_out.to_csv(TEST_OUT_PATH, index=False)
    print(f"Saved {len(new_out)} rows with predicted final bids to {TEST_OUT_PATH}")


if __name__ == "__main__":
    main()
