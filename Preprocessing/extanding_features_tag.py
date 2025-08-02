
# Compute textual, temporal and channel-level features to assistYouTube view-prediction models. No thumbnail-based attributes.

#  Adding a tags column  (hashtags of title + description)

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import emoji
    EMOJI_RE = emoji.get_emoji_regexp()
except ImportError:
    EMOJI_RE = re.compile("[\U00010000-\U0010ffff]")  

try:
    import nltk
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except Exception:
    STOPWORDS = set()

# -----------------------------------------------------------------------
features_added = [
    # —— title
    "title_char_count", "title_word_count", "title_unique_ratio", "title_upper_ratio",
    "title_has_question", "title_has_exclamation", "title_hashtag_count",
    "title_avg_word_length", "title_stopword_ratio", "title_has_number",
    "title_starts_with_number", "title_has_capslock_word", "title_emojis_count",
    "title_special_char_count", "title_word_overlap_desc",
    # —— description
    "description_char_count", "description_word_count", "description_hashtag_count",
    "description_link_count", "description_mention_count", "description_emojis_count",
    "description_link_ratio", "description_sentiment_polarity",
    # —— temporal
    "publish_year", "publish_month", "publish_dayofweek", "publish_hour",
    "publish_weekofyear", "video_age_days", "is_weekend_publish",
    "publish_part_of_day", "is_holiday_season",
    # —— channel
    "channel_known", "channel_video_count", "channel_mean_views",
    "channel_median_views", "channel_q25_views", "channel_q75_views",
    "channel_min_views", "channel_max_views", "channel_iqr_views",
    # —— sentiments (title)
    "title_polarity", "title_subjectivity",
    # —— tags
    "tags"
]

SPECIAL_CHARS_RE = re.compile(r"[!*$~¿¡]")
NUMBER_RE = re.compile(r"\d")
CAPSLOCK_RE = re.compile(r"\b[A-Z]{3,}\b")
TAGS_RE = re.compile(r"#(\w+)")                       

# -----------------------------------------------------------------------
def add_features(df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Enrich *df* with engineered features (no image features).
    Returns a new DataFrame.
    """
    out = df.copy()

    # ——————————————————————————— Text features ———————————————————————————
    out["title"].fillna("", inplace=True)
    out["description"].fillna("", inplace=True)

    # ---------- NEW : tags (string) -------------------------------------
    def _extract_tags(title: str, desc: str) -> str:
        """Return space-separated unique hashtags (lower-case, no #)."""
        t_tags = TAGS_RE.findall(title)
        d_tags = TAGS_RE.findall(desc)
        seen, uniq = set(), []
        for tag in t_tags + d_tags:               
            tag_lc = tag.lower()
            if tag_lc not in seen:
                uniq.append(tag_lc)
                seen.add(tag_lc)
        return " ".join(uniq)                     

    out["tags"] = out.apply(lambda r: _extract_tags(r["title"], r["description"]), axis=1)
    out["tags"].replace("", "no_tags", inplace=True)

    # --------------------------------------------------------------------

    # basic counts
    out["title_char_count"] = out["title"].str.len()
    out["title_word_count"] = out["title"].str.split().str.len()
    out["title_unique_ratio"] = out["title"].str.lower().str.split().apply(
        lambda w: len(set(w)) / (len(w) + 1e-6)
    )
    out["title_upper_ratio"] = out["title"].str.count(r"[A-Z]") / (out["title_char_count"] + 1e-6)

    # punctuation / tokens
    out["title_has_question"] = out["title"].str.contains(r"\?").astype("int8")
    out["title_has_exclamation"] = out["title"].str.contains(r"!").astype("int8")
    out["title_hashtag_count"] = out["title"].str.count(r"#\w+")

    # lexical richness
    out["title_avg_word_length"] = out["title"].str.split().apply(
        lambda w: sum(len(tok) for tok in w) / (len(w) + 1e-6)
    )
    out["title_stopword_ratio"] = out["title"].str.split().apply(
        lambda w: sum(tok.lower() in STOPWORDS for tok in w) / (len(w) + 1e-6)
    )

    # patterns & emojis
    out["title_has_number"] = out["title"].str.contains(NUMBER_RE).astype("int8")
    out["title_starts_with_number"] = out["title"].str.match(r"\s*\d").astype("int8")
    out["title_has_capslock_word"] = out["title"].str.contains(CAPSLOCK_RE).astype("int8")
    out["title_emojis_count"] = out["title"].apply(lambda t: len(EMOJI_RE.findall(t)))
    out["title_special_char_count"] = out["title"].str.count(SPECIAL_CHARS_RE)

    # description core
    out["description_char_count"] = out["description"].str.len()
    out["description_word_count"] = out["description"].str.split().str.len()
    out["description_hashtag_count"] = out["description"].str.count(r"#\w+")
    out["description_link_count"] = out["description"].str.count(r"https?://")
    out["description_mention_count"] = out["description"].str.count(r"@\w+")
    out["description_emojis_count"] = out["description"].apply(lambda t: len(EMOJI_RE.findall(t)))
    out["description_link_ratio"] = out["description_link_count"] / (out["description_word_count"] + 1e-6)

    # title-description overlap
    out["title_word_overlap_desc"] = out.apply(
        lambda r: len(set(r["title"].lower().split()) &
                      set(r["description"].lower().split())) / (len(r["title"].split()) + 1e-6),
        axis=1,
    )

    # sentiment
    try:
        from textblob import TextBlob
        out["title_polarity"] = out["title"].apply(lambda t: TextBlob(t).sentiment.polarity)
        out["title_subjectivity"] = out["title"].apply(lambda t: TextBlob(t).sentiment.subjectivity)
        out["description_sentiment_polarity"] = out["description"].apply(lambda t: TextBlob(t).sentiment.polarity)
    except ImportError:
        out["title_polarity"] = 0.0
        out["title_subjectivity"] = 0.0
        out["description_sentiment_polarity"] = 0.0

    # ——————————————————————————— Temporal features ———————————————————————————
    out["publish_dt"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out["publish_year"] = out["publish_dt"].dt.year.astype("int16")
    out["publish_month"] = out["publish_dt"].dt.month.astype("int8")
    out["publish_dayofweek"] = out["publish_dt"].dt.dayofweek.astype("int8")
    out["publish_hour"] = out["publish_dt"].dt.hour.astype("int8")
    out["publish_weekofyear"] = out["publish_dt"].dt.isocalendar().week.astype("int8")

    out["is_weekend_publish"] = (out["publish_dayofweek"] >= 5).astype("int8")
    out["publish_part_of_day"] = out["publish_hour"].apply(
        lambda h: 0 if h < 6 else 1 if h < 12 else 2 if h < 18 else 3
    ).astype("int8")
    out["is_holiday_season"] = out["publish_month"].isin([7, 8, 12]).astype("int8")

    ref_date = pd.Timestamp("2025-04-15", tz="UTC")
    out["video_age_days"] = (ref_date - out["publish_dt"]).dt.days.astype("int32")

    # ——————————————————————————— Channel aggregates ———————————————————————————
    base = reference_df if reference_df is not None else out
    ch_stats = (
        base.groupby("channel", as_index=False)
        .agg(
            channel_video_count=("id", "count"),
            channel_mean_views=("views", "mean"),
            channel_median_views=("views", "median"),
            channel_q25_views=("views", lambda x: np.percentile(x, 25)),
            channel_q75_views=("views", lambda x: np.percentile(x, 75)),
            channel_min_views=("views", "min"),
            channel_max_views=("views", "max"),
        )
    )
    out = out.merge(ch_stats, on="channel", how="left")
    out["channel_known"] = out["channel_video_count"].notna().astype("int8")

    fill_cols = [
        "channel_video_count", "channel_mean_views", "channel_median_views", "channel_q25_views",
        "channel_q75_views", "channel_min_views", "channel_max_views",
    ]
    out[fill_cols] = out[fill_cols].fillna(0)
    out["channel_iqr_views"] = out["channel_q75_views"] - out["channel_q25_views"]

    out.drop(columns=["publish_dt"], inplace=True)
    return out


def main() -> None:
    train_df = pd.read_csv("dataset/train.csv")
    val_df   = pd.read_csv("dataset/val.csv")
    test_df = pd.read_csv("dataset/test.csv")

    if not train_df.empty:
        train_fe = add_features(train_df, reference_df=train_df)
        train_fe.to_csv("dataset/train_fe.csv", index=False)

    if not val_df.empty:
        val_fe = add_features(val_df, reference_df=train_df)
        val_fe.to_csv("dataset/val_fe.csv", index=False)

    if not test_df.empty:
        test_fe = add_features(test_df, reference_df=train_df)
        test_fe.to_csv("dataset/test_fe.csv", index=False)

    print(" Feature engineering complete – files saved ")
    if not train_df.empty:
        print(f"train_fe rows: {len(train_fe)}")
        print(f"train_fe columns: {train_fe.columns.tolist()}")
    if not val_df.empty:
        print(f"val_fe rows:   {len(val_fe)}")

if __name__ == "__main__":
    main()
