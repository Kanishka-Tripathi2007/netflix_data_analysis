import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Netflix Data Analysis System
# Place netflix_titles.csv in the same folder as this Python file, or update DATA_FILE.
DATA_FILE = "netflix_titles.csv"


def load_data(file_path):
    """Load and prepare the Netflix titles dataset."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise SystemExit(
            f"Dataset not found: {file_path}\n"
            "Place netflix_titles.csv beside this script or change DATA_FILE."
        )

    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    required_columns = {"show_id", "type", "title", "director", "cast", "country",
                        "date_added", "release_year", "rating", "duration", "listed_in"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise SystemExit("Dataset is missing required columns: " + ", ".join(sorted(missing)))

    data["date_added"] = pd.to_datetime(data["date_added"], errors="coerce")
    data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")
    data["duration_value"] = pd.to_numeric(
        data["duration"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    data["duration_unit"] = data["duration"].astype(str).str.extract(r"([A-Za-z]+)", expand=False)
    data["year_added"] = data["date_added"].dt.year
    data["month_added"] = data["date_added"].dt.month_name()
    return data


def split_values(series):
    """Split comma-separated fields into trimmed individual values."""
    return (series.dropna().astype(str).str.split(", ").explode().str.strip()
            .replace("", np.nan).dropna())


def show_summary(data):
    print("\n" + "=" * 60)
    print("NETFLIX DATA ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"Total titles: {len(data):,}")
    print(f"Movies: {(data['type'].str.lower() == 'movie').sum():,}")
    print(f"TV Shows: {(data['type'].str.lower() == 'tv show').sum():,}")
    print(f"Unique release years: {data['release_year'].nunique():,}")
    print(f"Missing values: {int(data.isna().sum().sum()):,}")

    print("\nDataset preview:")
    print(data[["show_id", "type", "title", "release_year", "rating", "duration"]].head(10).to_string(index=False))

    print("\nTitles by content type:")
    print(data["type"].value_counts(dropna=False).to_string())

    if data["release_year"].notna().any():
        print("\nFive years with the most released titles:")
        print(data["release_year"].value_counts().head(5).to_string())

    if data["rating"].notna().any():
        print("\nMost common content ratings:")
        print(data["rating"].value_counts().head(10).to_string())

    countries = split_values(data["country"])
    if not countries.empty:
        print("\nTop 10 producing countries:")
        print(countries.value_counts().head(10).to_string())

    genres = split_values(data["listed_in"])
    if not genres.empty:
        print("\nTop 10 genres:")
        print(genres.value_counts().head(10).to_string())

    directors = split_values(data["director"])
    if not directors.empty:
        print("\nTop 10 directors by number of titles:")
        print(directors.value_counts().head(10).to_string())

    movie_minutes = data.loc[data["type"].str.lower() == "movie", "duration_value"].dropna()
    if not movie_minutes.empty:
        print(f"\nAverage movie duration: {movie_minutes.mean():.1f} minutes")

    season_counts = data.loc[data["type"].str.lower() == "tv show", "duration_value"].dropna()
    if not season_counts.empty:
        print(f"Average TV show length: {season_counts.mean():.1f} seasons")


def create_visualizations(data):
    """Create a dashboard of charts and save it as netflix_analysis.png."""
    sns.set_theme(style="whitegrid", palette="Set2")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Netflix Titles Analysis", fontsize=20, fontweight="bold")

    # Titles by type
    type_counts = data["type"].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=axes[0, 0])
    axes[0, 0].set_title("Titles by Type")
    axes[0, 0].set_xlabel("Content type")
    axes[0, 0].set_ylabel("Number of titles")
    for bar in axes[0, 0].patches:
        axes[0, 0].annotate(f"{int(bar.get_height()):,}",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center", va="bottom", fontsize=9)

    # Release trends
    releases = data.dropna(subset=["release_year"]).groupby("release_year").size()
    releases = releases.loc[releases.index >= 1980]
    if not releases.empty:
        sns.lineplot(x=releases.index, y=releases.values, ax=axes[0, 1], marker="o")
    axes[0, 1].set_title("Titles by Release Year (1980 onward)")
    axes[0, 1].set_xlabel("Release year")
    axes[0, 1].set_ylabel("Number of titles")

    # Most common genres
    genre_counts = split_values(data["listed_in"]).value_counts().head(10).sort_values()
    if not genre_counts.empty:
        sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=axes[1, 0])
    axes[1, 0].set_title("Top 10 Genres")
    axes[1, 0].set_xlabel("Number of titles")
    axes[1, 0].set_ylabel("Genre")

    # Rating distribution
    rating_counts = data["rating"].value_counts().head(10)
    if not rating_counts.empty:
        sns.barplot(x=rating_counts.values, y=rating_counts.index, ax=axes[1, 1])
    axes[1, 1].set_title("Content Ratings")
    axes[1, 1].set_xlabel("Number of titles")
    axes[1, 1].set_ylabel("Rating")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("netflix_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\nChart saved as netflix_analysis.png")


def main():
    netflix_data = load_data(DATA_FILE)
    show_summary(netflix_data)
    create_visualizations(netflix_data)


if __name__ == "__main__":
    main()
