import os
import pandas as pd

def main():
    original_file = "data/raw/reviews.csv"
    repaired_file = "data/raw/reviews_repaired.csv"
    final_file = "data/raw/reviews_final.csv"

    if not os.path.exists(original_file):
        print(f"Error: Original reviews file not found: {original_file}")
        return

    if not os.path.exists(repaired_file):
        print(f"Error: Repaired reviews file not found: {repaired_file}")
        return

    print("Loading data files...")
    df_orig = pd.read_csv(original_file)
    df_rep = pd.read_csv(repaired_file)

    # Get the unique names of destinations that were repaired
    repaired_dests = df_rep["destination_name"].dropna().unique()
    print(f"Found {len(repaired_dests)} destinations that were repaired in reviews_repaired.csv:")
    for dest in sorted(repaired_dests):
        orig_count = len(df_orig[df_orig["destination_name"] == dest])
        rep_count = len(df_rep[df_rep["destination_name"] == dest])
        print(f" - {dest} (Original: {orig_count} reviews -> Repaired: {rep_count} reviews)")

    # Filter original reviews: remove rows for repaired destinations
    df_orig_filtered = df_orig[~df_orig["destination_name"].isin(repaired_dests)]

    # Concatenate the filtered original and repaired dataframes
    df_final = pd.concat([df_orig_filtered, df_rep], ignore_index=True)

    # Normalize 'has_text' column to uppercase TRUE/FALSE strings for consistency
    if "has_text" in df_final.columns:
        df_final["has_text"] = df_final["has_text"].astype(str).str.upper()

    # Save to the final reviews path
    df_final.to_csv(final_file, index=False)

    print("\nMerge Statistics:")
    print(f" - Original reviews file rows: {len(df_orig)}")
    print(f" - Repaired reviews file rows: {len(df_rep)}")
    print(f" - Filtered original rows (excluding repaired destinations): {len(df_orig_filtered)}")
    print(f" - Final merged reviews file rows: {len(df_final)}")
    print(f"🎉 Merged file saved successfully to: {final_file}")

if __name__ == "__main__":
    main()
