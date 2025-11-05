# gpxfun/io_fit.py
from fitparse import FitFile
import pandas as pd

def parse_fit(filepath: str) -> pd.DataFrame:
    """Parse FIT file and return dataframe with key metrics."""
    fitfile = FitFile(filepath)
    records = []
    for record in fitfile.get_messages("record"):
        r = {}
        for data in record:
            r[data.name] = data.value
        records.append(r)
    df = pd.DataFrame(records)
    df = df.rename(columns={"position_lat":"lat","position_long":"lon"})
    # Convert semicircles → degrees
    if "lat" in df and "lon" in df:
        df["lat"] = df["lat"] * (180 / 2**31)
        df["lon"] = df["lon"] * (180 / 2**31)
    return df.dropna(subset=["lat","lon"], how="any")
