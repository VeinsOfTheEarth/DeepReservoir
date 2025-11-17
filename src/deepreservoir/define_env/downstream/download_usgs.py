# Quick script to overcome the 10000 limit for daily discharge data
import pandas as pd
import requests

names = {'09368000' : 'sj_shiprock',
         '09379500' : 'sj_bluff',
         '09371010' : 'sj_fourcorners',
         '09355500' : 'sj_archuleta',
         '09365000' : 'sj_farmington',
         '09364500' : 'animas_farmington',
         '09367950' : 'chaco_waterflow',
         '09379200' : 'chinle_mexicanwater',
         '09367500' : 'laplata_farmington',
         '09371000' : 'mancos_towaoc'
         }

def usgs_dv_multisite(sites, start="1900-01-01", end="2100-01-01",
                      parameter="00060", stat="00003"):
    url = "https://waterservices.usgs.gov/nwis/dv/"
    params = {
        "format": "json",
        "sites": ",".join(sites),
        "parameterCd": parameter,
        "statCd": stat,
        "startDT": start,
        "endDT": end,
        "siteStatus": "all",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    rows = []
    for ts in js["value"]["timeSeries"]:
        site = ts["sourceInfo"]["siteCode"][0]["value"]
        for pt in ts["values"][0]["value"]:
            v = pt.get("value")
            if v is None or v == "":
                continue
            rows.append((site, pt["dateTime"][:10], float(v)))

    df = pd.DataFrame(rows, columns=["site", "date", "q_cfs"])
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="site", values="q_cfs").sort_index()

df = usgs_dv_multisite(names.keys())

df.rename(names, axis=1, inplace=True)

from pathlib import Path
outdir = Path("data/daily_flows")

for col in df.columns:
    s = df[col].dropna()
    out = s.rename("value").to_frame()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()  # date-only
    out.index.name = "time"
    out.to_csv(outdir / f"daily_{col}.csv")
