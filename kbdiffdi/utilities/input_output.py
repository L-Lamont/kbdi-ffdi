import os
import csv
import datetime

import numpy as np
# from osgeo import gdal
# from osgeo import ogr
# from osgeo import osr

from kbdiffdi.utilities import conversion
from kbdiffdi.features import feature

def load_csv(filename):
    """
    Load daily meteorological data from a csv.

    Columns are located by header name, so several layouts are supported:
      * station files (Knysna / Plett / George / PortElizabeth) with YYYYMMDD
        dates, temperature in degrees C and wind in m/s; and
      * climate-model output (CMIP-style, e.g. KACE) with DD/MM/YYYY dates,
        a "TMax ... K" temperature in Kelvin and a wind column already in km/h.

    Units are normalised automatically: temperature that looks like Kelvin is
    converted to Celsius, and wind given in m/s is converted to km/h (a column
    already in km/h is used as-is). Trailing or interior rows with missing /
    sentinel values are dropped by keeping the longest gap-free run of valid
    days, since the KBDI calculation needs continuous daily data.

    Parameters:
    ------------
    filename: str
        the full path and filename of the input csv

    Returns:
    --------
    rain: feature.RasterStack
        daily precipitation data in millimeters
    temp: feature.RasterStack
        daily temperature data in celsius
    relhum: feature.RasterStack
        daily relative humidity data in percent
    wind: feature.RasterStack
        daily wind speed data in kilometers per hour
    """
    indata = np.loadtxt(filename, dtype=str, delimiter=",", encoding="latin-1")

    header = [h.strip().lower() for h in indata[0]]
    def find_col(keywords, required=True):
        for i, name in enumerate(header):
            if any(k in name for k in keywords):
                return i
        if required:
            raise ValueError("could not find a column matching %s in header %s"
                             % (keywords, indata[0].tolist()))
        return None

    date_col = find_col(["date", "yyyy"])
    rain_col = find_col(["rainfall", "rain", "tp_mm", "precip"])
    # prefer an explicit maximum-temperature column (the indices want daily max)
    temp_col = find_col(["tmax", "temp"])
    hum_col = find_col(["humid"])
    # wind: use a km/h column directly if present, otherwise read m/s and convert
    wind_col = find_col(["kmph", "km/h", "kmh"], required=False)
    wind_in_ms = wind_col is None
    if wind_in_ms:
        wind_col = find_col(["wind"])

    # --- date parsing: support YYYYMMDD and slash-separated dates ---------
    date_strs = [r[date_col].strip() for r in indata[1:]]
    if any("/" in s for s in date_strs):
        # slash dates; decide day-first vs month-first from the data itself
        firsts = [int(s.split("/")[0]) for s in date_strs if "/" in s]
        day_first = max(firsts) > 12  # a value > 12 can only be a day
        def parse_date(s):
            a, b, c = (int(x) for x in s.strip().split("/"))
            return datetime.datetime(year=c, month=(b if day_first else a),
                                     day=(a if day_first else b))
    else:
        def parse_date(s):
            s = s.strip()
            return datetime.datetime(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]))

    # --- drop rows with missing / sentinel values -------------------------
    data = indata[1:]
    needed = [rain_col, temp_col, hum_col, wind_col]
    def _num(s):
        # tolerate a trailing percent sign / surrounding whitespace (e.g. "27.06%")
        return s.replace("%", "").strip()
    def row_valid(r):
        for c in needed:
            v = _num(r[c])
            if v == "":
                return False
            try:
                float(v)
            except ValueError:
                return False
        # reject absolute-zero temperature sentinels (e.g. -273.15 from 0 K)
        return float(_num(r[temp_col])) > -100.0
    valid = [row_valid(r) for r in data]

    # keep the longest contiguous run of valid days (continuity matters for KBDI)
    best_start, best_len, cur_start = 0, 0, None
    for i, ok in enumerate(valid + [False]):
        if ok and cur_start is None:
            cur_start = i
        elif not ok and cur_start is not None:
            if i - cur_start > best_len:
                best_start, best_len = cur_start, i - cur_start
            cur_start = None
    if best_len == 0:
        raise ValueError("no valid rows found in %s" % filename)
    data = data[best_start:best_start + best_len]
    dropped = len(valid) - best_len
    if dropped:
        print("[INFO] kept %d valid day(s) (%s .. %s); dropped %d row(s) with missing/sentinel data"
              % (best_len, data[0][date_col].strip(), data[-1][date_col].strip(), dropped))

    datelist = [parse_date(r[date_col]) for r in data]
    def _col_to_float(col):
        # tolerate a trailing percent sign / whitespace on numeric cells (e.g. "27.06%")
        return np.char.strip(np.char.replace(data[:, col].astype(str), "%", "")).astype(float)
    out_rainfall = _col_to_float(rain_col).reshape(-1, 1, 1, 1)
    out_rel_hum = _col_to_float(hum_col)
    # the FFDI equation expects relative humidity as a percentage (0-100). Some
    # sources give it as a 0-1 fraction instead; if the values clearly sit on a
    # 0-1 scale, rescale to percent. (A genuine percentage record reaches ~30+.)
    if np.max(out_rel_hum) <= 1.0:
        out_rel_hum = out_rel_hum * 100.0
        print("[INFO] relative humidity looks like a 0-1 fraction; converted to percent")
    out_rel_hum = out_rel_hum.reshape(-1, 1, 1, 1)
    out_wind = _col_to_float(wind_col).reshape(-1, 1, 1, 1)
    out_temp = _col_to_float(temp_col)
    # convert temperature from Kelvin to Celsius when it clearly looks like Kelvin
    if np.median(out_temp) > 150:
        out_temp = out_temp - 273.15
        print("[INFO] temperature looks like Kelvin; converted to Celsius")
    out_temp = out_temp.reshape(-1, 1, 1, 1)

    # create the featureStacks
    rain = feature.RasterStack()
    rain.create_sc_stack(out_rainfall, datelist, None, "standard", 0, 0, 1, -1)
    temp = feature.RasterStack()
    temp.create_sc_stack(out_temp, datelist, None, "standard", 0, 0, 1, -1)
    relhum = feature.RasterStack()
    relhum.create_sc_stack(out_rel_hum, datelist, None, "standard", 0, 0, 1, -1)
    wind = feature.RasterStack()
    wind.create_sc_stack(out_wind, datelist, None, "standard", 0, 0, 1, -1)
    if wind_in_ms:
        conversion.mpers_to_kmperh(wind)
    return rain, temp, relhum, wind

def write_kbdi(inputfilename, outputfilename, KBDIobject):
    kbdi = KBDIobject.data.flatten()
    with open(inputfilename, "r", encoding="latin-1") as inputcsv:
        with open(outputfilename, "w", newline="", encoding="latin-1") as outputcsv:
            reader = csv.reader(inputcsv, delimiter=",")
            writer = csv.writer(outputcsv, delimiter=",")
            index = 0
            firstRow = True
            for row in reader:
                if firstRow:
                    firstRow = False
                    row.extend(["KBDI"])
                else:
                    row.extend([kbdi[index]])
                    index+=1
                writer.writerow(row)

def write_csv(inputfilename, outputfilename, KBDIobject=None, FFDIobject=None, DFobject=None, spinup_days=0):
    if KBDIobject:
        kbdi = KBDIobject.data.flatten()
    if FFDIobject:
        ffdi = FFDIobject.data.flatten()
    if DFobject:
        df = DFobject.data.flatten()
    with open(inputfilename, "r", encoding="latin-1") as inputcsv:
        with open(outputfilename, "w", newline="", encoding="latin-1") as outputcsv:
            reader = csv.reader(inputcsv, delimiter=",")
            writer = csv.writer(outputcsv, delimiter=",")
            # number of computed days; the loader may have dropped trailing rows
            # with missing data, so there can be fewer values than input rows.
            n_values = len(kbdi)
            index = 0
            firstRow = True
            for row in reader:
                if firstRow:
                    firstRow = False
                    row.extend(["KBDI", "DF", "FFDI"])
                    writer.writerow(row)
                else:
                    # index tracks the day; advance it for every input data row.
                    # write only rows that have a computed value (index < n_values)
                    # and are past the spin-up window. (Assumes the kept day-span
                    # starts at the first data row, which holds for the supported
                    # formats - the loader logs anything it drops.)
                    if spinup_days <= index < n_values:
                        row.extend([kbdi[index],df[index],ffdi[index]])
                        writer.writerow(row)
                    index+=1
                    