# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 16:20:53 2026

@author: skanojia
"""
# ----------------------------- Imports -----------------------------
import os
import numpy as np
import pandas as pd
import openpyxl
import configparser
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sqlalchemy import create_engine
import sys
import re
from sklearn.preprocessing import StandardScaler


# ----------------------- DB Connection Setup ------------------------
def db_conn(host, user, pwd, schema):
    """
    Creates a database connection using provided credentials.
    ---------------------------------------------------------
    Input : host [str], user [str], pwd [str], schema [str]
    Output: SQLAlchemy connection engine
    """
    db_engine = create_engine(f"mysql+pymysql://{user}:{pwd}@{host}/{schema}")
    return db_engine

# SLOPE FUNCTION       
def rolling_slope(series, window):
    x = np.arange(window)

    def slope(y):
        m, _ = np.polyfit(x, y, 1)
        return abs(m*window)  #absolute slope

    return series.rolling(window).apply(slope, raw=True)

def process_drift_data(
    file_path,
    master_tag,
    master_tag_online_val,
    sens_df,
    threshold=2,
    load_min_pct=0.98,        # % of latest master tag value
    load_max_pct=1.02,        # % of latest master tag value
    resample_freq="D",        # "H", "D", "W", "M"
    slope_window=30,
    data_check_window=10,
    min_pct=0.5,
    max_pct=2,
):

    # ---------------- Load & preprocess ----------------
    df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
    df_pivot = df_pivot.sort_values("timestamp").set_index("timestamp")

    # ---------------- Resample ----------------
    df_resampled = df_pivot.resample(resample_freq).mean()

    # ---------------- Filter shutdown data ----------------
    df_resampled = df_resampled[df_resampled[master_tag] > master_tag_online_val]

    # ---------------- Date filter ----------------
    # df_resampled = df_resampled.loc[SOR:EOR]

    if df_resampled.empty:
        raise ValueError("No data available after online/date filtering")

    # ---------------- Load range filter (LATEST MASTER TAG BASED) ----------------
    latest_master_value = df_resampled[master_tag].iloc[-1]

    load_min = latest_master_value * load_min_pct
    load_max = latest_master_value * load_max_pct
    
    loan_min = 974 
    load_max = 976

    df_load_filter = df_resampled[
        (df_resampled[master_tag] >= load_min) &
        (df_resampled[master_tag] <= load_max)
    ]

    if df_load_filter.empty:
        raise ValueError("No data available after load range filtering")

    # ---------------- Scaling ----------------
    scaler = StandardScaler()

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_load_filter),
        index=df_load_filter.index,
        columns=df_load_filter.columns,
    )

    # ---------------- Remove extreme points ----------------
    df_scaled = df_scaled.loc[
        (df_scaled[master_tag] >= -threshold) &
        (df_scaled[master_tag] <= threshold)
    ].copy()

    # ---------------- Inverse scaling ----------------
    df_scaled = pd.DataFrame(
        scaler.inverse_transform(df_scaled),
        index=df_scaled.index,
        columns=df_scaled.columns,
    )

    # ---------------- Rolling slope ----------------
    for col in df_scaled.select_dtypes(include="number").columns:
        df_scaled[f"{col}_slope"] = (
            rolling_slope(df_scaled[col], slope_window) / df_scaled[col]
        ) * 100

    # ---------------- Signal generation ----------------
    
    
    slope_cols = [c for c in df_scaled.columns if c.endswith("_slope")]

    for col in slope_cols:

        base_tag = col.replace("_slope", "")
    
        # ---------- Get min/max ----------
        if (base_tag in sens_df.index and
            pd.notna(sens_df.at[base_tag, "min_pct"]) and
            pd.notna(sens_df.at[base_tag, "max_pct"])):
    
            tag_min_pct = sens_df.at[base_tag, "min_pct"]
            tag_max_pct = sens_df.at[base_tag, "max_pct"]
    
        else:
            tag_min_pct = min_pct
            tag_max_pct = max_pct
    
        # ---------- Logic ----------
        last_n_all_gt_min = (
            (df_scaled[col] > tag_min_pct)
            .rolling(data_check_window, min_periods=data_check_window)
            .min()
            .fillna(0)
            .astype(bool)
        )
    
        signal_col = col.replace("_slope", "_signal")
    
        df_scaled[signal_col] = np.select(
            condlist=[
                df_scaled[col] < tag_min_pct,
                df_scaled[col] > tag_max_pct,
                (df_scaled[col].between(tag_min_pct, tag_max_pct)) & last_n_all_gt_min,
                (df_scaled[col].between(tag_min_pct, tag_max_pct)) & (~last_n_all_gt_min),
            ],
            choicelist=[0, 2, 1, 3],
            default=0,
        )


    return df_scaled


def drift_data(schema, alertx_id,db_connection,SOR):
    query = f'''SELECT
              a.timestamp,
              a.alias,
              a.actual_value,
              c.description,
              c.uom
            FROM {schema}.alertx_final_alert_output_{alertx_id} a
            LEFT JOIN 
            {schema}.alertx_system_config c
              ON a.alias = c.alias_name
              AND c.alertx_id = {alertx_id}
              WHERE a.timestamp > '{SOR}'
            ORDER BY a.timestamp;
            '''
    drift_check_df = pd.read_sql(query, db_connection)
    
    return drift_check_df

# def drift_data(schema, alertx_id, db_connection, window, SOR=None, EOR=None):

#     base_query = f"""
#         SELECT
#             a.timestamp,
#             a.alias,
#             a.actual_value,
#             c.description,
#             c.uom
#         FROM {schema}.alertx_final_alert_output_{alertx_id} a
#         LEFT JOIN {schema}.alertx_system_config c
#           ON a.alias = c.alias_name
#          AND c.alertx_id = {alertx_id}
#     """

#     # Case 1: Explicit start and end
#     if SOR and EOR:
#         where_clause = f"""
#             WHERE a.timestamp BETWEEN '{SOR}' AND '{EOR}'
#         """

#     # Case 2: Only EOR → window is number of DAYS
#     elif EOR:
#         where_clause = f"""
#             WHERE a.timestamp BETWEEN
#                   DATE_SUB('{EOR}', INTERVAL {window} DAY)
#               AND '{EOR}'
#         """

#     else:
#         where_clause = ""

#     final_query = base_query + where_clause + """
#         ORDER BY a.timestamp ASC
#     """

#     return pd.read_sql(final_query, db_connection)




################################ START OF CODE EXCECUTION ################################


parser = configparser.ConfigParser()
parser.read(filenames=["config/config.ini"])

# ---- DB Section ----
db_config   = dict(parser["DB"].items())
host        = parser.get("DB", "host")
user        = parser.get("DB", "user")
password    = parser.get("DB", "pass")
output_path = parser.get("DB", "output_path")
team        = parser.get("DB", "team")

# ---- Config Section ----
master_tag_online_val = parser.getint("Config", "master_tag_online_val")
threshold             = parser.getint("Config", "threshold")
window                = parser.getint("Config", "window")
data_check_window     = parser.getint("Config", "data_check_window")

SOR = pd.to_datetime(parser.get("Config", "SOR")).strftime('%Y-%m-%d %H:%M:%S')
EOR = pd.to_datetime(parser.get("Config", "EOR")).strftime('%Y-%m-%d %H:%M:%S')

master_tag = parser.get("Config", "master_tag")



eqpt_df         = pd.read_csv("config/equipment_config.csv")
sens_df         = pd.read_csv("config/signal_gen_min_max.csv")
alertx_id       = eqpt_df['alertx_id'][0]
schema          = eqpt_df['schema'][0]

# Initialize database connection
db_connection = db_conn(
        db_config["host"],
        db_config["user"],
        db_config["pass"],
        schema,
    )
print("Database connection successful")

#------ Processed Drift Data --------

df = drift_data(schema, alertx_id, db_connection,SOR )

df_updated = df[['timestamp','alias','actual_value']]

df_pivot = df_updated.pivot_table(
    index='timestamp',
    columns='alias',
    values='actual_value',
    aggfunc='first'   # or 'mean', 'max', etc.
)

df_pivot = df_pivot.reset_index()



df_daily = process_drift_data(
    file_path="input/All_tag_drift_check.xlsx",
    master_tag=master_tag,
    master_tag_online_val=master_tag_online_val,
    sens_df=sens_df,
    threshold=2,
    load_min_pct=0.98,
    load_max_pct=1.02,
    resample_freq="D",      #change here: "H", "D", "W", "M"
    slope_window=30,
    data_check_window=3,
    min_pct=1,
    max_pct=3)
