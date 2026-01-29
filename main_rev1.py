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
        return (m*window)  #absolute slope

    return series.rolling(window).apply(slope, raw=True)

def walk_forward_drift_signal(
    df_pivot,
    master_tag,
    master_tag_online_val,
    date_search,
    sens_df,
    resample_freq="D",
    pct_range=0.05,
    slope_window=30,
    data_check_window=3,
    max_pct=2,
    threshold=2,   # added explicitly
):
    """
    End-to-end walk-forward drift analysis:
    preprocessing → resampling → online filtering → scaling → slope + signal
    """

    # ---------------- Load & preprocess ----------------
    df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
    df_pivot = df_pivot.sort_values("timestamp").set_index("timestamp")

    # ---------------- Resample ----------------
    df_resampled_unfiltered = df_pivot.resample(resample_freq).mean()

    # ---------------- Filter shutdown / offline data ----------------
    
    df_resampled = df_resampled_unfiltered[
        df_resampled_unfiltered[master_tag] > master_tag_online_val
    ]
    

    if df_resampled.empty:
        raise ValueError("No data available after online filtering")

    # ---------------- Walk-forward drift logic ----------------
    df_resampled = df_resampled.sort_index()
    date_search = pd.to_datetime(date_search)

    results = []

    for current_date in df_resampled.loc[date_search:].index:
        print(current_date)
        # ---------- Anchor master tag ----------
        master_val = df_resampled.at[current_date, master_tag]
        low = master_val * (1 - pct_range)
        high = master_val * (1 + pct_range)

        # ---------- Historical filter (NO FUTURE DATA) ----------
        hist_df = df_resampled.loc[:current_date]
        hist_df = hist_df[
            (hist_df[master_tag] >= low) &
            (hist_df[master_tag] <= high)
        ]

      

        # ---------- Scaling ----------
        scaler = StandardScaler()

        df_scaled = pd.DataFrame(
            scaler.fit_transform(hist_df),
            index=hist_df.index,
            columns=hist_df.columns,
        )

        # ---------- Remove extreme master-tag points ----------
        df_scaled = df_scaled[
            (df_scaled.abs() <= 4).all(axis=1)
        ]
        
        
        df_scaled = df_scaled.loc[
            (df_scaled[master_tag] >= -threshold) &
            (df_scaled[master_tag] <= threshold)
        ]
        
        # ---------- Minimum data check ----------
        if len(df_scaled) < slope_window-(len(hist_df)-len(df_scaled)) and slope_window-(len(hist_df)-len(df_scaled)) >20 :
            row = {}
            for col in df_resampled.select_dtypes(include="number").columns:
                row[f"{col}_value"] = df_resampled.at[current_date, col]
                row[f"{col}_pct_change"] = 0
                row[f"{col}_alert"] = 3
            results.append(pd.DataFrame(row, index=[current_date]))
            continue

        # ---------- Inverse scaling ----------
        df_scaled = pd.DataFrame(
            scaler.inverse_transform(df_scaled),
            index=df_scaled.index,
            columns=df_scaled.columns,
        )

        # ---------- Rolling slope ----------
        slope_df = pd.DataFrame(index=df_scaled.index)

        for col in df_scaled.select_dtypes(include="number").columns:
            slope_df[col] = (
                rolling_slope(df_scaled[col], ( slope_window-(len(hist_df)-len(df_scaled)))) / df_scaled[col]
            ) * 100

        # ---------- Signal + value ----------
        row = {}

        for col in df_scaled.select_dtypes(include="number").columns:
            print(col)
        
            # Tag value at current date
            row[f"{col}_value"] = df_resampled.at[current_date, col]
        
            # Current slope
            current_slope = slope_df[col].iloc[-1]
            abs_current_slope = abs(current_slope)
            row[f"{col}_pct_change"] = current_slope
        
            # ---------- Thresholds ----------
            if (col in sens_df.index and
                pd.notna(sens_df.at[col, "max_pct"])):
        
                raw_min_pct = sens_df.at[col, "max_pct"] / 3
                raw_max_pct = sens_df.at[col, "max_pct"]
            else:
                raw_min_pct = max_pct / 3
                raw_max_pct = max_pct
        
            # ---------- Enforce min = max/3 rule ----------
            tag_max_pct = raw_max_pct
            tag_min_pct = max(raw_min_pct, raw_max_pct / 3)
        
            # ---------- Rolling consistency (ABS) ----------
            last_n_all_gt_min = (
                (slope_df[col].abs() > tag_min_pct)
                .rolling(data_check_window, min_periods=data_check_window)
                .min()
                .iloc[-1]
            )
        
            # ---------- Signal logic (ABS) ----------
            if abs_current_slope < tag_min_pct:
                signal = 0
            elif abs_current_slope > tag_max_pct:
                signal = 2
            elif tag_min_pct <= abs_current_slope <= tag_max_pct and last_n_all_gt_min:
                signal = 1
            else:
                signal = 0
        
            row[f"{col}_alert"] = signal
        
        results.append(pd.DataFrame(row, index=[current_date]))

        # results.append(pd.DataFrame(row, index=pd.Index([current_date], name="timestamp")))

    # if not results:
    row = {}
    sd_df = df_resampled_unfiltered.loc[date_search:]

    missing_dates = sd_df.index.difference(df_resampled.index)

    for current_date in missing_dates:
        row = {}
    
        for col in sd_df.select_dtypes(include="number").columns:
            row[f"{col}_value"] = sd_df.at[current_date, col]
            row[f"{col}_pct_change"] = 0
            row[f"{col}_alert"] = 4

        results.append(pd.DataFrame(row, index=[current_date]))

    return pd.concat(results),df_resampled_unfiltered

def drift_data(schema, alertx_id,db_connection,SOR,time_upto):
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
              WHERE a.timestamp >='{SOR}' AND a.timestamp < '{time_upto}'
            ORDER BY a.timestamp;
            '''
    drift_check_df = pd.read_sql(query, db_connection)
    
    return drift_check_df

def desc(schema,alertx_id):
    desc_query = f'''
        SELECT 
            a.alias_name,
            a.description, 
            a.tag_name, 
            b.tag_name AS IP21_Tag,
            b.uom,
            'taglist' AS tag_source
        FROM {schema}.alertx_system_config a
        LEFT JOIN {schema}.icap_independent_table_taglist b
            ON a.tag_name = b.alias
        WHERE a.alertx_id = {alertx_id}
    
        UNION ALL
    
        SELECT 
            a.alias AS alias_name,
            a.description,
            NULL AS tag_name,
            NULL AS IP21_Tag,
            NULL AS uom,
            'calculated tag' AS tag_source
        FROM {schema}.alertx_system_config_calculation_tag a
        LEFT JOIN {schema}.icap_independent_table_calculationtag calc
            ON a.alias = calc.alias
        WHERE a.alertx_id = {alertx_id}
        '''
        
    desc_df = pd.read_sql(desc_query, db_connection)
    desc_df = desc_df.fillna("")
    
    return desc_df

def make_alias_unique(df, alias_col='alias'):
    df = df.copy()
    df[alias_col] = (
        df.groupby(alias_col).cumcount()
        .astype(str)
        .radd('_')
        .where(df.duplicated(alias_col, keep=False), '')
        .radd(df[alias_col])
    )
    return df

def replace_alias_with_desc(col):
    for alias, desc in sorted_aliases:
        if col == alias or col.startswith(alias + "_"):
            return col.replace(alias, desc, 1)
    return col


################################ READ CONFIG FILES ################################


parser = configparser.ConfigParser()
parser.read(filenames=["config/config.ini"])

# ---- DB Section ----
db_config   = dict(parser["DB"].items())
host        = parser.get("DB", "host")
user        = parser.get("DB", "user")
password    = parser.get("DB", "pass")
output_path = parser.get("DB", "output_path")
team        = parser.get("DB", "team")

report_run_date = pd.read_csv("config/report_run_dates.csv")
start_time_dt = pd.to_datetime(report_run_date['start_time'][0])
end_time_dt = pd.to_datetime(report_run_date['end_time'][0])
eqpt_df         = pd.read_csv("config/equipment_config.csv")
overall_sens_df         = pd.read_csv("config/signal_gen_min_max.csv").set_index('tag')

################################ START OF CODE EXCECUTION ################################



for k , eqpt in eqpt_df.iterrows():

    alertx_id               = eqpt['alertx_id']
    schema                  = eqpt['schema']
    master_tag_online_val   = eqpt['master_tag_online_value']
    master_tag              = eqpt['master_tag']
    SOR                     = pd.to_datetime(eqpt['SOR_date']).strftime('%Y-%m-%d %H:%M:%S')
    print(alertx_id,schema)
    
    alert_output_path = (
    f"{output_path}"
    f"{user}_{schema}_alertx_id_{alertx_id}_alert_data_"
    f"{start_time_dt.strftime('%Y-%m-%d')}"
    f"_to_"
    f"{end_time_dt.strftime('%Y-%m-%d')}.xlsx")
# alert_output_path = "output/alert_data.xlsx"

    # Initialize database connection
    db_connection = db_conn(
            db_config["host"],
            db_config["user"],
            db_config["pass"],
            schema,
        )
    print("Database connection successful")
    
    
    #------ Processed Drift Data --------
    sens_df = overall_sens_df[overall_sens_df['alertx_id']==alertx_id]
    
    df = drift_data(schema, alertx_id, db_connection,SOR,end_time_dt)
    
    df_updated = df[['timestamp','alias','actual_value']]
    
    df_pivot = df_updated.pivot_table(
        index='timestamp',
        columns='alias',
        values='actual_value',
        aggfunc='first'   # or 'mean', 'max', etc.
    )
    
    df_pivot = df_pivot.reset_index()
    
    
    #------ Processed Drift Data --------
    daily_drift_alerts,drift_daily_data = walk_forward_drift_signal(
        df_pivot=df_pivot,
        master_tag=master_tag,
        master_tag_online_val=master_tag_online_val,
        date_search=start_time_dt,
        sens_df=sens_df,
    )
    
    daily_drift_alerts = daily_drift_alerts.sort_index()
    
    
    results_df = daily_drift_alerts  # for Summary DF making
    
    #--------- Call Description Mapping DF--------#
    desc_df = desc(schema,alertx_id)
    desc_df   = make_alias_unique(desc_df, alias_col='description')
    
    
    #--------- Create Dict of Alias to COrreponding DICT-----------#
    alias_to_desc = (
        desc_df
        .dropna(subset=["alias_name", "description"])
        .set_index("alias_name")["description"]
        .to_dict()
    )

    # longest alias first to avoid partial replacement
    sorted_aliases = sorted(alias_to_desc.items(), key=lambda x: len(x[0]), reverse=True)
    
    results_df.columns = [replace_alias_with_desc(c) for c in results_df.columns]
    
    
    #---------------------- GENERATION OF SUMMARY--------------------#
    summary_dict = {}
    desc_df = desc_df.set_index('description')
    for col in results_df.columns:
        if not col.endswith("_alert"):
            continue

        tag = col.replace("_alert", "")
        alert_col = f"{tag}_alert"
        value_col = f"{tag}_value"
        pct_col   = f"{tag}_pct_change"

        # Filter rows where alert is NOT 0 or 3or 4
        df_tag = results_df.loc[~results_df[alert_col].isin([0, 3, 4])].copy() # 0-No alert , 3-No Sufficient Data , 4-Offline
        if df_tag.empty:
            continue
        
        tag_label = f"{tag} ({desc_df.loc[tag,'IP21_Tag']})".rstrip(" ()")

        # Determine trend
        trend = pd.Series(
            np.where(df_tag[pct_col] > 0, "increasing", "decreasing"),
            index=df_tag.index
        )

        # Base calculations
        typical_value = (df_tag[value_col] * (1 + (df_tag[pct_col] / 100))).round(2)
        current_value = df_tag[value_col].round(2)

        # # Swap values if trend is decreasing
        # typical_value = pd.Series(
        #     np.where(trend == "decreasing", current_calc, typical_calc),
        #     index=df_tag.index
        # )
        

        # current_value = pd.Series(
        #     np.where(trend == "decreasing", typical_calc, current_calc),
        #     index=df_tag.index
        # )

        # Timestamp text
        timestamps = df_tag.index.to_series().dt.strftime("%d-%m-%Y")

        # Build alert sentences
        sentences = (
            tag_label
            + " is indicating an "
            + trend
            + " trend from its typical value "
            + current_value.astype(str)
            + f" {desc_df.loc[tag,'uom']} to "
            + typical_value.astype(str)
            + f" {desc_df.loc[tag,'uom']} on "
            + timestamps
        )

        # Store sentences
        for ts, text in sentences.items():
            summary_dict.setdefault(ts, []).append(text)


    summary_df = pd.DataFrame(
        {
            "message": [
                msg
                for msgs in summary_dict.values()
                for msg in msgs
            ]
        },
        index=pd.Index(
            ["Drift Alert"] * sum(len(v) for v in summary_dict.values()),
            name="Alert type"
        )
    )
    
    with pd.ExcelWriter(alert_output_path) as writer:
        daily_drift_alerts.to_excel(writer, sheet_name="drift_alerts", index=True)
        summary_df.to_excel(writer, sheet_name="Drift Summary", index=True)
        drift_daily_data.to_excel(writer, sheet_name="drift_daily_data", index=True)
        
#-----------------------------------------------------------------------------------------------------------------


#







#-----------------------------------------------------

    
# alias_to_desc = (
#     desc_df
#     .dropna(subset=["alias_name", "description"])
#     .set_index("alias_name")["description"]
#     .to_dict()
# )

# def replace_alias_with_desc(col):
#     for alias, desc in alias_to_desc.items():
#         if col.startswith(alias):
#             return col.replace(alias, desc, 1)
#     return col





# daily_drift_alerts.columns = [replace_alias_with_desc(c) for c in daily_drift_alerts.columns]
