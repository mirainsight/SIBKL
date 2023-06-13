#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="Ministry Health", page_icon="🎸")
st.title("Zone Health")
st.header("Filter Data")

# text_input = st.text_input(
#     "Label your database 👇",
# )   
# st.title(text_input)
# uploaded_file = st.file_uploader("Choose a file")


    
# if uploaded_file is not None:

#     # Can be used wherever a "file-like" object is accepted:
dataframe2 = pd.read_excel("Ministry_Health_Consolidated.xlsx")
dataframe2 = dataframe2.replace(np.nan,'',regex=True)
#     #st.write(dataframe)
#     dataframe2 = pd.read_excel(uploaded_file, sheet_name=0)
#     dataframe2.dropna( inplace=True)

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
# if uploaded_file is None: 
#     dataframe2 = pd.read_excel("C:/Users/Mirac/OneDrive/Documents/SIB/Cell_Health.xlsx")


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """


    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()
    

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_numeric_dtype(df[column]) and df[column].nunique()>1:
                _min = int(df[column].min())
                _max = int(df[column].max())
                step = (_max - _min) / 100
                options = df[column].unique().tolist()
                options.sort()
                user_num_input = right.select_slider(
                f"Values for {column}",
                options=options,
                value=[_min,_max],
                )
                df = df[df[column].between(*user_num_input)]
            elif is_numeric_dtype(df[column]):
                st.write('hi mira')
                _min = int(df[column].min())
                options = df[column].unique().tolist()
                options.sort()
                user_num_input = right.select_slider(
                f"Values for {column}",
                options=options,
                value=[_min,_min],
                )
                df = df[df[column]==user_num_input]

            elif is_categorical_dtype(df[column]) or df[column].nunique() < 200:
                selected_all = st.checkbox('Select All', key=f"{column}")                  
                values = df[column].unique().tolist()
                values.sort()
                if selected_all: 
                    user_cat_input = right.multiselect(
                    f"Values for {column}",
                    values,
                    default=list(df[column].unique()),
                    )
                else:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        values,
                        #default=list(df[column].unique()),
                    )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                    )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

#    except: 
#        st.write("Please check input.")
    return df

filtered_df = filter_dataframe(dataframe2)
df = dataframe2
percentage = round((len(filtered_df)/len(df))*100, 0)   
filtered_df = filtered_df.sort_values(by=filtered_df.columns.tolist())

st.divider()
st.header("Filtered Data")
show_data = st.checkbox('Show Data')
choose_columns = st.checkbox("Choose Columns")
if show_data and choose_columns:
    to_filter_columns_inc = st.multiselect("I want", df.columns)
    filtered_df = filtered_df.sort_values(by=to_filter_columns_inc)
    st.dataframe(filtered_df[to_filter_columns_inc])
    st.header(f"There are {len(filtered_df)} such cells ({percentage}% of all cells).")
elif show_data: 
    st.dataframe(filtered_df)
    st.subheader(f"There are {len(filtered_df)} such cells ({percentage}% of all cells).")



