import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from io import StringIO
from datetime import datetime
from datetime import timedelta
from datetime import date
from dateutil.relativedelta import relativedelta  



st.set_page_config(page_title="Cell Grow", page_icon="🏫")
st.title("Cell Grow")
st.header("Filter Data")

# text_input = st.text_input(
#     "Label your database 👇",
# )   
# st.title(text_input)
# uploaded_file = st.file_uploader("Choose a file")



# if uploaded_file is not None:

#     # Can be used wherever a "file-like" object is accepted:
dataframe2 = pd.read_csv("Cell_Grow.csv")
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
                df[col] = pd.to_datetime(df[col], format="%d/%m/%Y")
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()
    time_container = st.container()
           
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

            elif column == "Groups":
                selected_all = st.checkbox('Select All', key=f"{column}")                  
                values = ["ALL", "CG", "Ministry", "Leaders", "Anchor Street", "His Street Makers", "Home Street", "King Street",
                "Legacy Street", "Life Street", "Royal Street", "Street Conquerors", "Street Fire", "Street Food", "Street Lights",
                "Street Salt", "Via Dolorosa Street", "Core Team", "Frontline", "HYPE", "Visual Storytellers", "Worship", "Newcomers"]
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
                    for i in user_cat_input: 
                        df = df[df[column].str.contains(i)]
                        
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
                    for i in user_cat_input: 
                        df = df[df[column].str.contains(i)]

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
                    user_date_input = tuple(map(pd.to_datetime, user_date_input, format="%d/%m/%Y"))
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
st.header("Specify Time Frame")
col_names = dataframe2.columns.values.tolist()
service_times = list()
required_times_columns = list()
for i in range(0, len(col_names)): 
    if len(col_names[i]) > 35: 
        time = col_names[i][6:10]
        required_times_columns.append(col_names[i])
        time = datetime.strptime(time[0:2]+"/"+time[2:10]+"/23", '%d/%m/%y')
        service_times.append(time)
service_times.sort()
service_times = pd.Series(service_times)  
service_times = service_times.dt.date
user_time_input = st.selectbox(label="Pick your time period", options=["1 week", "1 month", "3 months", "6 months", "1 year", "All time", "Custom"])

if user_time_input == "1 week":
    end_date = datetime.today().strftime('%d/%m/%y')
    end_date = datetime.strptime(end_date, '%d/%m/%y').date()
    start_date = datetime.now() + timedelta(days=-7)
    start_date = start_date.date()
elif user_time_input == "1 month":
    end_date = datetime.today().strftime('%d/%m/%y')
    end_date = datetime.strptime(end_date, '%d/%m/%y')
    start_date = end_date - relativedelta(months = 1) 
    end_date = end_date.date()
    start_date = start_date.date()

elif user_time_input == "3 months":
    end_date = datetime.today().strftime('%d/%m/%y')
    end_date = datetime.strptime(end_date, '%d/%m/%y')
    start_date = end_date - relativedelta(months = 3) 
    end_date = end_date.date()
    start_date = start_date.date()   

elif user_time_input == "6 months":
    end_date = datetime.today().strftime('%d/%m/%y')
    end_date = datetime.strptime(end_date, '%d/%m/%y')
    start_date = end_date - relativedelta(months = 6) 
    end_date = end_date.date()
    start_date = start_date.date()   

elif user_time_input == "1 year":
    end_date = datetime.today().strftime('%d/%m/%y')
    end_date = datetime.strptime(end_date, '%d/%m/%y')
    start_date = end_date - relativedelta(months = 12) 
    end_date = end_date.date()
    start_date = start_date.date()  

elif user_time_input == "All time":
    end_date = service_times.max()
    start_date = service_times.min()  

elif user_time_input == "Custom":

    st.write(f"Start date: {service_times.min()} - End date: {service_times.max()}")
    user_date_input = st.date_input(
        f"Choose your time period",
        value=(
            service_times.min(),
            service_times.max(),
        ),
    )

    start_date, end_date = user_date_input
st.write(f"Showing dates from: {start_date}-{end_date}")
list_service_times = service_times.tolist()
if start_date > service_times.max(): 
    st.write("There are no such services.")
else: 
    start_index = None
    end_index = None
    for date in list_service_times: 
        if date >= start_date: 
            start_index = list_service_times.index(date)
            break
    reversed_service = list_service_times.copy()
    reversed_service.sort(reverse=True)
    for date in reversed_service: 
        if date <= end_date and (date >= start_date): 
            end_index = list_service_times.index(date)
            break
    if start_index == None or end_index == None: 
        st.write("Range too small. Please try a bigger range.")
    test_list = []
    services_count = 0
    for i in range(start_index, end_index+1): 
        i = st.checkbox(str(col_names[i+6]), value=1)
        services_count += 1
        test_list.append(i)
    accepted_columns = col_names[0:4]
    for i in range(start_index, end_index+1): 
        accepted_columns.append(col_names[i+6])
#                 df[col_names[i+5]] = df[col_names[i+5]].replace(['Y'], 1)
#                 df[col_names[i+5]] = df[col_names[i+5]].replace(['N'], 0)
    df = df[accepted_columns]
    df['Attended'] = df.apply(lambda x: str(x.eq("Y").sum())+'/'+str(services_count), axis=1)
    df['Absent'] = df.apply(lambda x: str(x.eq("N").sum())+'/'+str(services_count), axis=1)
    accepted_columns.insert(4, 'Attended')
    accepted_columns.insert(5, 'Absent')
    df = df[accepted_columns]
st.divider()
st.header("Filtered Data")
show_data = st.checkbox('Show Data')
choose_columns = st.checkbox("Choose Columns")
if show_data and choose_columns:
    to_filter_columns_inc = st.multiselect("I want", df.columns)
    filtered_df = filtered_df.sort_values(by=to_filter_columns_inc)
    st.dataframe(filtered_df[to_filter_columns_inc])
    st.header(f"There are {len(filtered_df)} such members(s) ({percentage}% of all members).")
elif show_data: 
    st.dataframe(filtered_df)
    st.subheader(f"There are {len(filtered_df)} such members(s) ({percentage}% of all members).")
