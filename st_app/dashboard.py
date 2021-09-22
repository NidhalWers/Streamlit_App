import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from functools import wraps

def log_time(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        f = open("log_dev.txt",'a',encoding="utf8")
        time_res = end - start
        f.write("\n"+func.__name__+ " time = " + str(time_res))
        return result

    return wrapper

def finished_log():
    f = open("log.txt", 'a', encoding="utf8")
    f.write("\n\n")
    for i in range(150):
        f.write("-")
    f.write("\n\n")


st.title("this is the dashboard of the lab 1")
if st.sidebar.button("uber data"):
    st.subheader("uber data")

    file_path = "uber-raw-data-apr14.csv"

    @log_time
    def read_file(file_path):
        return pd.read_csv(file_path, delimiter=',')

    df = read_file(file_path)

    st.write(df.head(5))
#########
    st.subheader("Data transformation")

    @log_time
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def map_to_datetime(dataframe):
        dataframe['Date/Time'] = dataframe['Date/Time'].map(pd.to_datetime)
        return dataframe

    df = map_to_datetime(df)
    expander = st.expander("Dataframe after the to_datetime mapping")
    expander.write(df.head(5))

    def get_dom(dt):
        return dt.day

    @log_time
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def add_column_dom(dataframe):
        dataframe['dom'] = dataframe['Date/Time'].map(get_dom)
        expander = st.expander("Dataframe after adding dom column")
        expander.write(dataframe.head(5))
        return dataframe
    df = add_column_dom(df)



    def get_weekday(dt):
        return dt.weekday()


    @log_time
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def add_column_Date_Time(dataframe):
        dataframe['weekday']= dataframe['Date/Time'].map(get_weekday)
        expander = st.expander("Dataframe after adding weekday column")
        expander.write(dataframe.head(5))
        return dataframe
    df = add_column_Date_Time(df)



    def get_hours(dt):
        return dt.hour


    @log_time
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def add_column_hours(dataframe):
        dataframe['hours'] = dataframe['Date/Time'].map(get_hours)
        expander = st.expander("Dataframe after adding hours column")
        expander.write(dataframe.head(5))
        return dataframe
    df = add_column_hours(df)


    @log_time
    def add_column_frequency(dataframe):
        dataframe['frequency']= dataframe['dom'].map(dataframe['dom'].value_counts())
        expander = st.expander("Dataframe after adding frequency column")
        expander.write(dataframe.head(5))
        return dataframe
    df = add_column_frequency(df)
#########
    st.subheader("Visual representation")


    @log_time
    def plot_frequency_by_dom(dataframe):
        figure, ax = plt.subplots()
        ax.hist(dataframe['dom'], bins=30, rwidth=0.8, range=(0.5,30.5))
        plt.title("Frequency by DoM - Uber - April 2014")
        plt.xlabel("Date of the month")
        plt.ylabel("Frequency")
        return figure

    figure = plot_frequency_by_dom(df)
    expander_plot = st.expander("Frequency by DoM - Uber - April 2014")
    expander_plot.write(figure)

    st.write("Data grouped by date")

    def count_rows(rows):
        return len(rows)

    @log_time
    def group_by_dom(dataframe):
        dataframe.groupby("dom").apply(count_rows)
        expander = st.expander("Dataframe after grouping the data by date of month")
        expander.write(dataframe.head(5))
    group_by_dom(df)

    # Use plot, bar function to plot the data by date
    @log_time
    def group_by_date(dataframe):
        figure = plt.figure()
        y = dataframe.groupby("dom").apply(count_rows)
        plt.bar(range(1,31),y)
        plt.title("data grouped by Date")
        plt.xlabel("Date")
        #expander_plot = st.expander("data grouped by Date in bar")
        #expander_plot.write(figure)
        plt.plot(range(1,31), y, color = 'tab:blue')
        return figure

    figure = group_by_date(df)
    expander_plot = st.expander("data grouped by Date in plot")
    expander_plot.write(figure)


    st.write("data grouped by the day of month")

    # Sort the data by date and use bar function to plot the sorted data by date (.sort_values())
    @log_time
    def sort_by_date(dataframe):
        figure = plt.figure()
        y = dataframe.groupby("dom").apply(count_rows).sort_values()
        plt.bar(range(1,31),y)
        #expander = st.expander("Dataframe after sorting by date in bar")
        #expander.write(figure)

        plt.plot(range(1,31), y, color = 'tab:blue')
        return figure

    figure = sort_by_date(df)
    expander_plot = st.expander("data after sorting by date in plot")
    expander_plot.write(figure)

    st.write("data grouped by hours")

    # Visualise the data by hours using histogram with bins=24,range=(0.5,24)
    @log_time
    def group_by_hours(datframe):
        figure, ax = plt.subplots()
        datframe.groupby("hours").apply(count_rows)
        datframe['hours'].hist(bins=24,range=(0.5,24))
        #ax.hist(df['hours'], bins=24, range=(0.5,24))
        expander_plot = st.expander("data grouped by hours")
        expander_plot.write(figure)
        return datframe
    df = group_by_hours(df)



    st.write("data grouped by weekday")

    # Visualise the data by weekday using histogram with bins=7,range = (-.5,6.5), rwidth=0.8
    @log_time
    def group_by_weekday(dataframe):
        figure, ax = plt.subplots()
        dataframe.groupby('weekday').apply(count_rows)
        dataframe['weekday'].hist(bins=24, range=(0.5, 24))
        #ax.hist(df['hours'], bins=24, range=(0.5,24))
        expander_plot = st.expander("data grouped by weekday")
        expander_plot.write(figure)
        return dataframe
    df = group_by_weekday(df)


    # Check the use of xticks and add 'Mon Tue Wed Thu Fri Sat Sun'.split()
    @log_time
    def group_by_weekday_days_name(dataframe):
        figure, ax = plt.subplots()
        ax.hist(dataframe['weekday'], bins=7, rwidth=0.8, range=(-.5,6.5))
        plt.title("data grouped by weekday")
        plt.xlabel("Weekdays")
        plt.xticks([0,1,2,3,4,5,6], 'Mon Tue Wed Thu Fri Sat Sun'.split())
        return figure

    figure = group_by_weekday_days_name(df)
    expander_plot = st.expander("data grouped by weekday with days's name")
    expander_plot.write(figure)

########
    # Performing Cross Analysis

    st.subheader("Cross Analysis")

    # Group the data by weekday and hour using .apply(count_rows).unstack()
    @log_time
    def group_data_weekday_hour(dataframe):
        figure, ax = plt.subplots()
        weekDay_hour = dataframe.groupby(['weekday','hours']).apply(count_rows).unstack()
        # Create heatmap using seaborn.heatmap for the grouped data
        np.random.seed(0)
        sns.set()
        ax = sns.heatmap(weekDay_hour)
        return figure

    figure = group_data_weekday_hour(df)
    expander_plot = st.expander("heatmap for the grouped data")
    expander_plot.write(figure)

    # Analyse both Latitude and Longitude data represent the specific ranges for each respectively
    @log_time
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def analyse_lon_lat(dataframe):
        figure = plt.figure()
        sns.jointplot(x="Lat",y="Lon",data=dataframe,height=7)
        plt.ylabel('longitude')
        plt.xlabel('latitude')
        return figure

    figure = analyse_lon_lat(df)
    expander_plot = st.expander("longitude & latitute analyse")
    expander_plot.write(figure)

    # Example Latitude range =(40.5,41) and likewise set for Longitude
    # Merge the two histograms Latitude and Longitude using twiny()
    @log_time
    def merge_lon_lat(dataframe):
        figure = plt.figure()
        plt.hist(dataframe['Lat'],range =(40.5,41), color="green", alpha=0.5)
        plt.twiny()
        plt.hist(dataframe['Lon'],range =(-74.3,-73.7), color="red")
        return figure

    figure = merge_lon_lat(df)
    expander_plot = st.expander("merging longitude & latitute")
    expander_plot.write(figure)

    # Plot the dots for both Latitude and Longitude set the figsize=(20, 20)
    @log_time
    #@st.cache(allow_output_mutation=True)
    def plot_dots_lon_lat(dataframe):
        figure = plt.figure(figsize=(15,15), dpi=100)
        plt.suptitle('Scatter plot - Uber - April 2014')
        plt.xlabel('Latitude')
        plt.xlabel('Longitude')
        plt.scatter(dataframe['Lat'].to_list(), dataframe['Lon'].to_list())
        plt.xlim(40.5,41)
        plt.ylim(-74.3,-73.7)
        return figure

    figure = plot_dots_lon_lat(df)
    expander_plot = st.expander("dots for both longitude & latitute")
    expander_plot.write(figure)


if st.sidebar.button("ny trips data"):
    st.subheader("ny trips data ")

    file_path = "ny-trips-data.csv"

    # reading csv into dataframe
    @log_time
    def read_file(file_path):
        return pd.read_csv(file_path, delimiter=',')
    df = read_file(file_path)
    st.write(df.head(5))

######################
    st.subheader("Data transformation")

    # overwriting data after changing format
    @log_time
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def map_to_datetime(dataframe):
        dataframe["tpep_pickup_datetime"] = dataframe["tpep_pickup_datetime"].map(pd.to_datetime)
        return dataframe

    df = map_to_datetime(df)
    expander = st.expander("Dataframe after the to_datetime mapping")
    expander.write(df.head(5))

    ###### Data transformation
    # Create functions for hours / minutes / seconds
    def get_hour(dt):
        return dt.hour

    @log_time
    # @st.cache(allow_output_mutation=True)
    def add_column_hour(dataframe):
        dataframe['hour'] = dataframe['tpep_pickup_datetime'].map(get_hour)
        return dataframe

    df = add_column_hour(df)
    expander = st.expander("adding a column hour")
    expander.write(df['hour'])



    def get_minute(dt):
        return dt.minute

    @log_time
    # @st.cache(allow_output_mutation=True)
    def add_column_minute(dataframe):
        dataframe['minute'] = dataframe['tpep_pickup_datetime'].map(get_minute)
        return dataframe

    df = add_column_minute(df)
    expander = st.expander("adding a column minute")
    expander.write(df['minute'])




    def get_second(dt):
        return dt.second

    @log_time
    # @st.cache(allow_output_mutation=True)
    def add_column_second(dataframe):
        dataframe['second'] = dataframe['tpep_pickup_datetime'].map(get_second)
        return dataframe

    df = add_column_second(df)
    expander = st.expander("adding a column second")
    expander.write(df['second'])

######################
    #####Visual represntation
    st.subheader("Visual representation")

    @log_time
    def plot_frequency_by_hour(dataframe):
        figure, ax = plt.subplots()
        ax.hist(dataframe['hour'], bins=30, rwidth=0.8, range=(0.5,30.5))
        plt.title("Frequency by hour of the day - NY Trips - April 2014")
        plt.xlabel("Hour of the day")
        plt.ylabel("Frequency")
        return figure

    figure = plot_frequency_by_hour(df)
    expander_plot = st.expander("Frequency by hour of the day - NY Trips - April 2014")
    expander_plot.write(figure)


    st.write("data grouped by hour")
    # Creating a function for Grouping the data by hour of the day (hour)
    def count_rows(rows):
        return len(rows)

    @log_time
    def group_by_hour(dataframe):
        dataframe.groupby("hour").apply(count_rows)
        expander = st.expander("data group by hour in dataframe")
        expander.write(dataframe.head(5))
    group_by_hour(df)


    @log_time
    def plot_group_by_hour(dataframe):
        figure = plt.figure()
        y = dataframe.groupby("hour").apply(count_rows)
        plt.bar(range(0, 24), y)
        #expander_plot = st.expander("data grouped by hour in bar")
        #expander_plot.write(figure)
        plt.plot(range(0, 24), y, color='tab:red')
        return figure

    figure = plot_frequency_by_hour(df)
    expander_plot = st.expander("data grouped by hour in plot")
    expander_plot.write(figure)


    st.write("data grouped by minute")
    # Use plot, bar function to plot the data by minute
    @log_time
    def plot_group_by_minute(dataframe):
        figure = plt.figure()
        y = dataframe.groupby("minute").apply(count_rows)
        plt.bar(range(0,60),y, color ='tab:orange', alpha=0.70)
        #expander_plot = st.expander("data grouped by minute in bar")
        #expander_plot.write(figure)
        plt.plot(range(0,60), y, color='tab:red')
        return figure

    figure = plot_group_by_minute(df)
    expander_plot = st.expander("data grouped by minute in plot")
    expander_plot.write(figure)


######################
    st.subheader("Cross Analysis")
    # Sort the data by hour and use bar function to plot the sorted data by hour (.sort_values())

    @log_time
    def data_sort_by_hour(dataframe):
        figure = plt.figure()
        y = dataframe.groupby("hour").apply(count_rows).sort_values()
        plt.bar(range(0, 24), y)
        #expander = st.expander("data sorted by hour...")
        #expander.write(figure)
        plt.plot(range(0, 24), y, color='tab:blue')
        return figure

    figure = data_sort_by_hour(df)
    expander = st.expander("data sorted by hour")
    expander.write(figure)


    # Sort the data by minute and use bar function to plot the sorted data by minute (.sort_values())
    @log_time
    def data_sort_by_minute(dataframe):
        figure = plt.figure()
        y = dataframe.groupby("minute").apply(count_rows).sort_values()
        plt.bar(range(0, 60), y, color='tab:orange', alpha=0.70)
        #expander = st.expander("data sorted by minute...")
        #expander.write(figure)
        plt.plot(range(0, 60), y)
        return figure

    figure = data_sort_by_minute(df)
    expander = st.expander("data sorted by minute")
    expander.write(figure)


finished_log()