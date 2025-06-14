import os
import pandas as pd
import geopandas as gpd

import json


import skmob
from skmob.utils.plot import plot_gdf

import folium
from folium.utilities import JsCode
from folium.plugins.timeline import Timeline, TimelineSlider
from folium.features import GeoJsonTooltip

import branca.colormap

import matplotlib.pyplot as plt
import seaborn as sns

from attendee_profiling import config


# Reading the Sónar polygons
day_polygons_clipped_plot = gpd.read_file(os.path.join(config.CLIPPED_POLYGONS_PATH, 'sonar_day_polygons_clipped.json'))
night_polygons_clipped_plot = gpd.read_file(os.path.join(config.CLIPPED_POLYGONS_PATH, 'sonar_night_polygons_clipped.json'))




# Function to generate n colors
def generate_colors(n):
    """
    Generates a list of colors from the 'viridis' colormap.

    Parameters:
    n (int): The number of colors to generate.

    Returns:
    list of tuple: A list of RGBA color tuples, where each tuple represents a color 
                   sampled from the 'viridis' colormap.
    """
    cmap = plt.get_cmap('viridis')
    return [cmap(i / n) for i in range(n)]





# Creating a function to visualize the trajectories with their corresponding Sónar polygons
def plot_trajectories_in_context(tdf, background, 
                                 start_end_markers=True, max_users=20, hex_color=None, **kwargs):
    """
    Visualizes trajectories with the Sónar polygons as background.

    Parameters:
        tdf (skmob.TrajDataFrame): The trajectories to plot.
        background (str or folium.Map): 'day', 'night', or an existing folium.Map object.
        start_end_markers (bool): Whether to display start and end markers.
        max_users (int): Maximum number of users to plot.
        hex_color (str): Hexadecimal color for the trajectories.
        **kwargs: Additional arguments for plotting trajectories.

    Returns:
        folium.Map: Map with plotted trajectories.
    """
    global day_polygons_clipped_plot, night_polygons_clipped_plot
    
    # I create a map of polygons for Sónar by day and Sónar by night.
    # These will be used as backround for plotting the trajectories unless an existing map is passed explicitly
    if isinstance(background, str):
        if background == 'day':
            background_map = plot_gdf(day_polygons_clipped_plot,
                                      popup_features=['stage', 'polygon_name', 'source_gis_file'], zoom=17)
        elif background == 'night':
            background_map = plot_gdf(night_polygons_clipped_plot,
                                      popup_features=['stage', 'polygon_name', 'source_gis_file'], zoom=17)
        else:
            raise ValueError("Invalid background value. Use 'day', 'night', or a folium.Map object.")
    elif isinstance(background, folium.Map):
        background_map = background
    else:
        raise ValueError("Invalid background value. Must be 'day', 'night', or a folium.Map object.")

    # Plotting the trajectory on top of an existing map
    map_trajectories = tdf.plot_trajectory(map_f=background_map,
                                           start_end_markers=start_end_markers, max_users=max_users, hex_color=hex_color, **kwargs)
    return map_trajectories




# Function to get the H3_GeoDataframe from the cell labels
def counts_h3_cells_to_gdf(counts_h3_cells_df, sonar_type='night'):
    """
    Converts a DataFrame containing counts for H3 hexagonal cells into a GeoDataFrame 
    by merging it with a GeoJSON file containing the corresponding H3 geometries.

    Parameters:
    ----------
    counts_h3_cells_df : pandas.DataFrame
        A DataFrame containing counts or data associated with H3 cells. 
        Must include a column named 'h3_cell' to match the H3 hexagon IDs.
    
    sonar_type : str, optional
        Specifies the type of SONAR data to use for geometries. 
        Accepted values are 'night' (default) or 'day'. Determines which GeoJSON 
        file is read to retrieve the H3 cell geometries.

    Returns:
    -------
    counts_per_cell : geopandas.GeoDataFrame
        A GeoDataFrame resulting from merging the input DataFrame with H3 geometries. 
        Contains both the data from `counts_h3_cells_df` and the geometry of the cells.
    """
    # Reading the H3 polygons
    if sonar_type == 'night':
        h3_cells_lookup_shp = gpd.read_file(os.path.join(config.CLIPPED_POLYGONS_PATH, 'h3_cells_night_lookup.json'))
    elif sonar_type == 'day':
        h3_cells_lookup_shp = gpd.read_file(os.path.join(config.CLIPPED_POLYGONS_PATH, 'h3_cells_day_lookup.json'))
    else:
        raise Exception("sonar_type must be 'night' or 'day'")
    
    counts_per_cell = h3_cells_lookup_shp.merge(counts_h3_cells_df,
                                                how='inner',
                                                on='h3_cell')
    
    return counts_per_cell




def trajectories_to_h3_counts_gdf(trajectories_df, count_type , window_duration=None, sonar_type='night'):
    """
    Converts a DataFrame of trajectories into a GeoDataFrame with H3 cell-based counts.

    This function groups trajectory data by H3 cells and an optional time window, then computes counts 
    based on the specified count type. The resulting counts are transformed into a GeoDataFrame for spatial analysis.

    Parameters:
    ----------
    trajectories_df : pandas.DataFrame
        A DataFrame containing trajectory data with at least 'h3_cell' and either 'uid' or record entries.
    count_type : str
        The type of count to compute. Options:
        - 'users': Counts unique users ('uid') per H3 cell.
        - 'records': Counts total trajectory records per H3 cell.
    window_duration : {'15min', '5min'}, optional
        The time window duration used to group data. If provided, it must be either '15min' or '5min'. Default is None.
    sonar_type : str, optional
        The sonar type used for polygon generation in `counts_h3_cells_to_gdf()`. Default is 'night'.

    Returns:
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with H3 cell counts and associated geometries for spatial visualization.

    """
    # Assign the columns to group by
    groupby_cols = ['h3_cell']
    if window_duration:
        groupby_cols.append('time_window_' + window_duration)

    # Group trajectories by H3 cells and time window (if available)
    trajectories_grouped = trajectories_df.groupby(by=groupby_cols)

    # Obtain the counts per cell and time_window
    if count_type == 'users':
        time_window_h3_counts = trajectories_grouped['uid'].nunique().reset_index()
        time_window_h3_counts.rename(columns={'uid':'counts_per_cell'}, inplace=True)
    elif count_type == 'records':
        time_window_h3_counts = trajectories_grouped.size().reset_index()
        time_window_h3_counts.rename(columns={0:'counts_per_cell'}, inplace=True)
    

    # Obtaining the corresponding polygons for plotting
    time_window_h3_counts_gdf = counts_h3_cells_to_gdf(counts_h3_cells_df=time_window_h3_counts, sonar_type=sonar_type)

    return time_window_h3_counts_gdf





def create_background_map_sonar(sonar_type):
    """
    Creates a background map using Folium with sonar polygon overlays for day or night configurations.

    Parameters:
        sonar_type (str): Specifies the type of sonar polygons to use. 
                        Acceptable values are 'day' or 'night'.

                        - 'day': Uses the `day_polygons_clipped_plot` global variable 
                            for polygon overlays and centers the map at location 
                            [41.373445, 2.152562].
                        - 'night': Uses the `night_polygons_clipped_plot` global variable 
                            for polygon overlays and centers the map at location 
                            [41.354439, 2.131013].

    Returns:
        folium.Map: A Folium map object with the specified sonar polygons overlaid.
    """
    global day_polygons_clipped_plot, night_polygons_clipped_plot

    if sonar_type == 'night':
        sonar_polygons = night_polygons_clipped_plot
        background_map_sonar = folium.Map(location=[41.354439, 2.131013], zoom_start=17)
    elif sonar_type == 'day':
        sonar_polygons = day_polygons_clipped_plot
        background_map_sonar = folium.Map(location=[41.373445, 2.152562], zoom_start=17)

    # Add polygons from sonar_polygons (optional for context)
    for _, row in sonar_polygons.iterrows():
        folium.GeoJson(
            data=row['geometry'],
            style_function=lambda x: {
                "color": "blue", 
                "fillOpacity": 0
            },
            name=row.get("polygon_name", "Unnamed Polygon"),  # Ensure a fallback name
        ).add_to(background_map_sonar)

    return background_map_sonar





# Function to plot the counts per cell (static in time)
def plot_counts_per_cell(sonar_type, h3_cells_counts, caption, filename_to_save=None, return_map=True):
    """
    Generates an interactive Folium map to visualize counts for H3 cells.

    Parameters:
    -----------
    sonar_type : str
        The type of dataset to visualize. Must be either 'day' or 'night'.
    h3_cells_counts : GeoDataFrame
        A GeoDataFrame containing H3 cells and their associated counts. 
        It must have the columns 'h3_cell' and 'counts_per_cell'.
    caption : str
        The caption to display for the colormap legend, describing the values of observation counts.
    filename_to_save : str, optional
        If provided, the map will be saved to an HTML file with this filename. Default is None.

    Returns:
    --------
    folium.Map
        A Folium Map object with the polygons and observation counts visualization.
    
    """
    # Creating the background map of the corresponding Sónar
    map_counts_per_cell = create_background_map_sonar(sonar_type)

    # Create a colormap for the Choropleth based on observation counts
    colormap = branca.colormap.LinearColormap(
        vmin=h3_cells_counts["counts_per_cell"].min(),
        vmax=h3_cells_counts["counts_per_cell"].max(),
        colors=["lightyellow", "orange", "red"],
        caption=caption,
    )

    # Create tooltip for H3 cells
    tooltip = folium.GeoJsonTooltip(
        fields=["h3_cell", "counts_per_cell"],  # Columns from GeoDataFrame
        aliases=["H3 Cell:", "Count:"],  # Display labels
        localize=True,
        sticky=True,
        labels=True,
        style="""background-color: #F0EFEF; 
                 border: 1px solid black; 
                 border-radius: 3px;
                 padding: 5px;""",
    )

    # Add GeoJson layer for H3 cells with the colormap
    folium.GeoJson(
        data=h3_cells_counts,
        style_function=lambda x: {
            "fillColor": colormap(x["properties"]["counts_per_cell"])
            if x["properties"]["counts_per_cell"] is not None
            else "transparent",
            "color": "black",
            "weight": 0.1,
            "fillOpacity": 0.5,
        },
        tooltip=tooltip,
    ).add_to(map_counts_per_cell)

    # Add colormap to the map
    colormap.add_to(map_counts_per_cell)

    # Save if necessary
    if filename_to_save:
        map_counts_per_cell.save(filename_to_save)
        print(f'File saved in {filename_to_save}.')

    # Return the map object
    if return_map:
        return map_counts_per_cell
    




# Function to plot the counts per cell (evolving in time)
def plot_counts_per_cell_timeline(sonar_type, h3_cells_counts, caption, window_duration='5min', filename_to_save=None, return_map=True):
    """
    Plots a timeline of observation counts per H3 cell on a Folium map with an interactive slider.

    Parameters:
        sonar_type (str): Specifies the type of sonar map to use ('day' or 'night'). 
                        This determines the base map created using `create_background_map_sonar`.

        h3_cells_counts (GeoDataFrame): A GeoPandas DataFrame containing H3 cell data, with the following required columns:
            - 'geometry': The polygon geometry for each H3 cell.
            - 'counts_per_cell': The observation count for each cell.
            - 'time_window': A datetime object representing the time window of observations.

        caption (str): Caption for the color scale legend, describing the observation counts.

        window_duration: Determines the granularity that is plotted and controls the duration playback. 
                        Can be '5min' or '15min'. Default is '5min'.

        filename_to_save (str, optional): Path to save the map as an HTML file. If not provided, the map is not saved.

        return_map (bool, optional): If True, returns the generated Folium map object. Defaults to True.

    Returns:
        folium.Map: A Folium map object with:
            - H3 cell overlays colored based on observation counts.
            - A time slider allowing visualization of data across the time windows.
            - A tooltip showing details about each H3 cell and its observation count.
            - A color legend indicating the range of observation counts.

    Notes:
        - The time intervals for the slider are derived from the `time_window` column, with a duration of 15 minutes per interval.
    """
    # Creating the background map of the corresponding Sónar
    map_counts_per_cell = create_background_map_sonar(sonar_type)

    # Create colormap for observation counts
    colormap = branca.colormap.LinearColormap(
        vmin=h3_cells_counts["counts_per_cell"].min(),
        vmax=h3_cells_counts["counts_per_cell"].max(),
        colors=["lightyellow", "orange", "red"],
        caption=caption,
    )

    # Add colormap for reference
    colormap.add_to(map_counts_per_cell)

    # Prepare GeoJSON with start and end times for each feature
    h3_cells_counts = h3_cells_counts.copy() # Copy to avoid modifying the original df
    
    # The start and end are added as epoch in milliseconds for JS
    h3_cells_counts["start"] = h3_cells_counts[f"time_window_{window_duration}"].apply(lambda x: x.timestamp()) * 1000
    h3_cells_counts["end"] = (h3_cells_counts[f"time_window_{window_duration}"] + pd.Timedelta(15,'min')).apply(lambda x: x.timestamp()) * 1000 
 
    h3_cells_geojson = json.loads(h3_cells_counts.drop(columns=[f"time_window_{window_duration}"]).to_json())

    # Add Timeline to map
    timeline = Timeline(
        h3_cells_geojson,
        get_interval=JsCode(
            """
            function(feature) {
                return {
                    start: feature.properties.start,
                    end: feature.properties.end
                };
            }
            """
        ),
        style=JsCode(
            """
            function(feature) {
                const color = feature.properties.counts_per_cell > 0 ? 
                    `rgba(255, 0, 0, ${Math.min(feature.properties.counts_per_cell / 10, 0.8)})` : 
                    "transparent";
                return {
                    color: "black",
                    weight: 0.1,
                    fillColor: color,
                    fillOpacity: 0.6,
                };
            }
            """
        ),
    ).add_to(map_counts_per_cell)

    # Add TimelineSlider to the map
    if window_duration=='15min':
        playback_duration = 9000
    elif window_duration=='5min':
        playback_duration = 60000
    
    TimelineSlider(
        auto_play=True,
        show_ticks=True,
        enable_keyboard_controls=True,
        playback_duration=playback_duration,
    ).add_timelines(timeline).add_to(map_counts_per_cell)

    # Add GeoJsonTooltip for tooltip information
    GeoJsonTooltip(
        fields=["h3_cell", "counts_per_cell"],
        aliases=["H3 Cell:", "Count:"],
        localize=True,
        sticky=True,
        labels=True,
        style="""background-color: #F0EFEF; border: 1px solid black; border-radius: 3px; padding: 5px;""",
    ).add_to(timeline)

    # Save to file if requested
    if filename_to_save:
        map_counts_per_cell.save(filename_to_save)
        print(f'File saved in {filename_to_save}.')

    # Return the map object
    if return_map:
        return map_counts_per_cell
    




def compute_event_final_scores(user_event_scores_durations_df):
    """
    Computes final scores for events by aggregating audience time and ranking points.

    This function groups event data by event title, stage, event duration timetable minutes, 
    activity type, and music type, then calculates total audience time minutes and ranking points 
    based on the sum of adjusted scores. The results are sorted in descending order of ranking points.

    Parameters:
    ----------
    user_event_scores_durations_df : pandas.DataFrame
        A DataFrame containing event-related data with columns: event_title, stage, 
        event_duration_timetable_mins, activity_type, music_type, total_audience_time_at_event_mins, and adj_score.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with columns: event_title, stage, event_duration_timetable_mins, activity_type, 
        music_type, total_audience_time_mins, and ranking_points.

    """
    event_final_scores = (user_event_scores_durations_df
                        .groupby(['event_title','stage'], as_index=False)
                        .agg(total_audience_time_mins=('total_audience_time_at_event_mins','sum'),
                            ranking_points=('adj_score','sum'))
                        .sort_values(by='ranking_points', ascending=False))
    return event_final_scores





def plot_rankings(scores_df, show_plot=True, filename_to_save=None):
    """
    Plots event rankings based on total audience time and adjusted scores.

    This function visualizes event rankings by sorting events into two lists: 
    one ranked by total audience time and the other by ranking points.

    Parameters:
    ----------
    scores_df : pandas.DataFrame
        A DataFrame containing event-related data with columns: event_title, stage, 
        total_audience_time_mins, and ranking_points.
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    filename_to_save : str, optional
        If provided, saves the plot to the specified filename. Default is None.

    Returns:
    -------
    None
        Displays the ranking plot if `show_plot` is True and/or saves it as an image 
        if `filename_to_save` is specified.
    """
    # Generate colors for unique stages
    unique_stages = scores_df['stage'].unique()
    stage_colors = {stage: color for stage, color in zip(unique_stages, generate_colors(len(unique_stages)))}

    # Sort the rankings
    left_rank = scores_df.sort_values(by="total_audience_time_mins", ascending=False).reset_index(drop=True)
    right_rank = scores_df.sort_values(by="ranking_points", ascending=False).reset_index(drop=True)

    # Assign rankings
    left_rank["left_rank"] = range(1, len(left_rank) + 1)
    right_rank["right_rank"] = range(1, len(right_rank) + 1)

    # Merge rankings by event_title
    merged = pd.merge(left_rank, right_rank, on="event_title")

    # Plot
    fig, ax = plt.subplots(figsize=(2.5, 13))

    # Plot left rankings
    for i, row in merged.iterrows():
        ax.text(
            x=0,
            y=row["left_rank"],
            s=row["event_title"],
            ha="right",
            va="center",
            fontsize=13,
            color=stage_colors[row["stage_x"]]
        )

    # Plot right rankings
    for i, row in merged.iterrows():
        ax.text(
            x=1,
            y=row["right_rank"],
            s=row["event_title"],
            ha="left",
            va="center",
            fontsize=13,
            color=stage_colors[row["stage_x"]]
        )

    # Connect lines
    for i, row in merged.iterrows():
        ax.plot(
            [0, 1],
            [row["left_rank"], row["right_rank"]],
            color="gray",
            alpha=0.6,
            linewidth=1
        )

    # Adjust plot aesthetics
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0.5, len(merged) + 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Add titles
    ax.text(-0.35, -1, "Total Audience Time Ranking", ha="right", fontsize=16, fontweight="bold")  
    ax.text(1.35, -1, "Adjusted Scores Ranking", ha="left", fontsize=16, fontweight="bold")

    # Add legend for stages
    for stage, color in stage_colors.items():
        ax.plot([], [], color=color, label=stage, linestyle='none', marker='o', markersize=8)
    ax.legend(title="Stages", loc="center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)

    plt.tight_layout()
    
    if filename_to_save:
        plt.savefig(filename_to_save, bbox_inches='tight')
        print(f'Ranking plot saved in {filename_to_save}')

    if show_plot:
        plt.show()
    else:
        plt.close()



def proportion_people_stage(trajectories_events_df):
    """
    Calculates the proportion of people present at each stage during specific time windows for different days or nights of an event.

    Parameters:
    -----------
    trajectories_events_df : pandas.DataFrame
        A DataFrame containing the following columns:
            - 'tid': Unique identifiers for days or nights.
            - 'stage_renamed': Stage labels.
            - 'time_window_15min': Time windows in 15-minute intervals.
            - 'uid': Unique identifiers for individuals.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the following columns:
            - 'tid': Day or night identifier.
            - 'stage_renamed': Stage names or labels.
            - 'time_window_15min': Time windows in 15-minute intervals.
            - 'num_people': The number of unique individuals at each stage and time window.
            - 'total_people_window': The total number of individuals across all stages for each time window.
            - 'proportion_people_window': The proportion of people at each stage relative to the total for the corresponding time window.
    """
    people_counts_window_stage = (trajectories_events_df.groupby(['tid', 'stage_renamed', 'time_window_15min'], observed=True)
                                  .agg(num_people=('uid', 'nunique'))
                                  .reset_index())
    
    # Group by `tid` and `time_window` to compute the total number of people at each time window
    total_people_window_night = (people_counts_window_stage.groupby(['tid', 'time_window_15min'], observed=True)
                                .agg(total_people_window=('num_people', 'sum'))
                                .reset_index()
                                )

    # Merge the total counts back into the grouped data
    people_counts_window_stage = pd.merge(people_counts_window_stage, total_people_window_night, on=['tid', 'time_window_15min'])

    # Compute the proportion of people at each stage
    people_counts_window_stage['proportion_people_window'] = people_counts_window_stage['num_people'] / people_counts_window_stage['total_people_window']

    return people_counts_window_stage




def plot_stage_proportion_heatmaps(data, sonar_type='night', filename_to_save=None):
    """
    Plots heatmaps showing the proportion of people present at different stages over time for each day or night of the event.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset containing the following columns:
            - 'tid': Unique identifiers for days or nights.
            - 'time_window_15min': Time windows in 15-minute intervals.
            - 'stage_renamed': Stage labels.
            - 'proportion_people_window': Proportion of people present at each stage per time window.

    sonar_type : str, optional, default='night'
        A label indicating the type of event (e.g., 'night'). This is included in the plot titles.

    filename_to_save : str, optional, default=None
        If provided, the plots can be saved to this file. Currently unused in the function.

    Returns:
    --------
    None
        Displays heatmaps for each unique day or night (tid) in the dataset, showing the distribution of people across stages over time.
    """
    # Get unique days/nights (tids)
    unique_tids = data['tid'].unique()

    # Limiting the plot axis to force the alignment between the plots for each day
    min_time_windows = data.groupby('tid')['time_window_15min'].nunique().min()
    
    for tid in unique_tids:
        # Filter data for the current tid
        tid_data = data[data['tid'] == tid]

        # Truncate to the minimum range of time windows
        truncated_time_windows = tid_data['time_window_15min'].unique()[:min_time_windows]
        tid_data = tid_data[tid_data['time_window_15min'].isin(truncated_time_windows)]
        
        # Pivot the data to get a matrix for heatmap
        heatmap_data = tid_data.pivot(index='stage_renamed',
                                      columns='time_window_15min', 
                                      values='proportion_people_window'
                                      )
        
        # Plot the heatmap (with a temporary change of theme to get rid of the grid)
        plt.figure(figsize=(11, 4))
        with sns.axes_style("dark"), sns.plotting_context("paper", rc={"axes.labelsize": 12,
                                                                       "axes.titlesize": 13,
                                                                       "xtick.labelsize": 11,
                                                                       "ytick.labelsize": 11,
                                                                       "legend.fontsize": 11}):
            ax = sns.heatmap(heatmap_data,
                            cmap='plasma',
                            cbar_kws={'label': "Proportion of people at each stage"},
                            annot=False
                            )
        
        # Format x-axis ticks with vertical labels
        time_labels = heatmap_data.columns.strftime('%H:%M')
        
        ax.set_xticks(ticks=[pos + 0.2 for pos in range(len(heatmap_data.columns))],
                      labels=time_labels,
                      rotation=90,
                      horizontalalignment='left'
                      )
        
        # Title and axis labels
        plt.title(f"Proportion of people with respect to each time window's total. {sonar_type.capitalize()} {tid}")
        plt.xlabel("Time window")
        plt.ylabel(None)
        
        plt.tight_layout()
        plt.show()




