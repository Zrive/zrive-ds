import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


# API configuration
START_DATE = "2010-01-01"
END_DATE = "2020-12-31"


def get_data_meteo_api(city, start_date=START_DATE, end_date=END_DATE):
    """
    Fetches weather data for a specific city from the Meteo API.

    Args:
        city: City name (must be in COORDINATES)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with weather data
    """
    # Build API parameters
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
        "timezone": "GMT",
    }

    # Make API request
    print(f"Fetching data for {city}...")
    response = requests.get(API_URL, params=params)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data["daily"])

    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Add city name as a column
    df["city"] = city

    print(f"Data retrieved for {city}: {len(df)} records")
    return df


def process_weather_data(data, resample_freq="ME"):
    """
    Processes weather data to reduce temporal resolution.

    Args:
        data: DataFrame with weather data
        resample_freq: Resample frequency ('ME' for monthly, 'Q' for quarterly,
        'Y' for yearly)

    Returns:
        DataFrame with processed data
    """
    # Set the date as index for resampling
    df = data.copy()
    df.set_index("time", inplace=True)

    # Resample by the specified frequency
    resampled_dfs = []

    for city in df["city"].unique():
        city_data = df[df["city"] == city].copy()

        # Apply different aggregation functions based on the variable
        agg_functions = {
            "temperature_2m_mean": "mean",  # Average of temperatures
            "precipitation_sum": "sum",  # Sum of precipitation
            "wind_speed_10m_max": "max",  # Maximum of maximum wind speeds
            "city": "first",  # Keep the city name
        }

        # Perform resampling with appropriate functions
        city_resampled = city_data.resample(resample_freq).agg(agg_functions)
        resampled_dfs.append(city_resampled)

    # Combine all processed DataFrames
    result = pd.concat(resampled_dfs)

    # Reset index to have the date as a column
    result.reset_index(inplace=True)

    return result


def plot_weather_data(data, output_dir="./plots"):
    """
    Generates plots from processed weather data.

    Args:
        data: DataFrame with processed data
        output_dir: Directory to save the plots

    Returns:
        Dictionary with paths to generated files
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_files = {}

    # Configure plot style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("Set2")

    # Combined plot of all variables
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    # For each city, plot in a different color
    colors = {"Madrid": "firebrick", "London": "royalblue", "Rio": "forestgreen"}

    for city in data["city"].unique():
        city_data = data[data["city"] == city]

        # Temperature
        ax1.plot(
            city_data["time"],
            city_data["temperature_2m_mean"],
            color=colors[city],
            marker="o",
            linestyle="-",
            label=city,
        )
        ax1.set_title("Mean Temperature (2010-2020)", fontsize=14)
        ax1.set_ylabel("Temperature (°C)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Precipitation
        ax2.plot(
            city_data["time"],
            city_data["precipitation_sum"],
            color=colors[city],
            marker="s",
            linestyle="-",
            label=city,
        )
        ax2.set_title("Precipitation (2010-2020)", fontsize=14)
        ax2.set_ylabel("Precipitation (mm)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Wind
        ax3.plot(
            city_data["time"],
            city_data["wind_speed_10m_max"],
            color=colors[city],
            marker="^",
            linestyle="-",
            label=city,
        )
        ax3.set_title("Maximum Wind Speed (2010-2020)", fontsize=14)
        ax3.set_xlabel("Date", fontsize=12)
        ax3.set_ylabel("Wind (km/h)", fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    plt.tight_layout()

    combined_file = f"{output_dir}/weather_variables_comparison_2010_2020.png"
    plt.savefig(combined_file, dpi=300)
    plot_files["comparison"] = combined_file
    plt.close()

    return plot_files


def main():
    """
    Main function that executes the entire workflow.
    """
    # Get data for each city
    all_data = []
    for city in COORDINATES.keys():
        city_data = get_data_meteo_api(city)
        all_data.append(city_data)

    # Combine all data
    combined_data = pd.concat(all_data)
    print(f"Total combined records: {len(combined_data)}")

    # Process data (reduce temporal resolution to monthly)
    processed_data = process_weather_data(combined_data, resample_freq="ME")
    print(f"Processed data: {len(processed_data)} records after resampling")

    # Generate plots
    print("Generating plot...")
    plot_files = plot_weather_data(processed_data)

    return processed_data, plot_files


if __name__ == "__main__":
    main()
