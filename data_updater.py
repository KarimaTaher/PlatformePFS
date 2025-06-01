import pandas as pd
from datetime import datetime
from scrapping import get_us_inflation_rate, get_us_petroleum_production, get_us_petroleum_export, get_us_petroleum_import

def build_new_data_row():
    # Get today's date in a format matching your dataset (e.g., YYYY-MM-DD)
    today = datetime.now().strftime("%Y-%m-%d")

    # Call your scraping functions
    inflation = get_us_inflation_rate()
    production = get_us_petroleum_production()
    export = get_us_petroleum_export()
    import_ = get_us_petroleum_import()

    # Prix, GDP, and Event are missing, so fill with None or some default values or scrape from somewhere else
    prix = None
    gdp = None
    event = ""

    # Build a dictionary matching your dataset columns
    new_data = {
        "Date": today,
        "Prix": prix,
        "Import (Thousand Barrels )": import_,
        "Export": export,
        "Production(Thousand Barrels per Day)": production,
        "Inflation (%)": inflation,
        "GDP(Billions of USD)": gdp,
        "Event": event
    }

    return new_data

def append_new_data_to_excel(file_path):
    # Build the new row
    new_row = build_new_data_row()

    try:
        # Load existing data
        df = pd.read_excel(file_path)
        # Append the new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    except FileNotFoundError:
        # If file doesn't exist, create a new DataFrame
        df = pd.DataFrame([new_row])

    # Save back to Excel
    df.to_excel(file_path, index=False)
    print(f"Added new row for {new_row['Date']} to {file_path}")

