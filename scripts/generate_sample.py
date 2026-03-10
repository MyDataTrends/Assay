import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(num_rows=1000, output_path="datasets/sample_sales_data.csv"):
    np.random.seed(42)
    
    # Generate dates from 2021 to 2023
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_rows)]
    
    # Categories
    regions = ["North America", "Europe", "Asia", "South America"]
    products = ["SaaS Subscription", "Enterprise License", "Support Add-on", "Hardware Appliance"]
    
    # Base pattern (synthetic seasonality)
    # Sine wave for summer spike, winter drop
    time_index = np.arange(num_rows)
    seasonality = np.sin(time_index * (2 * np.pi / 365)) * 1000
    base_sales = 5000 + (time_index * 2) + seasonality + np.random.normal(0, 500, num_rows)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "Region": np.random.choice(regions, num_rows, p=[0.4, 0.3, 0.2, 0.1]),
        "Product": np.random.choice(products, num_rows, p=[0.5, 0.2, 0.2, 0.1]),
        "Customer_Age": np.random.randint(18, 70, num_rows),
        "Sales": base_sales.round(2)
    })
    
    # Add units and profit with some correlation
    df["Units_Sold"] = (df["Sales"] / np.random.uniform(50, 150, num_rows)).astype(int)
    df["Profit"] = (df["Sales"] * np.random.uniform(0.1, 0.4, num_rows)).round(2)
    
    # Inject an anomaly: A huge spike on Black Friday 2022
    black_friday_2022 = datetime(2022, 11, 25)
    bf_idx = df[df["Date"] == black_friday_2022].index
    if not bf_idx.empty:
        df.loc[bf_idx, "Sales"] *= 4.5
        df.loc[bf_idx, "Profit"] *= 5.0
        df.loc[bf_idx, "Units_Sold"] *= 4
    
    # Inject missing values for realism
    missing_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[missing_indices, "Profit"] = np.nan
    
    missing_age = np.random.choice(df.index, size=8, replace=False)
    df.loc[missing_age, "Customer_Age"] = np.nan
    
    import os
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
        
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows of sample data to {output_path}")
    print("Features: Seasonality, 1 Anomaly (Black Friday 2022), Missing values in Profit/Age.")

if __name__ == "__main__":
    generate_sample_data()
