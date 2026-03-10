import pandas as pd
from visualization.smart_charts import recommend_charts

try:
    df = pd.read_csv("datasets/sample_sales_data.csv")
    recs = recommend_charts(df)
    print(f"Recommendations count: {len(recs)}")
    for i, rec in enumerate(recs):
        print(f"{i}: {rec.chart_type} | {rec.title} | {rec.reason}")
except Exception as e:
    print(f"Error: {e}")
