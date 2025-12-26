# src/tools/visualizer.py
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List

class VisualizerTool:
    def create_dynamic_chart(self, chart_type: str, labels: List[str], values: List[float], title: str):
        """
        Dynamically creates charts with safety checks for None values.
        """
        print(f"📊 [Tool: Visualizer] Generating {chart_type} chart: {title}")
        
        # FIX: Ensure no None values exist in the list before processing
        cleaned_values = [v if v is not None else 0.0 for v in values]
        
        # Build DataFrame with cleaned data
        data = pd.DataFrame({"Category": labels, "Value": cleaned_values})
        
        # Professional Chart selection logic
        if chart_type.lower() == "line":
            fig = px.line(data, x="Category", y="Value", title=title, markers=True)
        elif chart_type.lower() == "area":
            fig = px.area(data, x="Category", y="Value", title=title)
        elif chart_type.lower() == "pie":
            fig = px.pie(data, names="Category", values="Value", title=title, hole=0.4)
        else: # Default Bar
            fig = px.bar(data, x="Category", y="Value", title=title, 
                         text_auto='.2s', color="Value", 
                         color_continuous_scale=px.colors.sequential.Blues)

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        
        st.plotly_chart(fig, use_container_width=True)
        return f"SUCCESS: {chart_type.capitalize()} chart rendered."