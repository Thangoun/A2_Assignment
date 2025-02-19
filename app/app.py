import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pickle
import pandas as pd
import dash_bootstrap_components as dbc
from model import Normal

# Load both models
with open("model/car_price_old.model", "rb") as f:
    old_loaded_model = pickle.load(f)

with open("model/car_price_new.model", "rb") as f:
    new_loaded_model = pickle.load(f)

# Extract model components
old_model = old_loaded_model['model']
old_scaler = old_loaded_model['scaler']

new_model = new_loaded_model['model']
new_scaler = new_loaded_model['scaler']

# Default values
mileage_default = new_loaded_model['mileage']
max_power_default = new_loaded_model['max_power']

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# App layout with styling
app.layout = dbc.Container([
    html.H1("Car Price Prediction", className="text-center text-primary mb-4"),
    
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Label("Select Model:", className="fw-bold"),
                    dcc.Dropdown(
                        id="model-type",
                        options=[
                            {"label": "New Model", "value": "new"},
                            {"label": "Old Model", "value": "old"}
                        ],
                        value="new",
                        className="mb-3"
                    ),
                    html.Label("Year", className="fw-bold"),
                    dcc.Input(id="input-year", type="number", placeholder="Enter year", className="form-control mb-3"),
                    html.Label("Mileage (kmpl)", className="fw-bold"),
                    dcc.Input(id="input-mileage", type="number", placeholder=f"Default: {mileage_default}", className="form-control mb-3"),
                    html.Label("Max Power (bhp)", className="fw-bold"),
                    dcc.Input(id="input-maxpower", type="number", placeholder=f"Default: {max_power_default}", className="form-control mb-3"),
                    dbc.Button("Predict", id="predict-button", color="primary", className="w-100 mt-3 btn-lg"),
                ])
            ], className="shadow-lg p-4"), width=6
        )
    ], justify="center"),
    
    dbc.Row([
        dbc.Col(html.Div(id="output-prediction", className="text-center text-success mt-4 display-4 fw-bold"))
    ])
], fluid=True)


@app.callback(
    Output("output-prediction", "children"),
    Input("predict-button", "n_clicks"),
    State("model-type", "value"),
    State("input-year", "value"),
    State("input-mileage", "value"),
    State("input-maxpower", "value")
)
def predict_price(n_clicks, model_type, year, mileage, max_power):
    if n_clicks and year:
        mileage = mileage if mileage is not None else mileage_default
        max_power = max_power if max_power is not None else max_power_default
        
        # Prepare input as DataFrame with feature names
        input_features = pd.DataFrame([[year, mileage, max_power]], columns=["year", "mileage", "max_power"])
        
        # Select model
        if model_type == "old":
            scaled_features = old_scaler.transform(input_features)
            predicted_price = old_model.predict(scaled_features)[0]
        else:
            scaled_features = new_scaler.transform(input_features)
            intercept = np.ones((scaled_features.shape[0], 1))
            sample_scaled_concat  = np.concatenate((intercept, scaled_features), axis=1)
            predicted_price = new_model.predict(sample_scaled_concat)[0]
        
        predicted_price = np.exp(predicted_price)  # Reverse log transformation
        return f"Predicted Price: {predicted_price:,.0f} Baht"
    
    return ""


if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1', port=7001)