import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import dash
from dash import dcc, html
import plotly.graph_objs as go

# Load data
data = pd.read_csv(r"C:\Users\arman\OneDrive\Desktop\Coding\data.csv")

# Preprocess data
data = pd.get_dummies(data, columns=["street", "city", "statezip", "country"])
data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data = data.drop(columns=["date"])

# Split features and target variable
X = data.drop(columns=["price"])
Y = data["price"]


def train_and_evaluate(model, X_train, X_test, Y_train, Y_test, algorithm_name):
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(Y_test, predictions)
    return {
        "Algorithm": algorithm_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Predictions": predictions,
        "Actuals": Y_test,
    }


# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# Train models
linear_results = train_and_evaluate(
    LinearRegression(), X_train, X_test, Y_train, Y_test, "Linear Regression"
)
gbr_results = train_and_evaluate(
    GradientBoostingRegressor(random_state=42),
    X_train,
    X_test,
    Y_train,
    Y_test,
    "Gradient Boosting Regression",
)
rf_results = train_and_evaluate(
    RandomForestRegressor(random_state=42),
    X_train,
    X_test,
    Y_train,
    Y_test,
    "Random Forest Regression",
)
xgb_results = train_and_evaluate(
    xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
    X_train,
    X_test,
    Y_train,
    Y_test,
    "XGBoost Regression",
)

# Create Dash app
app = dash.Dash(__name__)


def create_scatter_plot(results, title):
    return dcc.Graph(
        figure={
            "data": [
                go.Scatter(
                    x=results["Actuals"],
                    y=results["Predictions"],
                    mode="markers",
                    name="Predicted vs Actual",
                    marker=dict(color="blue"),
                ),
                go.Scatter(
                    x=results["Actuals"],
                    y=results["Actuals"],
                    mode="lines",
                    name="Ideal Fit",
                    line=dict(color="red"),
                ),
            ],
            "layout": go.Layout(
                title=title,
                xaxis={"title": "Actual Prices", "range": [0, 4000000]},
                yaxis={"title": "Predicted Prices"},
                showlegend=True,
            ),
        }
    )


app.layout = html.Div(
    children=[
        html.H1(children="House Price Prediction Model Comparison"),
        create_scatter_plot(
            linear_results, "Linear Regression: Predicted vs Actual Prices"
        ),
        create_scatter_plot(
            gbr_results, "Gradient Boosting Regression: Predicted vs Actual Prices"
        ),
        create_scatter_plot(
            rf_results, "Random Forest Regression: Predicted vs Actual Prices"
        ),
        create_scatter_plot(
            xgb_results, "XGBoost Regression: Predicted vs Actual Prices"
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
