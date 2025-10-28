import os
import sys
from pathlib import Path
import requests

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'app-ml', 'src'))
os.chdir(project_root)

# Force working directory to the project root
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from common.data_manager import DataManager
from common.utils import read_config, make_prediction_figures

# Load configuration using utils function
config_path = project_root / 'config' / 'config.yaml'
config = read_config(config_path)

# Override host for Docker environment if environment variable is set
inference_api_host = os.environ.get('INFERENCE_API_HOST', config.get('inference_api', {}).get('host', 'localhost'))
inference_api_port = config.get('inference_api', {}).get('port', 5001)
inference_api_endpoint = config.get('inference_api', {}).get('endpoint', '/run-inference')
INFERENCE_API_URL = f"http://{inference_api_host}:{inference_api_port}{inference_api_endpoint}"

# Initialize data manager and production database
data_manager = DataManager(config)
data_manager.initialize_prod_database()

# Use the default Bootstrap (light) theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create the layout
app.layout = dbc.Container([
    dcc.Store(id='shared-xaxis-range'), # Store of data for zooming to the x-axis
    dcc.Store(id='inference-trigger', data=0),  # Store for inference trigger
    dbc.Row([
        # Control Panel
        dbc.Col([
            html.H4("Control Panel", style={"color": "#222"}),
            html.Div([
                html.Label("Plots display time (last N hours)", style={"color": "#222"}),
                dcc.Input(
                    id='lookback-hours',
                    type='number',
                    min=1,
                    step=1,
                    value=config['ui']['default_lookback_hours'],
                    style={"marginBottom": "16px", "width": "100%"}
                ),
                html.Label("Select features to display", style={"marginTop": "10px", "color": "#222"}),
                dcc.Dropdown(
                    id='parameter-dropdown',
                    options=[
                        {'label': 'Temperature', 'value': 'temp'},
                        {'label': 'Humidity', 'value': 'hum'},
                        {'label': 'Wind Speed', 'value': 'windspeed'},
                        {'label': 'Week Day', 'value': 'weekday'},
                        {'label': 'Working Day', 'value': 'workingday'},
                        {'label': 'Weather', 'value': 'weathersit'},
                    ],
                    value=['temp', 'hum'],
                    multi=True,
                    style={"backgroundColor": "#fff", "color": "#222"}
                ),
                html.Div([
                    dbc.Button(
                        "Predict Next Step", 
                        id="run-inference-btn", 
                        color="primary", 
                        n_clicks=0, 
                        style={"marginTop": "16px", "width": "100%", "marginBottom": "8px"}
                    ),
                    html.Div(id="inference-status", style={"fontSize": "12px", "color": "#666"})
                ]),
            ], style={"backgroundColor": "#fff", "borderRadius": "12px", "padding": "16px", "border": "1px solid #e0e0e0"}),
            html.Div([
                html.H5("ML Application Overview", style={"marginTop": "24px", "color": "#222"}),
                html.Ul([
                    html.Li("The ML application running an end-to-end ML pipeline (preprocessing, feature engineering, inference, postprocessing) in real-time"),
                    html.Li("The top plot shows predicted bike count for the next hour vs true values."),
                    html.Li("The bottom plot displays selected features associated with the predicted bike count."),
                    html.Li("Plots update when 'Predict Next Step' is clicked."),
                    html.Li("The UI app and inference pipeline run in 2 Docker containers."),
                    html.Li("The data and model are stored and shared in Docker volumes."),
                ], className="overview-list", style={"color": "#444", "fontSize": "15px", "padding": "10px", "lineHeight": "1.4"})
            ], style={"backgroundColor": "#fff", "borderRadius": "12px", "padding": "20px", "border": "1px solid #e0e0e0",
                      "height": "100%", "display": "flex", "flexDirection": "column", "marginBottom": "10px", "marginTop": "10px"
                          }
                          )
        ], width=3, style={"display": "flex", "flexDirection": "column", "height": "100%", "paddingTop": "10px"}),
        # Graphs
        dbc.Col([
            html.H5("Real-time Bike Count Predictions", style={"marginBottom": "1px", "color": "#222"}),
            dcc.Graph(id='graph-1', clear_on_unhover=True, style={
                "backgroundColor": "#fff",
                "borderRadius": "12px",
                "padding": "8px",
                "height": "50%", 
                "width": "100%",
                "minHeight": 0   
            }),
            html.H5("Features Data", style={"marginBottom": "1px", "color": "#222"}),
            dcc.Graph(id='graph-2', clear_on_unhover=True, style={
                "backgroundColor": "#fff",
                "borderRadius": "12px",
                "padding": "8px",
                "height": "50%", 
                "width": "100%",
                "minHeight": 0
            }),
        ], width=9, style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "padding": "0",
            "gap": "10px",
            "paddingBottom": "10px",
            "paddingTop": "10px",
            "paddingRight": "10px",
        })
            ], align="stretch", style={"flex": 1, "height": "100%"})
        ], fluid=True, style={"height": "100vh", "minHeight": "100vh", "backgroundColor": "#e9e9f0"})


# Callback to update the shared x-axis range when either plot is zoomed or panned
@callback(
    Output('shared-xaxis-range', 'data'),
    [Input('graph-1', 'relayoutData'),
     Input('graph-2', 'relayoutData')],
    [State('shared-xaxis-range', 'data')]
)
def sync_xaxis_range(relayout1, relayout2, stored_range):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    relayout = relayout1 if trigger == 'graph-1' else relayout2 if trigger == 'graph-2' else None
    if relayout and ('xaxis.range[0]' in relayout and 'xaxis.range[1]' in relayout):
        return [relayout['xaxis.range[0]'], relayout['xaxis.range[1]']]
    elif relayout and 'xaxis.autorange' in relayout:
        return None  # Reset to autorange
    return stored_range


# Main callback to update both plots, now using the shared x-axis range
@callback(
    [Output('graph-1', 'figure'),
     Output('graph-2', 'figure')],
    [
     Input('lookback-hours', 'value'),
     Input('parameter-dropdown', 'value'),
     Input('shared-xaxis-range', 'data'),
     Input('inference-trigger', 'data')]  # Use inference trigger instead of button clicks
)
def update_graphs(lookback_hours, parameters, shared_xrange, inference_trigger):
    try:
        if lookback_hours is None or lookback_hours < 1:
            lookback_hours = config['ui']['default_lookback_hours']
        try:
            df_pred = data_manager.load_prediction_data()
        except Exception:
            df_pred = None
        df_prod = data_manager.load_prod_data()
        fig1, fig2 = make_prediction_figures(
            df_prod, df_pred, parameters, config, lookback_hours, shared_xrange
        )
        return fig1, fig2
    except Exception as e:
        fig1 = go.Figure()
        fig2 = go.Figure()
        for fig in [fig1, fig2]:
            fig.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=40, t=20, b=20),
                height=330,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Value",
                annotations=[{
                    'text': f'Error loading data: {str(e)}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 20}
                }],
                plot_bgcolor='#fff',
                paper_bgcolor='#fff'
            )
        return fig1, fig2


# Callback for the inference button
@callback(
    Output('inference-status', 'children'),
    Output('run-inference-btn', 'disabled'),
    Output('inference-trigger', 'data'),  # Add a trigger for plot updates
    Input('run-inference-btn', 'n_clicks'),
    prevent_initial_call=True
)
def trigger_inference(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return "", False, 0
    try:
        # Disable button during inference
        # Call the inference API in the other container
        response = requests.post(INFERENCE_API_URL)
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return f"✅ Prediction completed for {result.get('timestamp', '')}", False, n_clicks
            else:
                return f"Error: {result.get('message', 'Unknown error')}", False, 0
        else:
            return f"Error: {response.text}", False, 0
    except Exception as e:
        return f"Error: {str(e)}", False, 0


server = app.server

if __name__ == '__main__':
    # Use debug=True for development, debug=False for production
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8050)

# import os
# import sys
# from pathlib import Path
# import requests
# import streamlit as st
# import plotly.graph_objects as go
# import pandas as pd
# from datetime import datetime, timedelta

# # Setup project paths
# project_root = Path(__file__).resolve().parents[1]
# sys.path.append(str(project_root))
# sys.path.append(os.path.join(project_root, 'src'))
# sys.path.append(os.path.join(project_root, 'app-ml', 'src'))
# os.chdir(project_root)

# from common.data_manager import DataManager
# from common.utils import read_config, make_prediction_figures

# # Page configuration
# st.set_page_config(
#     page_title="Real-time Bike Count Predictions",
#     page_icon="🚴‍♂️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #222;
#         margin-bottom: 1rem;
#         text-align: center;
#     }
#     .section-header {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #222;
#         margin-top: 1rem;
#         margin-bottom: 0.5rem;
#     }
#     .status-success {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .status-error {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     .overview-box {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border: 1px solid #e9ecef;
#         margin: 1rem 0;
#     }
#     .metric-container {
#         background-color: white;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border: 1px solid #e9ecef;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_resource
# def initialize_app():
#     """Initialize application components"""
#     # Load configuration
#     config_path = project_root / 'config' / 'config.yaml'
#     config = read_config(config_path)
    
#     # Setup API URL
#     inference_api_host = os.environ.get('INFERENCE_API_HOST', config.get('inference_api', {}).get('host', 'localhost'))
#     inference_api_port = config.get('inference_api', {}).get('port', 5001)
#     inference_api_endpoint = config.get('inference_api', {}).get('endpoint', '/run-inference')
#     inference_api_url = f"http://{inference_api_host}:{inference_api_port}{inference_api_endpoint}"
    
#     # Initialize data manager
#     data_manager = DataManager(config)
#     data_manager.initialize_prod_database()
    
#     return config, data_manager, inference_api_url

# def load_data(data_manager):
#     """Load prediction and production data"""
#     try:
#         try:
#             df_pred = data_manager.load_prediction_data()
#         except Exception:
#             df_pred = None
#         df_prod = data_manager.load_prod_data()
#         return df_pred, df_prod, None
#     except Exception as e:
#         return None, None, str(e)

# def create_error_figure(error_message):
#     """Create error figure when data loading fails"""
#     fig = go.Figure()
#     fig.update_layout(
#         template='plotly_white',
#         margin=dict(l=40, r=40, t=20, b=20),
#         height=330,
#         showlegend=True,
#         xaxis_title="Time",
#         yaxis_title="Value",
#         annotations=[{
#             'text': f'Error loading data: {error_message}',
#             'xref': 'paper',
#             'yref': 'paper',
#             'x': 0.5,
#             'y': 0.5,
#             'showarrow': False,
#             'font': {'size': 16, 'color': 'red'}
#         }],
#         plot_bgcolor='#fff',
#         paper_bgcolor='#fff'
#     )
#     return fig

# def run_inference(api_url):
#     """Trigger inference via API call"""
#     try:
#         response = requests.post(api_url, timeout=30)
#         if response.status_code == 200:
#             result = response.json()
#             if result.get("status") == "success":
#                 return True, f"✅ Prediction completed for {result.get('timestamp', '')}"
#             else:
#                 return False, f"Error: {result.get('message', 'Unknown error')}"
#         else:
#             return False, f"Error: {response.text}"
#     except requests.exceptions.Timeout:
#         return False, "Error: Request timeout (30s)"
#     except Exception as e:
#         return False, f"Error: {str(e)}"

# def main():
#     # Initialize app components
#     config, data_manager, inference_api_url = initialize_app()
    
#     # Main header
#     st.markdown('<h1 class="main-header">🚴‍♂️ Real-time Bike Count Predictions</h1>', unsafe_allow_html=True)
    
#     # Sidebar - Control Panel
#     with st.sidebar:
#         st.markdown('<h2 class="section-header">Control Panel</h2>', unsafe_allow_html=True)
        
#         # Time range control
#         st.markdown("**Display Time Range**")
#         lookback_hours = st.number_input(
#             "Plots display time (last N hours)",
#             min_value=1,
#             step=1,
#             value=config['ui']['default_lookback_hours'],
#             help="Number of hours to display in the plots"
#         )
        
#         # Feature selection
#         st.markdown("**Feature Selection**")
#         parameter_options = [
#             ('Temperature', 'temp'),
#             ('Humidity', 'hum'),
#             ('Wind Speed', 'windspeed'),
#             ('Week Day', 'weekday'),
#             ('Working Day', 'workingday'),
#             ('Weather', 'weathersit'),
#         ]
        
#         selected_params = st.multiselect(
#             "Select features to display",
#             options=[param[1] for param in parameter_options],
#             default=['temp', 'hum'],
#             format_func=lambda x: next(param[0] for param in parameter_options if param[1] == x)
#         )
        
#         # Inference controls
#         st.markdown("**Inference Control**")
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             run_prediction = st.button(
#                 "🔮 Predict Next Step",
#                 type="primary",
#                 use_container_width=True
#             )
        
#         with col2:
#             auto_refresh = st.button("🔄", help="Refresh data")
        
#         # Auto-refresh option
#         enable_auto_refresh = st.checkbox("Enable auto-refresh (30s)", value=False)
        
#         if enable_auto_refresh:
#             st.rerun()
        
#         # Status display
#         if 'inference_status' not in st.session_state:
#             st.session_state.inference_status = ""
        
#         if st.session_state.inference_status:
#             if "✅" in st.session_state.inference_status:
#                 st.markdown(f'<div class="status-success">{st.session_state.inference_status}</div>', 
#                            unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="status-error">{st.session_state.inference_status}</div>', 
#                            unsafe_allow_html=True)
        
#         # Overview section
#         st.markdown('<h3 class="section-header">ML Application Overview</h3>', unsafe_allow_html=True)
#         st.markdown("""
#         <div class="overview-box">
#         <ul style="font-size: 14px; line-height: 1.4;">
#             <li>End-to-end ML pipeline running in real-time</li>
#             <li>Top plot: Predicted vs actual bike counts</li>
#             <li>Bottom plot: Selected feature data</li>
#             <li>Click "Predict Next Step" to run inference</li>
#             <li>UI and inference run in separate containers</li>
#             <li>Data and models stored in Docker volumes</li>
#         </ul>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Handle inference button click
#     if run_prediction:
#         with st.spinner('Running prediction...'):
#             success, message = run_inference(inference_api_url)
#             st.session_state.inference_status = message
#             if success:
#                 st.rerun()  # Refresh to show new predictions
    
#     # Main content area
#     # Load data
#     df_pred, df_prod, error = load_data(data_manager)
    
#     if error:
#         st.error(f"Error loading data: {error}")
#         # Show error figures
#         col1, col2 = st.columns(1)
#         with col1:
#             st.subheader("Real-time Bike Count Predictions")
#             st.plotly_chart(create_error_figure(error), use_container_width=True)
            
#             st.subheader("Features Data")
#             st.plotly_chart(create_error_figure(error), use_container_width=True)
#     else:
#         # Create and display plots
#         try:
#             # Use session state to maintain zoom state across updates
#             shared_xrange = st.session_state.get('shared_xaxis_range', None)
            
#             fig1, fig2 = make_prediction_figures(
#                 df_prod, df_pred, selected_params, config, lookback_hours, shared_xrange
#             )
            
#             # Display plots
#             st.subheader("📈 Real-time Bike Count Predictions")
#             event1 = st.plotly_chart(fig1, use_container_width=True, key="plot1")
            
#             st.subheader("📊 Features Data")
#             event2 = st.plotly_chart(fig2, use_container_width=True, key="plot2")
            
#             # Handle zoom synchronization (simplified for Streamlit)
#             # Note: Full zoom sync is more complex in Streamlit compared to Dash
            
#         except Exception as e:
#             st.error(f"Error creating plots: {str(e)}")
#             st.plotly_chart(create_error_figure(str(e)), use_container_width=True)
    
#     # Data summary metrics (if data is available)
#     if df_prod is not None and not df_prod.empty:
#         st.markdown("---")
#         st.subheader("📊 Data Summary")
        
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric(
#                 "Total Records",
#                 len(df_prod),
#                 help="Total number of data points"
#             )
        
#         with col2:
#             if 'cnt' in df_prod.columns:
#                 latest_count = df_prod['cnt'].iloc[-1] if not df_prod.empty else 0
#                 st.metric(
#                     "Latest Count",
#                     f"{latest_count:.0f}",
#                     help="Most recent bike count"
#                 )
        
#         with col3:
#             if 'timestamp' in df_prod.columns:
#                 latest_time = df_prod['timestamp'].iloc[-1] if not df_prod.empty else "N/A"
#                 if latest_time != "N/A":
#                     latest_time = pd.to_datetime(latest_time).strftime("%H:%M:%S")
#                 st.metric(
#                     "Last Updated",
#                     latest_time,
#                     help="Time of last data point"
#                 )
        
#         with col4:
#             if df_pred is not None and not df_pred.empty:
#                 prediction_count = len(df_pred)
#                 st.metric(
#                     "Predictions Made",
#                     prediction_count,
#                     help="Number of predictions available"
#                 )

#     # Auto-refresh functionality
#     if enable_auto_refresh:
#         import time
#         time.sleep(30)
#         st.rerun()

# if __name__ == '__main__':
#     main()