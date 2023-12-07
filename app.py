import io
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.html import Hr
from dash.development.base_component import Component
from dash import dash_table
import jupyter_dash
import dash_bootstrap_components as dbc
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import dash
from dash import dcc, html
import plotly.graph_objs as go
from yahoo_fin import stock_info as si

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# jupyter_dash.JupyterDash(external_stylesheets=['/content/style.css']
dff=pd.read_csv("./datos/Prices.csv")
dff.head(2)
#change Date column type to datetime type
dff['Date'] = pd.to_datetime(dff['Date'])
# check changing type
dff['Date'] = pd.to_datetime(dff['Date'], format='%Y-%m-%d')
dff.info()
dff.tail(2)
df = dff.copy()
df.tail(2)
df["year"] = df["Date"].map(lambda x : x.year)
df["year"].tail(2)
df_mean_by_year = pd.DataFrame(df.groupby(["year"]).mean())
df_mean_by_year.tail()
# print(df_mean_by_year.loc[2021,"United States(USD)"])
for col in df.iloc[:,1:-1].columns:
    print(col)
    print(df[col].isna().sum())
    if 2023 in df_mean_by_year.index:
        df[col] = df[col].fillna(value=df_mean_by_year.loc[2023,col])
# #filter DataFrame depend on start year and end year
# # df[(df['Date'] <= '2015') & (df['Date'] >= '2010')]
df_mean_by_year.reset_index(inplace=True)
df_mean_by_year.head(2)
df_max_by_year = pd.DataFrame(df.groupby(["year"]).max())
df_max_by_year.reset_index(inplace=True)
df_max_by_year.drop("Date",axis=1,inplace=True)
df_max_by_year.tail()
df.tail(2)
# List of currencies 
currency = list(df.iloc[:,1:-1].columns)
currency
currency[0:2]
# years = pd.to_datetime(df['Date']).dt.year
years = df["year"]
years
years=list(years.unique())
years
colors = {
    'background': '#03045e',
    'background2': '#001845',
    'text': 'White'
}
df["Date"]
df_new =df_mean_by_year[(df_mean_by_year['year'] < 2010) & (df_mean_by_year['year'] >= 2000)]
px.bar(df_new,x='year', y=currency)
df4=pd.read_csv("./datos/metal_rates.csv")
jupyter_dash.JupyterDash(__name__)



def remove_outliers_with_limits(df, data_column, lower_limit_column, upper_limit_column):
    """
    Esta función elimina los outliers de un DataFrame basado en los límites predefinidos en otras columnas.

    :param df: DataFrame de pandas.
    :param data_column: El nombre de la columna con los datos para verificar outliers.
    :param lower_limit_column: El nombre de la columna con el límite inferior.
    :param upper_limit_column: El nombre de la columna con el límite superior.
    :return: DataFrame de pandas sin outliers.
    """
    # Asegúrate de que los límites son numéricos y no contienen valores nulos
    df[lower_limit_column] = pd.to_numeric(df[lower_limit_column], errors='coerce')
    df[upper_limit_column] = pd.to_numeric(df[upper_limit_column], errors='coerce')

    # Filtrar el DataFrame para eliminar los valores fuera de los límites
    filter = (df[data_column] >= df[lower_limit_column]) & (df[data_column] <= df[upper_limit_column])
    return df.loc[filter]

def get_metal_prices():
    metals = ['SI=F', 'GC=F', 'PL=F', 'PA=F', 'HG=F']
    metal_names = ['Plata', 'Oro', 'Platino', 'Paladio', 'Cobre']
    roundBy = 1
    prices = {}

    for metal, name in zip(metals, metal_names):
        try:
            price = round(si.get_live_price(metal), roundBy)
            prices[name] = price
        except Exception as e:
            print(f"Error al obtener precio para {name}: {e}")
            prices[name] = None

    return prices

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Leer la hoja 'BD Leyes'
        df_bd_leys = pd.read_excel(io.BytesIO(decoded), sheet_name='BD Leyes', skiprows=1)
        # Leer la hoja 'Calculos' desde B16 hasta H33
        df_calculos_1 = pd.read_excel(io.BytesIO(decoded), sheet_name='Calculos', skiprows=15, nrows=17, usecols='B:H')
        # Leer otra tabla en la hoja 'Calculos' desde K16 hasta Q31
        df_calculos_2 = pd.read_excel(io.BytesIO(decoded), sheet_name='Calculos', skiprows=15, nrows=15, usecols='K:Q')
        
        # Borrar la primera columna y la columna 'Precio ($/lb)' si existe
        df_calculos_2 = df_calculos_2.iloc[:, 1:]
        if 'Precio ($/lb)' in df_calculos_2.columns:
            df_calculos_2 = df_calculos_2.drop(columns=['Precio ($/lb)'])

        # Obtener los precios de los metales
        metal_prices = get_metal_prices()

        # Agregar cada precio de metal como su propia columna
        for metal, price in metal_prices.items():
            df_calculos_2[f'Precio {metal}'] = price
        
        # Leer la hoja 'BD Wi'
        df_bd_wi = pd.read_excel(io.BytesIO(decoded), sheet_name='BD Wi')
    except Exception as e:
        print(e)
        return None, None, None, None

    # Retornar los DataFrame modificados
    return df_bd_leys, df_calculos_1, df_calculos_2, df_bd_wi



app.layout = html.Div([
    # Tu nuevo bloque de código para el logo y la franja
    html.Div([
        html.Div(style={'background-color': '#d9d9d9', 'height': '100px', 'position': 'absolute', 'width': '140%', 'z-index': '1'}),
        html.Img(src='/assets/logo.png', style={'height':'100px', 'width':'auto', 'position': 'absolute', 'z-index': '2'}),
    ], style={'position': 'relative'}),

    # Espacio adicional para mover los Tabs hacia abajo
    html.Div(style={'height': '14vh', 'width': '100%', 'margin': '0', 'padding': '0'}),

    # Los Tabs
    dbc.Tabs([
        dbc.Tab(label="Precios de los Metales", style={"text-align": "center"}, children=[
    # Contenido de la primera pestaña
    html.Div([
        html.Div([
            html.Div([
                html.H3('Precios de los Metales'),
                html.A(
                    html.Img(
                        src="http://www.kitconet.com/images/quotes_7a.gif",
                        alt="[Most Recent Quotes from www.kitco.com]"
                    ),
                    href="http://www.kitco.com/connecting.html",
                )
            ], className="align-items-center")
        ], className="d-flex justify-content-center"),

        # Contenedor para los gráficos del oro y la plata
        html.Div([
            # Gráfico del Precio del Oro
            html.Div([
                html.Div([
                    html.H3('Precio del Oro de las ultimas 24hs'),
                    html.A(
                        html.Img(
                            src="http://www.kitconet.com/images/live/s_gold.gif",
                            alt="[Most Recent Quotes from www.kitco.com]",
                            height="300px",
                            width="300px"
                        ),
                        href="http://www.kitco.com/connecting.html",
                    )
                ], className="align-items-center")
            ], className="d-flex justify-content-center"),

            # Gráfico del Precio del Cobre
            html.Div([
                html.Div([
                    html.H3('Precio del Cobre de las últimas 24 horas', style={'color': 'white', 'text-align': 'center'}),
                    html.Div([
                        html.A(
                            html.Img(
                                src="http://www.kitconet.com/charts/metals/base/t24_cp450x275.gif",
                                alt="[Gráfico del precio del cobre de las últimas 24 horas de www.kitco.com]",
                                style={'height': '275px', 'width': '450px', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}
                            ),
                            href="http://www.kitco.com/market/",
                        )
                    ], style={'background-color': 'black', 'padding': '10px'})
                ], style={'text-align': 'center', 'margin': 'auto'})
            ], className="d-flex justify-content-center flex-column"),


            # Gráfico del Precio de la Plata
            html.Div([
                html.Div([
                    html.H3('Precio de la Plata de las ultimas 24hs'),
                    html.A(
                        html.Img(
                            src="http://www.kitconet.com/images/live/s_silv.gif",
                            alt="[Most Recent Quotes from www.kitco.com]",
                            height="300px",
                            width="300px"
                        ),
                        href="http://www.kitco.com/connecting.html",
                    )
                ], className="align-items-center")
            ], className="d-flex justify-content-center"),
        ], className="d-flex justify-content-around")
    ])
]),
        dbc.Tab(label="Selección de Moneda", children=[
            html.Div([           
                html.Br(), html.Br(),           
                html.Div([               
                    html.Br(), html.Br(),                 
                    html.P('Select Currency:', className='fix_label', style={'color': 'White', "background": colors["background2"]}),                 
                    dcc.Dropdown(
                        currency,
                        id='Currency',
                        multi=False,
                        clearable=True,
                        disabled=False,
                        style={'display': True},
                        value='United States(USD)',                              
                        className='dcc_compon'                                
                    ), 
                    html.P('Select Start Year:', className='fix_label', style={'color': 'White', "background": colors["background2"]}),
                    dcc.Dropdown(
                        options=[{'label': str(year), 'value': year} for year in years],
                        id='minyear',
                        multi=False,
                        clearable=True,
                        disabled=False,
                        style={'display': True},
                        value='1978',                     
                        className='dcc_compon'
                    ),
                    html.P('Select End Year:', className='fix_label', style={'color': 'White', "background": colors["background2"]}),
                    dcc.Dropdown(
                        options=[{'label': str(year), 'value': year} for year in years],
                        id='maxyear',
                        multi=False,
                        clearable=True,
                        disabled=False,
                        style={'display': True},
                        value='2023',                             
                        className='dcc_compon'
                    ),               
                ], className="create_container three columns", style={"background": colors["background2"] }),
                html.Div([      
                    html.Br(), html.Br(),     
                    html.P('GOLD CALCULATOR', className='fix_label', style={'color': colors["text"], "text-align": "center", "background": colors["background2"]}),      
                    html.P('Enter The money:', className='fix_label', style={"text-align": "left", 'color': 'White'}),   
                    dcc.Input(id='money', type='number', className='dcc_compon'),
                    html.P('You can Buy in (g):', className='fix_label', style={'color': 'White'}),
                    html.Div(id="my_div", style={"color": "White"})                             
                ], className="create_container three columns", style={"background": colors["background2"] })
            ], className="row flex-display", style={"background": colors["background"] }),
                html.Div([
                    html.Br(),
                    dcc.Graph(id="line graph")            
                ], className="create_container six columns", style={"width": "90%", "color": "#001845", "background": "#001845" }),
                
        ]),
        dbc.Tab(label="Modelo", children=[
            html.Div([
                html.H1("Modelo Geometalurgico ", style={'color': 'White'}),
                # Video insertado aquí
                html.Video(src="https://qoravideo.netlify.app/Video.mp4", controls=True, style={'width': '100%'}),

                # Botón para descargar el archivo
                html.Button("Descargar Excel", id="btn-descargar-excel", style={'color': 'white'}),

                # Título agregado entre los botones
                html.H3(" ", style={'color': 'white', 'text-align': 'center'}),

                dcc.Download(id="descarga-excel"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('Subir Archivo Excel', style={'color': 'white'}),
                    multiple=False
                ),
                dcc.Graph(id='scatter-plot', style={'display': 'none'}),
                html.Div(id='output-data-upload', style={'color': 'white'}),
                html.H1("Calcular Ganancias", style={'text-align': 'center', 'color': 'white'}),
                html.Div([
                    html.Div([
                        html.H3("Tonelaje"),
                        dcc.Input(id='input_tonelaje', type='number', placeholder='Tonelaje', min=0, step=0.1)
                    ], className='four columns'),
                    html.Div([
                        html.H3("Tipo de Material"),
                        dcc.Dropdown(id='input_tipo_material', options=[
                            {'label': 'Cobre', 'value': 'cobre'},
                            {'label': 'Oro', 'value': 'oro'},
                            {'label': 'Plata', 'value': 'plata'}
                        ], placeholder='Selecciona un tipo de material')
                    ], className='four columns'),
                    dcc.Graph(id='new-scatter-plot-wi', style={'display': 'none'}),
                    html.Div(id='output_ganancias'),
                    html.Div(id='output-table-2'),
                ])
            ])
        ])
    ])
])

@app.callback(
    Output("descarga-excel", "data"),
    Input("btn-descargar-excel", "n_clicks"),
    prevent_initial_call=True
)
def descargar_excel(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Ruta al archivo Excel que quieres que los usuarios puedan descargar
    ruta_archivo_excel = './Simulacion.xlsx'

    # Retorna el contenido del archivo para la descarga
    return dcc.send_file(ruta_archivo_excel)

@app.callback(
    [dash.dependencies.Output('scatter-plot', 'figure'),
     dash.dependencies.Output('scatter-plot', 'style'),
     dash.dependencies.Output('output-data-upload', 'children'),
     dash.dependencies.Output('output-table-2', 'children'),
     dash.dependencies.Output('new-scatter-plot-wi', 'figure'),  # Salida para el gráfico de dispersión
     dash.dependencies.Output('new-scatter-plot-wi', 'style')],  # Salida adicional para el estilo del gráfico de dispersión
    [dash.dependencies.Input('upload-data', 'contents')]
)

def update_output(contents):
    if contents is None:
        # Si no hay contenido, devuelve los valores predeterminados
        return dash.no_update, {'display': 'none'}, dash.no_update, dash.no_update, go.Figure(), {'display': 'none'}

    df, cal_df, cal_df_2, df_bd_wi = parse_contents(contents)
    if df is None or cal_df is None or cal_df_2 is None or df_bd_wi is None:
        return dash.no_update, {'display': 'none'}, 'Hubo un error procesando el archivo.', dash.no_update, dash.no_update

    # Usar df_bd_wi en lugar de leer de nuevo el archivo
    # Por ejemplo, renombrar columnas y realizar cálculos
    df_bd_wi.columns = ['Codigo', 'xi_Wi', 'yi_Rec_Cu', 'x_y', 'xi2', 'yi2', 'xi_X', 'yi_Y', 'Promedio', 'Lim_Min', 'Lim_Max', 'y_prime']
    df_bd_wi['y_calculada'] = -2.0069 * df_bd_wi['xi_Wi'] + 104.32

    # Creando el gráfico de dispersión
    scatter_plot = go.Figure(data=[
        go.Scatter(x=df_bd_wi['xi_Wi'], y=df_bd_wi['y_calculada'], mode='markers', name='Datos BD Wi')
    ])
    # Ajustando el diseño del gráfico
    scatter_plot.update_layout(
        title='Gráfico de Dispersión de la hoja BD Wi',
        xaxis_title='xi Wi',
        yaxis_title='y calculada',
        template='plotly_dark'
    )
# Mantener el gráfico oculto si hay un error
    # Eliminar filas que contienen valores NaN en las columnas relevantes
    df_clean = df.dropna(subset=['xi\nCuOx/CuT, %', 'yi\nRec Cu'])
    # Extraer las columnas relevantes para la regresión
    x = df_clean['xi\nCuOx/CuT, %']
    y = df_clean['yi\nRec Cu']
    # Identificar y eliminar outliers
    #outlier_indices_manual = (x > 2000) | (y > 6000)
    outlier_indices_manual = (x > 2000) | (y > 1000)
    x_no_outliers_manual = x[~outlier_indices_manual]
    y_no_outliers_manual = y[~outlier_indices_manual]
    # Standardize the data for density estimation
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_no_outliers_manual.values.reshape(-1, 1))
    y_scaled = scaler.fit_transform(y_no_outliers_manual.values.reshape(-1, 1))
    # Estimate the density of the data points
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(np.hstack([x_scaled, y_scaled]))
    # Calculate sample weights based on the density
    sample_weights = np.exp(kde.score_samples(np.hstack([x_scaled, y_scaled])))
    # Fit a weighted linear regression model
    model_linear_weighted = LinearRegression()
    model_linear_weighted.fit(x_no_outliers_manual.values.reshape(-1, 1), y_no_outliers_manual.values.reshape(-1, 1), sample_weight=sample_weights)
    y_pred_linear_weighted = model_linear_weighted.predict(x_no_outliers_manual.values.reshape(-1, 1))
    # Calculate R^2 and MSE
    r2_linear_weighted = r2_score(y_no_outliers_manual.values.reshape(-1, 1), y_pred_linear_weighted, sample_weight=sample_weights)
    mse_linear_weighted = mean_squared_error(y_no_outliers_manual.values.reshape(-1, 1), y_pred_linear_weighted, sample_weight=sample_weights)
    # Create the plot
    figure = {
        'data': [
            go.Scatter(x=x_no_outliers_manual, y=y_no_outliers_manual, mode='markers', name='Data points'),
            go.Scatter(x=x_no_outliers_manual, y=y_pred_linear_weighted.flatten(), mode='lines', name='Weighted linear regression line', line={'color': 'red'}),
            go.Scatter(x=x_no_outliers_manual, y=y_pred_linear_weighted.flatten() + 20, mode='lines', name='Limite superior', line={'color': 'blue', 'dash': 'dash'}),
            go.Scatter(x=x_no_outliers_manual, y=y_pred_linear_weighted.flatten() - 20, mode='lines', name='Limite inferior', line={'color': 'green', 'dash': 'dash'}),
        ],
        'layout': {
            'xaxis': {'title': 'xi (CuOx/CuT, %)'},
            'yaxis': {'title': 'yi (Rec Cu)'},
            'title': 'Weighted Linear Regression for Material Recovery',
            'legend': {'x': 1.05, 'y': 1}
        }
    }
    output_data = [
        html.Label(f"Weighted Linear Regression - Coefficient of Determination (R²): {r2_linear_weighted:.2f}"),
        html.Br(),
        html.Label(f"Weighted Linear Regression - Mean Squared Error (MSE): {mse_linear_weighted:.2f}")
    ]
    # Crea la segunda tabla con cal_df_2
    table_2 = dash_table.DataTable(
        data=cal_df_2.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in cal_df_2.columns]
    )
     # Añadir la lógica para el nuevo gráfico de dispersión
    # Decodificar y leer el archivo Excel para la hoja "BD Wi"
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df_wi = pd.read_excel(io.BytesIO(decoded), sheet_name='BD Wi')

    # Renombrar las columnas
    df_wi.columns = ['Codigo', 'xi_Wi', 'yi_Rec_Cu', 'x_y', 'xi2', 'yi2', 'xi_X', 'yi_Y', 'Promedio', 'Lim_Min', 'Lim_Max', 'y_prime']
    
    # Calcular y_calculada
    df_wi['y_calculada'] = -2.0069 * df_wi['xi_Wi'] + 104.32

    
    # Calcular el IQR para 'yi_Rec_Cu'
    Q1_real = df_wi['yi_Rec_Cu'].quantile(0.25)
    Q3_real = df_wi['yi_Rec_Cu'].quantile(0.75)
    IQR_real = Q3_real - Q1_real

    # Definir límites para filtrar outliers en datos reales
    lower_bound_real = Q1_real - 1.5 * IQR_real
    upper_bound_real = Q3_real + 1.5 * IQR_real

    # Filtrar el DataFrame para excluir outliers en datos reales
    filtered_df_real = df_wi[(df_wi['yi_Rec_Cu'] >= lower_bound_real) & (df_wi['yi_Rec_Cu'] <= upper_bound_real)]

    # Calcular el IQR para 'yi_Rec_Cu'
    Q1_real = df_wi['yi_Rec_Cu'].quantile(0.25)
    Q3_real = df_wi['yi_Rec_Cu'].quantile(0.75)
    IQR_real = Q3_real - Q1_real

    # Definir límites para filtrar outliers en datos reales
    lower_bound_real = Q1_real - 1.5 * IQR_real
    upper_bound_real = Q3_real + 1.5 * IQR_real

    # Filtrar el DataFrame para excluir outliers en datos reales
    filtered_df_real = df_wi[(df_wi['yi_Rec_Cu'] >= lower_bound_real) & (df_wi['yi_Rec_Cu'] <= upper_bound_real)]

    # Crear el gráfico de dispersión con los datos calculados (sin outliers)
    scatter_plot_wi = go.Figure(data=[
        go.Scatter(x=filtered_df_real['xi_Wi'], y=filtered_df_real['y_calculada'], mode='markers', name='Datos Calculados')
    ])

    # Añadir los valores reales al gráfico (sin outliers)
    scatter_plot_wi.add_trace(
        go.Scatter(x=filtered_df_real['xi_Wi'], y=filtered_df_real['yi_Rec_Cu'], mode='markers', name='Datos Reales', marker_color='red')
    )

    # Añadir línea de referencia para el límite superior
    scatter_plot_wi.add_hline(y=upper_bound_real, line_dash="dash", line_color="green", annotation_text="Límite Superior")

    # Añadir línea de referencia para el límite inferior
    scatter_plot_wi.add_hline(y=lower_bound_real, line_dash="dash", line_color="blue", annotation_text="Límite Inferior")

    # Devolver todos los componentes necesarios, incluyendo las dos tablas y los dos gráficos
    return figure, {'display': 'block'}, output_data, table_2, scatter_plot_wi, {'display': 'block'}

#TONELAJE
# Inicializar la variable global
tonelaje_estimado = 28.6  # o cualquier valor inicial que desees
# Actualizar la variable tonelaje_estimado basado en la entrada del usuario
@app.callback(
    Output('some-output-component', 'some-property'),  # Actualiza el componente que depende de tonelaje_estimado
    [Input('tonelaje-input', 'value')]
)
def update_output(value):
    global tonelaje_estimado
    tonelaje_estimado = value
    return value





@app.callback( 
    Output(component_id="line graph",component_property='figure'),
    Input(component_id='Currency',component_property='value'),
    Input(component_id='minyear',component_property='value'),
    Input(component_id='maxyear',component_property='value'),
)

def line_chart_for_country_change(coun,mini,maxi):    
    mini_date = pd.Timestamp(f"{mini}-01-01")
    maxi_date = pd.Timestamp(f"{maxi}-12-31")
    df_new = df[(df['Date'] >= mini_date) & (df['Date'] <= maxi_date)]
    fig = go.Figure()
    fig.add_trace(
    go.Line(x=df_new["Date"], y=df_new[coun]))
    fig.update_layout(       
    xaxis=dict(        
        title= "year",       
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
    ,
       yaxis=dict(
        title=coun,
        titlefont_size=20,
        tickfont_size=20,
    ),
)    
    return fig

    
@app.callback(
    Output(component_id="my_div",component_property='children'),
    Input(component_id='Currency',component_property='value'),
    Input(component_id='money',component_property='value')

)
def gold_calculator(curr, mon):
    if mon is None:
        return 0
    else:
        # Precio del oro por onza en la moneda seleccionada
        price_in_ounce = df[curr][df.shape[0] - 1]
        
        # Convertir el precio por onza a precio por gramo (1 onza = 31.1035 gramos)
        price_in_gram = price_in_ounce / 31.1035
        
        # Calcular la cantidad de oro en gramos que se puede comprar con el dinero dado
        grams_of_gold = mon / price_in_gram
        
        return grams_of_gold



@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),   
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, "Gold-Prices.csv")

if __name__ == '__main__':
    # Cambia el puerto 5000 al puerto por defecto que desees, por ejemplo, 80
    app.run(host='0.0.0.0', port=80)
