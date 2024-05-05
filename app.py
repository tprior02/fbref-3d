from widgets import radio_item, drop_down, check_list, range_slider, slider
from dash import Dash, html, dcc, no_update
from dash_bootstrap_components import Col, Row, Card, themes
from dash.dependencies import Input, Output, State
from plotly.graph_objs import Figure
from plotly.express import scatter_3d, scatter
from maps import *
from textalloc import allocate_text
from sklearn.ensemble import IsolationForest
from ast import literal_eval
from PIL import ImageFont
from query_functions import *

"""
The callbacks follow this pattern:

dataframe -> graph -> limits -> screen width -> x/y per pixel -> outliers -> annotations 

This is so the text box coordinates can be expressed in relation to the x and y range rather than in pixels. The graph 
first needs to be plotted to get the limits. The screen width is needed to interpolate the pixels on the x-axis (as it
is dynamically sized in relation to the screen width). The outliers are calculated. Then the annotations can be plotted 
using code modified from https://github.com/ckjellson/textalloc. 
"""

app = Dash(external_stylesheets=[themes.BOOTSTRAP])

server = app.server

text_size = 10
font = ImageFont.truetype('assets/fonts/arial.ttf', text_size)

app.layout = html.Form(autoComplete="off", children=[
    Col([
    html.Div(id='screen_width', style={'display': 'none'}),
    html.Div(id='limits_of_plot',
             children='[[-0.0643421312,1.095342131],[-0.0261490669,0.4511490669]]', style={'display': 'none'}),
    dcc.Store(id='plot_data'),
    dcc.Store(id='xy_per_pixel'),
    dcc.Store(id='label_data'),
    dcc.Markdown('''
        # FBREF-3D
    
        A dashboard for the visualising player data as scatter plots. Match statistics are from opta, scraped from 
        fbref.com. Player information has been scraped from transfermarkt.com. 
        
        Some features which I think are useful:
        - Outliers are automatically calculated and annotated
        - Annotations avoid overlapping (using code modified from [here](https://github.com/ckjellson/textalloc))
        - Non-outliers can be annotated too
        - Players can have seasons grouped together or they can be shown individually
        - Players can be filtered using a wide range of criteria
        - Hover over a marker to see information about that player
        
        To contact me (and for occasional blog posts) visit [here](https://arsestack.substack.com).
        If you know how to make a dash app look nice please get in touch. 
        '''),
    Card([
        Row([
            Col(html.Label('x-axis')),
            Col(html.Label('y-axis')),
            Col(html.Label('z-axis'))
        ]),
        Row([
            Col([
                drop_down('x_table', categories_dictionary, 'shooting'),
                drop_down('x', shooting_dictionary, 'non_penalty_goals')
            ]),
            Col([
                drop_down('y_table', categories_dictionary, 'passing'),
                drop_down('y', passing_dictionary, 'xa')
            ]),
            Col([
                drop_down('z_table', categories_dictionary, 'possession'),
                drop_down('z', possession_dictionary, 'touch')
            ])
        ])
    ], style={"width": "100%", "height": "50%"}, body=True),
    Card([
        Row([
            Col(Row(html.Label('scatter type'), justify='center')),
            Col(Row(html.Label('marker colours'), justify='center')),
            Col(Row(html.Label('per minute or totals'), justify='center')),
            Col(Row(html.Label('annotation'), justify='center')),
            Col(Row(html.Label('group by season?'), justify='center')),

        ]),
        Row([
            Col(Row(radio_item(id='show_as_2d', options={
                "2D": True,
                "3D": False
            }, value=True), justify='center')),
            Col(Row(radio_item(id="marker_colour_representation", options={
                "position": "position",
                "z-axis": "z",
                "club": "current_club",
            }, value="position"), justify='center')),
            Col(Row(radio_item(id="show_as_per_90", options={
                "per 90": True,
                "totals": False
            }, value=True), justify='center')),
            Col(Row(radio_item(id="show_annotation", options={
                "outliers": True,
                "none": False}, value=True), justify='center')),
            Col(Row(radio_item(id="group_by_season", options={
                "yes": True,
                "no": False}, value=True), justify='center')),
        ]),
    ], style={"width": "100%"}, body=True),
    Card([
        Row([
            Col([
                Row(html.Label('seasons')),
                Row(check_list('chosen_seasons', seasons_dictionary)),
            ]),
            Col([
                Row(html.Label('competitions')),
                Row(check_list('chosen_competitions', competitions_dictionary, 'EL')),
            ]),
            Col([
                Row(html.Label('positions')),
                Row(check_list('chosen_positions', positions_list)),
            ]),
            Col([
                Row(html.Label('foot')),
                Row(check_list('chosen_feet', ['right', 'left', 'both'])),
            ])
        ])
    ], style={"width": "100%"}, body=True),
    Card([
        Row([
            Col(html.Label('age')),
            Col(html.Label('value (€)')),
            Col(html.Label('minutes')),
            Col(html.Label('height (m)')),
            Col(html.Label('no. outliers'))

        ]),
        Row([
            Col(Row(range_slider('chosen_ages', min_age, max_age, min_age, max_age, 1))),
            Col(Row(range_slider('chosen_values', 0, max_value, 0, max_value, 0.5))),
            Col(Row(range_slider('chosen_minutes', 0, max_minutes, 1250, max_minutes, 100))),
            Col(Row(range_slider('chosen_heights', min_height, max_height, min_height, max_height, 0.01))),
            Col(Row(slider('number_of_outliers_to_plot', 0, 50, 25, 1)))
        ]),
    ], style={"width": "100%"}, body=True),
    Card([
        Row([
            Col(html.Label('nationality')),
            Col(html.Label('club')),
            Col(html.Label('agent')),
            Col(html.Label('outfitter')),

        ]),
        Row([
            Col(dcc.Dropdown(id='chosen_nationalities', options=nation_options, value=[], multi=True)),
            Col(dcc.Dropdown(id='chosen_clubs', options=club_options, value=[], multi=True)),
            Col(dcc.Dropdown(id='chosen_agents', options=agent_options, value=[], multi=True)),
            Col(dcc.Dropdown(id='chosen_outfitters', options=outfitter_options, value=[], multi=True)),

        ]),
    ], style={"width": "100%", "height": "50%"}, body=True),
    Card([
        Row(Col(html.Label('names'))),
        Row([Col(dcc.Dropdown(id='additional_players_to_include', options=name_options, value=[], multi=True), width=9),
            Col(radio_item(id="annotate_additional_players", options={"add and annotate": True, "add": False},
                           value=True), width=3),
        ]),
    ], style={"width": "100%", "height": "50%"}, body=True),
    Row(html.Div([dcc.Graph(id='main_plot', config={'displayModeBar': False})]))
])
])

html.Form(autoComplete="off", children=[dcc.Dropdown(id='additional_players_to_include', options=name_options, value=[], multi=True)])

"""
if the plot is 2D
"""
app.clientside_callback(
    """
    function(limits_of_plot, show_as_2d) {
        if (!show_as_2d) {
            return window.dash_clientside.no_update
        } else {
            var w = window.innerWidth;
            return w;
        }
    }
    """,
    Output('screen_width', 'children'),
    Input('limits_of_plot', 'children'),
    State('show_as_2d', 'value'),
)

app.clientside_callback(
    """
    function(fig, show_as_2d) {
        if (!show_as_2d) {
            return window.dash_clientside.no_update
        } else {
            const x_range = fig.layout.xaxis.range;
            const y_range = fig.layout.yaxis.range;
            return JSON.stringify([x_range, y_range])
        }
    }
    """,
    Output('limits_of_plot', 'children'),
    Input('main_plot', 'figure'),
    State('show_as_2d', 'value'),
    prevent_initial_call=True
)


@app.callback(
    Output('xy_per_pixel', 'data'),
    Input('screen_width', 'children'),
    State('limits_of_plot', 'children'),
    prevent_initial_call=True
)
def get_xy_per_pixel(screen_width, limits):
    """
    :param screen_width: the width of the screen in pixels
    :param limits: the limits of the current plot
    :return: the limits of the plot per pixel
    """
    limits = literal_eval(limits)
    x_lims, y_lims = limits[0], limits[1]
    x_pixels = screen_width - (53 + 93 + (121 - 93) * (screen_width - 450) / (1920 - 450))
    x_per_pixel = (x_lims[1] - x_lims[0]) / x_pixels
    y_per_pixel = (y_lims[1] - y_lims[0]) / 700
    return [x_per_pixel, y_per_pixel]


@app.callback(
    Output('plot_data', 'data', allow_duplicate=True),
    inputs=[
        Input('x', 'value'),
        Input('y', 'value'),
        Input('z', 'value'),
        Input('chosen_competitions', 'value'),
        Input('chosen_seasons', 'value'),
        Input('chosen_positions', 'value'),
        Input('chosen_minutes', 'value'),
        Input('chosen_values', 'value'),
        Input('chosen_ages', 'value'),
        Input('chosen_heights', 'value'),
        Input('chosen_clubs', 'value'),
        Input('chosen_nationalities', 'value'),
        Input('chosen_agents', 'value'),
        Input('chosen_outfitters', 'value'),
        Input('chosen_feet', 'value'),
        Input('additional_players_to_include', 'value'),
        Input('show_as_per_90', 'value'),
        Input('group_by_season', 'value'),

    ],
    prevent_initial_call=True
)
def construct_xyz_dataframe(
        x: str,
        y: str,
        z: str,
        chosen_competitions: List[str],
        chosen_seasons: List[str],
        chosen_positions: List[str],
        chosen_minutes: List[int],
        chosen_values: List[int],
        chosen_ages: List[int],
        chosen_heights: List[int],
        chosen_clubs: List[str],
        chosen_nationalities: List[str],
        chosen_agents: List[str],
        chosen_outfitters: List[str],
        chosen_feet: List[str],
        additional_names_to_include: List[str],
        show_as_per_90: bool,
        group_by_season: bool,
) -> pd.DataFrame:
    players_df = players_query(
        chosen_positions,
        chosen_values,
        chosen_ages,
        chosen_heights,
        chosen_clubs,
        chosen_nationalities,
        chosen_agents,
        chosen_outfitters,
        chosen_feet,
        additional_names_to_include,
        group_by_season
    )
    x_df = query(x, chosen_seasons, chosen_competitions, 'x', group_by_season)
    y_df = query(y, chosen_seasons, chosen_competitions, 'y', group_by_season)
    z_df = query(z, chosen_seasons, chosen_competitions, 'z', group_by_season)
    minutes_df = minutes_query(chosen_competitions, chosen_seasons, group_by_season)
    df = players_df.join(x_df).join(y_df).join(z_df).join(minutes_df)
    if show_as_per_90:
        if x not in exclude_from_per_90_set:
            df['x'] = df['x'] / df['minutes'] * 90
        if y not in exclude_from_per_90_set:
            df['y'] = df['y'] / df['minutes'] * 90
        if z not in exclude_from_per_90_set:
            df['z'] = df['z'] / df['minutes'] * 90
    df = df[(df['minutes'] >= chosen_minutes[0]) & (df['minutes'] <= chosen_minutes[1])]
    df = df.fillna(0)
    return df.to_dict('records')


@app.callback(
    Output('main_plot', 'figure', allow_duplicate=True),
    inputs=[
        State('x', 'value'),
        State('y', 'value'),
        State('z', 'value'),
        State('x', 'options'),
        State('y', 'options'),
        State('z', 'options'),
        Input('plot_data', 'data'),
        Input('show_as_2d', 'value'),
        State('show_as_per_90', 'value'),
        Input('marker_colour_representation', 'value'),
        Input('show_annotation', 'value'),
    ],
    prevent_initial_call=True
)
def get_figure(
        x,
        y,
        z,
        x_options,
        y_options,
        z_options,
        plot_data,
        show_as_2d,
        show_as_per_90,
        marker_colour_representation,
        show_annotation):
    x_label = [dictionary['label'] for dictionary in x_options if dictionary['value'] == x][0]
    y_label = [dictionary['label'] for dictionary in y_options if dictionary['value'] == y][0]
    z_label = [dictionary['label'] for dictionary in z_options if dictionary['value'] == z][0]
    x_label = f'{x_label} / 90' if show_as_per_90 and x not in exclude_from_per_90_set else x_label
    y_label = f'{y_label} / 90' if show_as_per_90 and y not in exclude_from_per_90_set else y_label
    z_label = f'{z_label} / 90' if show_as_per_90 and z not in exclude_from_per_90_set else z_label

    df = pd.DataFrame(plot_data)
    graph_params = dict(
        data_frame=df,
        x='x',
        y='y',
        color=marker_colour_representation,
        hover_name='name',
        hover_data={
            'position',
            'value',
            'age',
            'current_club',
            'citizenship',
            'foot',
            'height',
            'minutes'
        },
        labels={
            "x": x_label,
            "y": y_label,
            "z": z_label,
            "name": "name",
            'current_club': "team",
        },
    )
    if show_as_2d:
        plot = scatter
    else:
        plot = scatter_3d
        graph_params['z'] = 'z'
    fig = plot(**graph_params)
    fig.update_layout(
        height=700,
        autosize=True,
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
        font_family='Arial',
        xaxis=dict(hoverformat='.3r'),
        yaxis=dict(hoverformat='.3r'),
    )
    fig.update_traces(
        marker_size=5
    )
    fig.update_coloraxes(
        colorbar_title=dict(
            side='right'
        )
    )
    if not show_as_2d:
        fig.update_scenes(
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="auto"
        )
    return fig


@app.callback(
    Output('label_data', 'data', allow_duplicate=True),
    State('plot_data', 'data'),
    Input('annotate_additional_players', 'value'),
    State('show_annotation', 'value'),
    Input('additional_players_to_include', 'value'),
    Input('xy_per_pixel', 'data'),
    Input('number_of_outliers_to_plot', 'value'),
    prevent_initial_call=True
)
def get_outliers(plot_data,
                 annotate_additional_players,
                 show_annotations,
                 additional_players_to_include,
                 per_pixel,
                 number_of_outliers_to_plot):
    """Stores the outliers as a separate dataframe"""
    if plot_data is None:
        return no_update
    df = pd.DataFrame(plot_data)
    if not show_annotations or number_of_outliers_to_plot == 0:
        df["ifo"] = 1
    else:
        player_count = len(df.index)
        isf = IsolationForest(
            n_estimators=100,
            random_state=42,
            contamination=0.5 if number_of_outliers_to_plot / player_count > 0.5 else number_of_outliers_to_plot /
                                                                                      player_count)
        predicates = isf.fit_predict(df[['x', 'y']].to_numpy())
        df['ifo'] = predicates
    player_count = len(df.index)
    if player_count == 0:
        df = pd.DataFrame()
    elif player_count < 50:
        df = df[['x', 'y', 'last_name']]
    elif annotate_additional_players and show_annotations:
        df = df[(df['ifo'] == -1) | (df['simple_name'].isin(additional_players_to_include))][['x', 'y', 'last_name']]
    elif not annotate_additional_players and show_annotations:
        df = df[df['ifo'] == -1][['x', 'y', 'last_name']]
    else:
        df = pd.DataFrame()
    return df.to_dict('records')


@app.callback(
    Output('main_plot', 'figure', allow_duplicate=True),
    Input('label_data', 'data'),
    State('plot_data', 'data'),
    State('xy_per_pixel', 'data'),
    State('limits_of_plot', 'children'),
    State('show_as_2d', 'value'),
    State('main_plot', 'figure'),

    prevent_initial_call=True
)
def add_annotations(label_data, plot_data, xy_per_pixel, plot_limits, dim, fig):
    """Adds annotations to plot"""
    if not dim:
        return no_update
    fig['layout'].update(annotations=[])
    fig = Figure(fig)
    limits = literal_eval(plot_limits)
    allocate_text(
        x=[player['x'] for player in label_data],
        y=[player['y'] for player in label_data],
        text_list=[player['last_name'] for player in label_data],
        fig=fig,
        x_lims=limits[0],
        y_lims=limits[1],
        x_per_pixel=xy_per_pixel[0],
        y_per_pixel=xy_per_pixel[1],
        font=font,
        x_scatter=[player['x'] for player in plot_data],
        y_scatter=[player['y'] for player in plot_data],
        text_size=text_size
    )
    return fig


@app.callback(
    Output('x', 'options'),
    Input('x_table', 'value'),
    prevent_initial_call=True
)
def update_x_dropdown(table: str):
    return [{"label": key, "value": value} for key, value in get_table_dictionary(table).items()]


@app.callback(
    Output('x', 'value'),
    Input('x', 'options'),
    prevent_initial_call=True
)
def update_x_value(options):
    return options[0]['value']


@app.callback(
    Output('y', 'options'),
    Input('y_table', 'value'),
    prevent_initial_call=True
)
def update_y_dropdown(table):
    return [{"label": key, "value": value} for key, value in get_table_dictionary(table).items()]


@app.callback(
    Output('y', 'value'),
    Input('y', 'options'),
    prevent_initial_call=True
)
def update_y_value(options):
    return options[0]['value']


@app.callback(
    Output('z', 'options'),
    Input('z_table', 'value'),
    prevent_initial_call=True
)
def update_z_dropdown(table):
    return [{"label": key, "value": value} for key, value in get_table_dictionary(table).items()]


@app.callback(
    Output('z', 'value'),
    Input('z', 'options')
)
def update_z_value(options):
    return options[0]['value']


def get_table_dictionary(category: str) -> dict:
    match category:
        case 'keepers':
            dictionary = keepers_dictionary
        case 'keepersadv':
            dictionary = keepersadv_dictionary
        case 'shooting':
            dictionary = shooting_dictionary
        case 'passing':
            dictionary = passing_dictionary
        case 'passing_types':
            dictionary = passing_types_dictionary
        case 'gca':
            dictionary = gca_dictionary
        case 'defense':
            dictionary = defense_dictionary
        case 'possession':
            dictionary = possession_dictionary
        case 'playingtime':
            dictionary = playingtime_dictionary
        case 'misc':
            dictionary = misc_dictionary
        case _:
            return dict()
    return dictionary


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)
