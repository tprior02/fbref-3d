from dash import dcc


def drop_down(id, options, value):
    return dcc.Dropdown(
            id=id,
            options=options if isinstance(options, list) else [{"label": k, "value": v} for k, v in options.items()],
            value=value,
            clearable=False
        )


def check_list(id, options, to_remove=[]):
    return dcc.Checklist(
        options=options if isinstance(options, list) else [{'label': k, 'value': v} for k, v in options.items()],
        value=options if isinstance(options, list) else [value for value in options.values() if value not in to_remove],
        id=id
    )


def radio_item(id, options, value):
    return dcc.RadioItems(
        options=options if isinstance(options, list) else [{'label': k, 'value': v} for k, v in options.items()],
        value=value,
        id=id,
        labelStyle={'display': 'inline-block'}
    )


def range_slider(id, min, max, value_min, value_max, step):
    return dcc.RangeSlider(
        min=min,
        max=max,
        value=[value_min, value_max],
        step=step,
        allowCross=False,
        id=id,
        marks={min: str(min), max: str(max)},
        tooltip={"placement": "bottom", "always_visible": True},
    )


def slider(id, min, max, value, step):
    return dcc.Slider(
        min=min,
        max=max,
        value=value,
        step=step,
        id=id,
        marks={min: str(min), max: str(max)},
        tooltip={"placement": "bottom", "always_visible": True},
    )

