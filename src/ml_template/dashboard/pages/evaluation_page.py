"""Read-only Final evaluation tab. No callback accesses dataset labels or runs evaluation."""

import json

from ml_template.dashboard.evaluation_data import list_evaluations, load_evaluation
from ml_template.dashboard.evaluation_figs import evaluation_figures


def _table(rows):
    from dash import dash_table, html

    if not rows:
        return html.P("Not requested / not available.")
    columns = list(dict.fromkeys(k for r in rows for k in r))
    records = [
        {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for k, v in r.items()
        }
        for r in rows
    ]
    return dash_table.DataTable(
        data=records,
        columns=[{"name": k, "id": k} for k in columns],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "100px",
            "maxWidth": "450px",
        },
    )


def layout(engine):
    from dash import dcc, html

    return html.Div(
        [
            html.H2("Final evaluation"),
            html.P(
                "Saved frozen-model assessments only. This page never trains, calibrates, selects policies or opens a new holdout evaluation."
            ),
            html.Button("Refresh saved evaluations", id="final-eval-refresh", n_clicks=0),
            dcc.Dropdown(
                id="final-eval-id", options=[], placeholder="Select an evaluation", clearable=True
            ),
            html.Div(id="final-eval-content"),
        ],
        style={"padding": "1rem"},
    )


def render_selection(engine, evaluation_id):
    from dash import dcc, html

    loaded = load_evaluation(engine, evaluation_id)
    if loaded["state"] != "verified":
        return html.Div([html.H3(loaded["state"]), html.P(loaded["message"])], role="status")
    data = loaded["data"]
    sections = [
        ("Lineage and assessment", [data["summary"]]),
        ("Raw and calibrated holdout metrics", data["metrics"]),
        ("Calibrated minus raw", data["differences"]),
        ("Frozen operating policy", data["policy_results"]),
        (
            "Development policy-selection provenance: not final-test evidence",
            [data["policy_selection"]] if data.get("policy_selection") else [],
        ),
        ("Acceptance criteria: not automatic promotion", data["acceptance"]["criteria"]),
        ("Reliability support", data["reliability"]),
        ("Subgroups: global policy, exploratory", data["subgroups"]),
        ("Development context: not independent final-test evidence", data["development_context"]),
    ]
    children = [
        html.P(loaded["message"]),
        html.H3("Warnings and limitations"),
        html.Ul([html.Li(w) for w in data["warnings"]]),
    ]
    for title, rows in sections:
        children.extend([html.H3(title), _table(rows)])
    children.extend(dcc.Graph(figure=f) for f in evaluation_figures(data))
    children.extend(
        [
            html.Details(
                [html.Summary("Frozen specification"), html.Pre(json.dumps(data["spec"], indent=2))]
            ),
            html.Details(
                [html.Summary("Provenance"), html.Pre(json.dumps(data["provenance"], indent=2))]
            ),
            html.Details(
                [
                    html.Summary("Bounded policy error examples (contain labels/features)"),
                    _table(data["errors"]),
                ]
            ),
        ]
    )
    return html.Div(children)


def register_callbacks(app, engine):
    from dash import Input, Output, State

    @app.callback(
        Output("final-eval-id", "options"),
        Output("final-eval-id", "value"),
        Input("final-eval-refresh", "n_clicks"),
        State("final-eval-id", "value"),
    )
    def refresh(_clicks, current):
        records = list_evaluations(engine)
        options = [
            {
                "label": f"{r['evaluation_id']} | {r['parent_run_id']} | {r['status']}",
                "value": r["evaluation_id"],
            }
            for r in records
        ]
        # Do not automatically reveal a final holdout result on page load.
        return options, current if any(r["evaluation_id"] == current for r in records) else None

    @app.callback(Output("final-eval-content", "children"), Input("final-eval-id", "value"))
    def select(evaluation_id):
        return render_selection(engine, evaluation_id)


# calibrated-policy update v1
