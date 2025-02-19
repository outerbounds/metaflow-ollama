import pandas as pd
import altair as alt

def beams_to_dataframe(beams):
    records = []
    for beam in beams:
        for step in beam.analysis:
            records.append({
                "BeamIndex": beam.index,
                "Iteration": step.iteration,
                "Score": step.score,
                "Pruned": step.pruned,
                "PrunedReason": step.pruned_reason or "",
                "CurrentText": step.current_text or ""
            })
    df = pd.DataFrame(records)
    return df


def trim_pruned_rows(df):
    """
    Return a copy of df where any row after the earliest prune iteration
    for each beam is removed. This ensures the line stops exactly where
    the beam was pruned.
    """

    # Copy to avoid mutating original df
    df_line = df.copy()

    # 1. Find earliest prune iteration per BeamIndex
    #    i.e., the smallest Iteration where Pruned == True
    pruned_iters = (
        df_line[df_line["Pruned"] == True]
        .groupby("BeamIndex")["Iteration"]
        .min()
    )

    # 2. For each beam that was pruned, drop rows with Iteration > that prune iteration
    for beam_index, prune_iter in pruned_iters.items():
        mask = (df_line["BeamIndex"] == beam_index) & (df_line["Iteration"] > prune_iter)
        df_line = df_line[~mask]

    return df_line

def plot_beam_scores_altair(df):
    """
    Plot lines that end at the prune iteration,
    plus red triangles at the exact iteration each beam was pruned.
    """
    # Make sure Iteration is numeric so Altair can draw lines in numeric order
    df["Iteration"] = df["Iteration"].astype(int)

    # Create a version of df that stops at the earliest prune iteration
    df_line = trim_pruned_rows(df)

    # Base chart for lines
    line_base = alt.Chart(df_line).encode(
        x=alt.X("Iteration:Q", title="Iteration", sort=None),
        y=alt.Y("Score:Q", title="Score"),
        color=alt.Color("BeamIndex:N", title="Beam Index"),
        order="Iteration:Q",  # Connect in ascending iteration order
        tooltip=[
            alt.Tooltip("BeamIndex:N", title="Beam"),
            alt.Tooltip("Iteration:Q", title="Iteration"),
            alt.Tooltip("Score:Q", title="Score"),
            alt.Tooltip("PrunedReason:N", title="Pruned Reason"),
            alt.Tooltip("CurrentText:N", title="Current Text")
        ],
    )

    # Layer 1: lines + optional points
    lines = line_base.mark_line(point=True)

    # Layer 2: pruned points (triangles) from the *original* df
    pruned_points = (
        alt.Chart(df)
        .transform_filter("datum.Pruned == true")
        .mark_point(shape="triangle-down", size=100, color="red")
        .encode(
            x="Iteration:Q",
            y="Score:Q",
            color=alt.Color("BeamIndex:N", title="Beam Index"),
            tooltip=[
                alt.Tooltip("BeamIndex:N", title="Beam"),
                alt.Tooltip("Iteration:Q", title="Iteration"),
                alt.Tooltip("Score:Q", title="Score"),
                alt.Tooltip("PrunedReason:N", title="Pruned Reason"),
                alt.Tooltip("CurrentText:N", title="Current Text")
            ]
        )
    )

    chart = alt.layer(lines, pruned_points).properties(
        title="Beam Scores over Iterations",
        width=600,
        height=400
    ).interactive()

    return chart