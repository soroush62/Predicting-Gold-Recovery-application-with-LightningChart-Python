import os
import time
import trimesh
import pandas as pd
import lightningchart as lc
import pandas as pd

lc.set_license("my-license-key")

df = pd.read_csv("Dataset/gold_recovery_full_with_profit.csv")
df = df.iloc[::10]

conc_features = [
    "final.output.concentrate_ag",
    "final.output.concentrate_pb",
    "final.output.concentrate_sol",
    "final.output.concentrate_au",
]
tail_features = [
    "final.output.tail_ag",
    "final.output.tail_pb",
    "final.output.tail_sol",
    "final.output.tail_au",
]
for feat in conc_features + tail_features:
    df[feat] = pd.to_numeric(df[feat], errors="coerce")

# Compute min/max for each concentrate feature for dynamic color mapping
min_conc = {feat: df[feat].min() for feat in conc_features}
max_conc = {feat: df[feat].max() for feat in conc_features}

# Compute min/max for tail features for dynamic scaling
min_tail = {feat: df[feat].min() for feat in tail_features}
max_tail = {feat: df[feat].max() for feat in tail_features}


# 2. Define File Paths for 3D Models

# Concentrate objects use rock.OBJ; tail objects use tail.obj.
rock_file = (
    r"D:/Computer Aplication/WorkPlacement/Projects/Project22/3D objects/rock.OBJ"
)
tail_file = (
    r"D:/Computer Aplication/WorkPlacement/Projects/Project22/3D objects/tail.obj"
)


def load_mesh_model(file_name):
    file_path = os.path.abspath(file_name)
    if not os.path.exists(file_path):
        print(f"Missing model file: {file_path}")
        return None, None, None
    try:
        scene = trimesh.load(file_path)
        mesh = (
            scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
        )
        vertices = mesh.vertices.flatten().tolist()
        indices = mesh.faces.flatten().tolist()
        normals = mesh.vertex_normals.flatten().tolist()
        return vertices, indices, normals
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def lerp_color(color1, color2, factor):
    """Linearly interpolate between two RGB colors (tuples) using factor in [0,1]."""
    r = int(color1[0] + (color2[0] - color1[0]) * factor)
    g = int(color1[1] + (color2[1] - color1[1]) * factor)
    b = int(color1[2] + (color2[2] - color1[2]) * factor)
    return lc.Color(r, g, b)


# Define display names for concentrate features.
display_names_conc = {
    "final.output.concentrate_ag": "Silver Concentration",
    "final.output.concentrate_pb": "Lead Concentration",
    "final.output.concentrate_sol": "Solid Concentration",
    "final.output.concentrate_au": "Gold Concentration",
}

# Define display names for tail features.
display_names_tail = {
    "final.output.tail_ag": "Silver Concentration in Tail",
    "final.output.tail_pb": "Lead Concentration in Tail",
    "final.output.tail_sol": "Solid Concentration in Tail",
    "final.output.tail_au": "Gold Concentration in Tail",
}

# Define color ranges for concentrate (rock) features:
conc_color_ranges = {
    "final.output.concentrate_ag": (
        (160, 160, 160),
        (220, 220, 220),
    ),  # Dark silver -> Light silver
    "final.output.concentrate_pb": (
        (50, 100, 150),
        (157, 208, 246),
    ),  # Dark steel blue -> Light steel blue
    "final.output.concentrate_sol": (
        (53, 53, 53),
        (161, 161, 161),
    ),  # Dark gray -> Light gray
    "final.output.concentrate_au": (
        (200, 170, 0),
        (255, 235, 150),
    ),  # Dark gold -> Light gold
}

# For tail objects, we will change their scale instead of color.
tail_scale_params = {
    "final.output.tail_ag": (0.008, 0.008),
    "final.output.tail_pb": (0.008, 0.008),
    "final.output.tail_sol": (0.008, 0.008),
    "final.output.tail_au": (0.008, 0.008),
}
dashboard = lc.Dashboard(rows=14, columns=4, theme=lc.Themes.Light)

# Dictionaries to hold charts, textboxes, and model references.
conc_charts = {}
conc_textboxes = {}
tail_charts = {}
tail_textboxes = {}
conc_models = {}
tail_models = {}

# --- Create Concentrate (Rock) Charts and Text Boxes ---
for j, feat in enumerate(conc_features):
    # 3D Chart for concentrate object using rock_file
    chart = dashboard.Chart3D(row_index=0, column_index=j, row_span=4, column_span=1)
    chart.set_title(f"{display_names_conc[feat]} 3D")
    chart.get_default_x_axis().set_tick_strategy("Empty")
    chart.get_default_y_axis().set_tick_strategy("Empty")
    chart.get_default_z_axis().set_tick_strategy("Empty")
    conc_charts[feat] = chart

    # Load rock model and add to chart
    vertices, indices, normals = load_mesh_model(rock_file)
    if vertices and indices and normals:
        model = chart.add_mesh_model()
        model.set_model_geometry(vertices=vertices, indices=indices, normals=normals)
        model.set_scale(0.016).set_model_location(0, 0, 0)
        # Initially set to the start color of the range; will be updated in the loop.
        start_color = conc_color_ranges[feat][0]
        model.set_color(lc.Color(*start_color))
        conc_models[feat] = model
    else:
        conc_models[feat] = None

    # Text box for concentrate value
    txt_chart = dashboard.ChartXY(
        row_index=4, column_index=j, row_span=1, column_span=1
    )
    txt_chart.set_title("")
    txt_chart.get_default_x_axis().set_tick_strategy("Empty").set_interval(
        0, 1, stop_axis_after=True
    )
    txt_chart.get_default_y_axis().set_tick_strategy("Empty").set_interval(
        0, 1, stop_axis_after=True
    )
    tb = txt_chart.add_textbox(f"{display_names_conc[feat]} Value: ---", 0.5, 0.5)
    tb.set_text_font(20, weight="bold")
    tb.set_stroke(thickness=0, color=lc.Color(0, 0, 0, 0))
    conc_textboxes[feat] = tb

# --- Create Tail Charts and Text Boxes ---
for j, feat in enumerate(tail_features):
    # 3D Chart for tail object using tail_file
    chart = dashboard.Chart3D(row_index=5, column_index=j, row_span=4, column_span=1)
    chart.set_title(f"{display_names_tail[feat]} 3D")
    chart.get_default_x_axis().set_tick_strategy("Empty")
    chart.get_default_y_axis().set_tick_strategy("Empty")
    chart.get_default_z_axis().set_tick_strategy("Empty")
    # chart.set_camera_location(0, 1, 5)
    tail_charts[feat] = chart

    # Load tail model and add to chart
    vertices, indices, normals = load_mesh_model(tail_file)
    if vertices and indices and normals:
        model = chart.add_mesh_model()
        model.set_model_geometry(vertices=vertices, indices=indices, normals=normals)
        model.set_scale(0.01).set_model_location(0, 0, 0)
        # Set an initial color (can remain constant if desired)
        model.set_color(lc.Color("gray"))
        tail_models[feat] = model
    else:
        tail_models[feat] = None

    # Text box for tail value
    txt_chart = dashboard.ChartXY(
        row_index=9, column_index=j, row_span=1, column_span=1
    )
    txt_chart.set_title("")
    txt_chart.get_default_x_axis().set_tick_strategy("Empty").set_interval(
        0, 1, stop_axis_after=True
    )
    txt_chart.get_default_y_axis().set_tick_strategy("Empty").set_interval(
        0, 1, stop_axis_after=True
    )
    tb = txt_chart.add_textbox(f"{display_names_tail[feat]} Value: ---", 0.5, 0.5)
    tb.set_text_font(18, weight="bold")
    tb.set_stroke(thickness=0, color=lc.Color(0, 0, 0, 0))
    tail_textboxes[feat] = tb

# --- Create Multiple Y-Axis Chart for Recovery and Profit ---
# This chart spans row 10 with row span 3 and all 4 columns.
multi_chart = dashboard.ChartXY(row_index=10, column_index=0, row_span=4, column_span=4)
multi_chart.set_title("Recovery and Profit Trends")
multi_chart.get_default_x_axis().set_title("Trial Number")
# Use the default y-axis for final.output.recovery.
y_axis_recovery = multi_chart.get_default_y_axis().set_palette_line_coloring(
    steps=[{"value": 0, "color": lc.Color("cyan")}],
    look_up_property="y",
)

y_axis_recovery.set_title("Final Output Recovery (%)")
# Add a second y-axis for profit on the opposite side.
y_axis_profit = multi_chart.add_y_axis(opposite=True)
y_axis_profit.set_title("Profit (USD)").set_palette_line_coloring(
    steps=[{"value": 0, "color": lc.Color("red")}],
    look_up_property="y",
)

# Create series for recovery and profit.
recovery_series = multi_chart.add_point_line_series(y_axis=y_axis_recovery)
recovery_series.set_name("Final Output Recovery")
recovery_series.set_line_color(lc.Color("cyan"))
recovery_series.set_point_shape("Triangle").set_point_size(6)

profit_series = multi_chart.add_point_line_series(y_axis=y_axis_profit)
profit_series.set_name("Profit")
profit_series.set_line_color(lc.Color("red"))
profit_series.set_point_shape("Circle").set_point_size(6)

dashboard.open(live=True, method="browser")


num_rows = len(df)
i = 0
while True:
    # Update concentrate charts and text boxes
    for feat in conc_features:
        conc_value = df[feat].iloc[i]
        if pd.isna(conc_value):
            conc_value = min_conc[feat]
        conc_textboxes[feat].set_text(
            f"{display_names_conc[feat]} Value: {conc_value:.2f}"
        )
        if max_conc[feat] != min_conc[feat]:
            factor = (conc_value - min_conc[feat]) / (max_conc[feat] - min_conc[feat])
        else:
            factor = 0
        amplified_factor = min(max(factor * 3, 0), 1)
        start_color, end_color = conc_color_ranges[feat]
        new_color = lerp_color(start_color, end_color, amplified_factor)
        if conc_models[feat]:
            conc_models[feat].set_color(new_color)

    # Update tail charts and text boxes
    for feat in tail_features:
        tail_value = df[feat].iloc[i]
        tail_textboxes[feat].set_text(
            f"Value of {display_names_tail[feat]}: {tail_value:.2f}"
        )
        if max_tail[feat] != min_tail[feat]:
            factor_tail = (tail_value - min_tail[feat]) / (
                max_tail[feat] - min_tail[feat]
            )
        else:
            factor_tail = 0
        base_scale, scale_range = tail_scale_params[feat]
        new_scale = base_scale + scale_range * factor_tail
        if tail_models[feat]:
            tail_models[feat].set_scale(new_scale)

    # Update multiple y-axis chart series for Recovery and Profit.
    # Use the trial number (i) as the x-axis value.
    recovery_value = df["final.output.recovery"].iloc[i]
    profit_value = df["profit"].iloc[i]

    # Append new data point to the recovery series.
    recovery_series.add(x=[i], y=[recovery_value])
    profit_series.add(x=[i], y=[profit_value])

    print(f"Row {i} updated: Recovery={recovery_value:.2f}, Profit={profit_value:.2f}")
    i = (i + 1) % num_rows
    time.sleep(1)
