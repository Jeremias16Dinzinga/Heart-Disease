import matplotlib.pyplot as plt
import pandas as pd

## Configurations
dataset_filename = "./data/cleveland.csv"
num_outputs = 1
variables = None # Histogram variables to show. None to show all
max_rows = 2
max_cols = 2
bins = 30

dataset = pd.read_csv(dataset_filename)

columns = dataset.columns

def num_groups(quantity, quantity_per_group):
    num_groups = quantity // quantity_per_group
    if quantity % quantity_per_group > 0:
        num_groups += 1

    return num_groups

def show_histograms(variables):
    num_variables = len(variables)
    rows = num_groups(num_variables, max_cols)
    columns = min(num_variables, max_cols)
    rows = min(rows, max_rows)

    max_graphics_plot = rows * columns

    if num_variables > max_graphics_plot:
        number_pages = num_groups(num_variables, max_graphics_plot)

        for i in range(0, number_pages):
            variables_to_show = variables[0:max_graphics_plot]
            variables = variables[max_graphics_plot:]
            show_histograms_subplots(variables_to_show, rows, columns)
    else:
        show_histograms_subplots(variables, rows, columns)

def show_histograms_subplots(variables, rows, columns):
    fig, axes = plt.subplots(rows, columns)
    axes = axes.flatten() # Flatten in case of a single row or column

    for i, col in enumerate(variables):
        axes[i].hist(dataset[col], bins=bins, edgecolor='k', alpha=0.7)
        axes[i].set_title(f"{col} histogram")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        plt.grid(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plt.show()

num_inputs = len(columns) - num_outputs
input_columns = columns[0:num_inputs]

if variables is None:
    num_inputs = len(columns) - num_outputs
    show_histograms(columns[0:num_inputs])
else:
    show_histograms(variables)
