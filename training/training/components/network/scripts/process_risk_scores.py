import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_risk_scores(file_path, moving_average_window=50, stats_output_file="outputs/risk_score_stats.csv", ma_output_file="outputs/risk_score_moving_avg.csv", graph_output="outputs/risk_scores_graph.png"):
    """
    Process the risk_scores.csv file to compute overall statistics, moving averages, and generate a graph.
    
    :param file_path: Path to the risk_scores.csv file.
    :param moving_average_window: Window size for moving averages.
    :param stats_output_file: Name of the output file for overall statistics.
    :param ma_output_file: Name of the output file for moving averages.
    :param graph_output: Name of the output graph image file.
    """
    print(f"Loading risk scores from {file_path}...")
    data = pd.read_csv(file_path)
    
    if "Risk Score" not in data.columns:
        raise ValueError("The file does not contain a 'Risk Score' column.")
    
    # Overall Statistics
    print("Computing overall statistics...")
    stats = {
        "Mean Risk Score": [data["Risk Score"].mean()],
        "Median Risk Score": [data["Risk Score"].median()],
        "Standard Deviation": [data["Risk Score"].std()],
        "Variance": [data["Risk Score"].var()],
        "Max Risk Score": [data["Risk Score"].max()],
        "Min Risk Score": [data["Risk Score"].min()],
        "Total Records": [len(data["Risk Score"])]
    }
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(stats_output_file, index=False)
    print(f"Overall statistics saved to {stats_output_file}.")

    # Moving Averages
    print("Calculating moving averages...")
    data["Moving Average"] = data["Risk Score"].rolling(window=moving_average_window).mean()
    moving_avg_df = data[["Moving Average"]].dropna().reset_index(drop=True)
    moving_avg_df.to_csv(ma_output_file, index=False)
    print(f"Moving averages saved to {ma_output_file}.")

    # Generate Graph
    print("Generating graph for risk scores...")
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Risk Score"], label="Risk Score", alpha=0.7)
    plt.title("Risk Scores")
    plt.xlabel("Record Index")
    plt.ylabel("Risk Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(graph_output, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {graph_output}.")

    print("Processing complete!")

if __name__ == "__main__":
    # Path to the risk_scores.csv file
    risk_scores_file = "outputs/risk_scores.csv"
    # Process and analyze the risk scores
    process_risk_scores(
        file_path=risk_scores_file,
        moving_average_window=50,
        stats_output_file="outputs/risk_score_stats.csv",
        ma_output_file="outputs/risk_score_moving_avg.csv",
        graph_output="outputs/risk_scores_graph.png"
    )