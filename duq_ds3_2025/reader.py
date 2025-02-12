import json
import pandas as pd
import matplotlib.pyplot as plt

class Reader:
    def __init__(self, path: str):
        """Read and parse JSON data file.

        Args:
            path (str): location of the JSON file
        """
        self.path = path
        self.data = self._load_json()
        self.df = self._json_to_dataframe()
        self._clean_data()

    def _load_json(self):
        """Load JSON file into a dictionary."""
        with open(self.path, 'r') as f:
            return json.load(f)

    def _json_to_dataframe(self):
        """Convert the nested JSON structure into a Pandas DataFrame."""
        records = []
        for user_id, entries in self.data.items():
            for day, values in entries.items():
                records.append({
                    "user_id": user_id,
                    "day": float(day),
                    "path": values.get("path", []),
                    "probability": values.get("prob", None)
                })
        return pd.DataFrame(records)

    def _clean_data(self):
        """Clean the dataset by handling missing values and ensuring proper types."""
        self.df["probability"] = pd.to_numeric(self.df["probability"], errors="coerce")
        self.df["path_length"] = self.df["path"].apply(len)

    def get_user_data(self, user_id: str):
        """Retrieve all data for a specific user."""
        return self.df[self.df["user_id"] == user_id]

    def filter_by_probability(self, min_prob: float = None, max_prob: float = None):
        """Filter records by probability range."""
        df_filtered = self.df
        if min_prob is not None:
            df_filtered = df_filtered[df_filtered["probability"] >= min_prob]
        if max_prob is not None:
            df_filtered = df_filtered[df_filtered["probability"] <= max_prob]
        return df_filtered

    def summarize(self):
        """Generate basic summary statistics."""
        summary = {
            "Total Users": self.df["user_id"].nunique(),
            "Total Observations": len(self.df),
            "Mean Probability": self.df["probability"].mean(),
            "Standard Deviation Probability": self.df["probability"].std(),
            "Max Path Length": self.df["path_length"].max(),
            "Min Path Length": self.df["path_length"].min(),
        }
        return pd.DataFrame(summary.items(), columns=["Metric", "Value"])

    def plot_probability_distribution(self):
        """Plot histogram of probability values."""
        self.df["probability"].hist(bins=50)
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.title("Probability Distribution")
        plt.show()

    def plot_path_length_distribution(self):
        """Plot histogram of path lengths."""
        self.df["path_length"].hist(bins=30)
        plt.xlabel("Path Length")
        plt.ylabel("Frequency")
        plt.title("Path Length Distribution")
        plt.show()

    def save_to_csv(self, filename="processed_data.csv"):
        """Save processed DataFrame to CSV."""
        self.df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_to_json(self, filename="processed_data.json"):
        """Save processed DataFrame to JSON."""
        self.df.to_json(filename, orient="records")
        print(f"Data saved to {filename}")

    def display(self):
        """Show a sample of the dataset."""
        print(self.df.head())

if __name__ == '__main__':
    reader = Reader('data\full_dataset_48days_viterbi.json')
    reader.display()
    reader.summarize()
    reader.plot_probability_distribution()
    reader.plot_path_length_distribution()
    reader.save_to_csv()
