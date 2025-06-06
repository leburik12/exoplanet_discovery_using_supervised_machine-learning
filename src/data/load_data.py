import pandas as pd
import os
from typing import Tuple, Optional


class ExoplanetDataLoader:
    def __init__(self, csv_path: str, verbose: bool = True):
        self.csv_path = csv_path
        self.verbose = verbose
        self.df: Optional[pd.DataFrame] = None

    def validate_path(self) -> None:
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.csv_path.endswith(".csv"):
            raise ValueError("Provided file must be a .csv file")
        if self.verbose:
            print(f"File path validated: {self.csv_path}")

    def load(self, skip_comments: bool = True) -> pd.DataFrame:
        self.validate_path()

        comment_char = "#" if skip_comments else None
        self.df = pd.read_csv(self.csv_path, comment=comment_char)

        if self.verbose:
            print(f"📊 Loaded dataframe with shape: {self.df.shape}")
            print(f"🧠 Columns loaded: {len(self.df.columns)}")

        return self.df

    def summarize(self) -> None:
        """Prints a summary of the dataset including NA values and basic statistics."""
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call `.load()` first.")

        print("\n Data Summary:")
        print(self.df.info())
        print("\n Missing Values Per Column:")
        print(self.df.isnull().sum())
        print("\n Basis Statistics:")
        print(self.df.describe(include="all").T)

    def get_feature_target_split(
        self, target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Splits the dataset into features and target.

        Parameters:
        - target_column (str): Column to be used as the target variable.

        Returns:
        - Tuple of features DataFrame and target Series
        """

        if self.df is None:
            raise ValueError("Dataframe not loaded. Call `.load()` first.")

        if target_column not in self.df.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset.")

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        if self.verbose:
            print(f"Feature matrix shape: {X.shape}")
            print(f"Target vector shape: {y.shape}")

        return X, y


# if __name__ == "__main__":
#     csv_path = "./data/exoplanets.csv"  # Path to your dataset
#     loader = ExoplanetDataLoader(csv_path)

#     df = loader.load()
#     loader.summarize()

