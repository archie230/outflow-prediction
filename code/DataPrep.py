import pandas as pd
import numpy as np
from datetime import datetime

class DataPrep:
    def __init__(self,
            transactions_df : pd.DataFrame,
            clients_df : pd.DataFrame,
            materials_df : pd.DataFrame,
            plants_df : pd.DataFrame,
            last_transaction_threshold : datetime):
        self.transactions_df = transactions_df
        self.clients_df = clients_df
        self.materials_df = materials_df
        self.plants_df = plants_df
        self.last_transaction_threshold = last_transaction_threshold

    def create_dataset(self):

