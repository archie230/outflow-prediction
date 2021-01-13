import pandas as pd
import numpy as np
from datetime import datetime

class DataPrep:
    def __init__(self,
            purchase_df : pd.DataFrame,
            return_df : pd.DataFrame,
            clients_df : pd.DataFrame,
            materials_df : pd.DataFrame,
            plants_df : pd.DataFrame,
            y_df : pd.DataFrame,
            last_transaction_threshold : datetime):
        
        self.purchase_df = purchase_df
        self.return_df = return_df
        self.clients_df = clients_df
        self.materials_df = materials_df
        self.plants_df = plants_df
        self.y_df = y_df
        self.last_transaction_threshold = last_transaction_threshold

    def create_dataset(self):
        print('Creating client transactions df...')
        client_transactions_dataset_df = self._create_client_transactions_dataset()
        print('Created!')
        
        print('Creating client info df...')
        client_info_dataset_df = self._create_client_info_dataset()
        print('Created!')
        
        print('Merging...')
        dataset_df = client_transactions_dataset_df\
                    .merge(client_info_dataset_df, on='client_id')\
                    .merge(self.y_df, on='client_id')
        print('Merged!')
        
        return dataset_df
    
    def _create_client_info_dataset(self):
        first_transaction_date = self.purchase_df['chq_date'].min()
        months_num = (self.last_transaction_threshold.year - first_transaction_date.year) * 12 \
            + self.last_transaction_threshold.month - first_transaction_date.month
        print('\tData available for {} months'.format(months_num))
        
        print('\tCreating client_purchase_dates_df...')
        client_purchase_dates_df = self.purchase_df.drop_duplicates(subset='chq_id', ignore_index=True)\
                        [['chq_date', 'client_id']]\
                        .groupby('client_id', as_index=False)\
                        .agg(lambda x : sorted(x.tolist()))
        client_purchase_dates_df = client_purchase_dates_df.rename(columns={'chq_date' : 'purch_dates'})
        client_purchase_dates_df = client_purchase_dates_df.set_index('client_id')
        print('\tDone!')
        
        print('\tCreating client_return_dates_df...')
        client_return_dates_df = self.return_df.drop_duplicates(subset='chq_id', ignore_index=True)\
                        [['chq_date', 'client_id']]\
                        .groupby('client_id', as_index=False)\
                        .agg(lambda x : sorted(x.tolist()))
        client_return_dates_df = client_return_dates_df.rename(columns={'chq_date' : 'return_dates'})
        client_return_dates_df = client_return_dates_df.set_index('client_id')
        print('\tDone!')
        
        print('\tCreating clients info df...')
        df = self.clients_df[['client_id', 'gender', 'city', 'birthyear']]
        df = df.set_index('client_id')
        
        df['days_since_last_purchase'] = (self.last_transaction_threshold - first_transaction_date).days
        df.loc[client_purchase_dates_df.index, 'days_since_last_purchase'] = client_purchase_dates_df['purch_dates']\
                                        .apply(lambda x : (self.last_transaction_threshold - x[-1]).days)
        df['purch_num'] = 0
        df.loc[client_purchase_dates_df.index, 'purch_num'] = client_purchase_dates_df['purch_dates']\
                                                            .apply(lambda x : len(x) / months_num) 
        df['return_num'] = 0
        df.loc[client_return_dates_df.index, 'return_num'] = client_return_dates_df['return_dates']\
                                                            .apply(lambda x : len(x) / months_num)
        print('\tDone!')
        
        df = df.reset_index()
        
        return df
        
        
    def _create_client_transactions_dataset(self):
        # purchase processing
        print('\tMerging purchase with materials...')
        df = self.purchase_df\
            .merge(self.materials_df, on='material')
        df = df.drop(labels=['chq_position', 'material', 'sales_count'], axis=1)
        print('\tDone!')
        print('\tMerging purchase with plants...')
        df = df.merge(self.plants_df, on='plant')
        df = pd.concat([df,
                        pd.get_dummies(df['hier_level_1'], prefix='product')],
                        axis=1)
        df = df.drop(labels=['hier_level_1'], axis=1)
        print('\tDone!')
        
        print('\tAggregating purchase by checks...')
        chq_agg_func = {
            'client_id' : lambda x : x.iloc[0],
            'plant_type' : lambda x : x.iloc[0],
            'plant' : lambda x : x.iloc[0],

            'sales_sum' : sum,
            'is_promo' : sum,
            'is_private_label' : sum,
            'is_alco' : sum,
            'product_FOOD' : sum,
            'product_NONFOOD' : sum
        }
        df = df.groupby('chq_id', as_index=False).agg(chq_agg_func)
        print('\tDone!')
        
        print('\tAggregating purchase by clients...')
        client_agg_func = {
            'plant_type' : lambda x : x.value_counts().idxmax(),
            'plant' : lambda x : len(x.unique()),

            'sales_sum' : lambda x : x.mean(),
            'is_promo' : lambda x : x.mean(),
            'is_private_label' : lambda x : x.mean(),
            'is_alco' : lambda x : x.mean(),
            'product_FOOD' : lambda x : x.mean(),
            'product_NONFOOD' : lambda x : x.mean()
        }
        df = df.groupby('client_id', as_index=False).agg(client_agg_func)
        print('\tDone!')
            
        # return processing
        print('\tMerging returns with materials...')
        df1 = self.return_df\
            .merge(self.materials_df, on='material')
        print('\tDone!')
        
        print('\tAggregating returns by checks...')
        df1 = df1[['chq_id', 'client_id', 'sales_sum']]\
            .groupby('chq_id', as_index=False)\
            .agg({
            'client_id' : lambda x : x.iloc[0],
            'sales_sum' : sum
        })
        print('\tDone!')
        
        print('\tAggregating returns by clients...')
        df1.groupby('client_id', as_index=False)\
            .agg({
            'sales_sum' : lambda x : x.mean()
        })
        print('\tDone!')
        
        df1 = df1.rename(columns={'sales_sum' : 'return_sum'})
        
        df = df.rename(columns={
            'plant_type' : 'most_freq_plant_type',
            'plant' : 'diff_plants_amount',
            'sales_sum' : 'check_sum',
            'is_promo' : 'promo_amount',
            'is_private_label' : 'priv_lbl_amount',
            'is_alco' : 'alco_amount',
            'product_FOOD' : 'food_amount',
            'product_NONFOOD' : 'nonfood_amount'
        })
        
        print('\tAdding return_sum to df...')
        df['return_sum'] = 0
        df = df.set_index('client_id')
        df1 = df1.set_index('client_id')
        df.loc[df1.index, 'return_sum'] = df1['return_sum']
        print('\tDone!')
        
        df = df.reset_index()
        
        return df