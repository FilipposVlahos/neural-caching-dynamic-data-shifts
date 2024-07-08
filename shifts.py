import numpy as np
import pandas as pd

def label_shift(df, shift_order):
    return apply_shift_order(df, shift_order)

def label_shift_partial(df, perc_rand, shift_order, seed):
    df_sorted = apply_shift_order(df, shift_order)

    # Determine the percentage of rows to be randomized
    num_randomized = int(len(df) * perc_rand)

    # Select random rows to shuffle
    np.random.seed(seed)
    random_indices = np.random.choice(df_sorted.index, size=num_randomized, replace=False)
    random_rows = df_sorted.loc[random_indices]

    # Shuffle the selected rows
    random_rows_shuffled = random_rows.sample(frac=1).reset_index(drop=True)

    # Replace the selected rows in the sorted DataFrame with the shuffled rows
    for i, idx in enumerate(random_indices):
        df_sorted.loc[idx] = random_rows_shuffled.loc[i]

    return df_sorted

def covariate_shift_typos(df, perc, typos_data_path, sort, seed):
    df_typos = pd.read_csv(typos_data_path)

    num_rows_to_replace = int(len(df) * perc)

    np.random.seed(seed)
    print('original length: ', len(df))
    print('covariate length:', len(df_typos))
    # Replace selected inputs with typos
    indices_to_replace = np.random.choice(df.index, size=num_rows_to_replace, replace=False)
    df.loc[indices_to_replace] = df_typos.loc[indices_to_replace].values

    if sort != 'none':
        # Sort so that input with typos appear first
        was_replaced = df.index.isin(indices_to_replace)
        df = df.iloc[was_replaced.argsort()[::-1]].reset_index(drop=True)

    return df


def apply_shift_order(df, shift_order):
    if shift_order=='ascending' or shift_order == 'descending':
        is_ascending = shift_order=='ascending'
        sort_by = 'gold_hard'  # 'llm_hard' 
        return df.sort_values(by=sort_by, ascending=is_ascending).reset_index(drop=True)
    if shift_order == 'label-agreement':
        df['agreement'] = df['gold_hard'].str.strip() == df['llm_hard'].str.strip()

        agreement_counts = df.groupby('gold_hard')['agreement'].sum().reset_index()
        agreement_counts.columns = ['class', 'agreement']

        # Step 2: Sort the classes based on their agreement levels
        agreement_counts = agreement_counts.sort_values(by='agreement', ascending=False)

        # Step 3: Create a mapping from class to sorted order
        class_order = {cls: i for i, cls in enumerate(agreement_counts['class'])}

        # Step 4: Sort the DataFrame based on the class order
        df['class_order'] = df['gold_hard'].map(class_order)
        df = df.sort_values(by=['class_order', 'gold_hard'])

        # Drop the temporary columns
        df = df.drop(columns=['agreement', 'class_order'])
        return df