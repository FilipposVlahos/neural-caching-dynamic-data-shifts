import numpy as np
import pandas as pd

def label_shift(df, shift_order, seed):
    return apply_shift_order(df, shift_order, seed)

def label_shift_partial(df, perc_rand, shift_order, seed):
    '''
    Apply label shift and then randomly shuffle a percentage of the rows 
    in the DataFrame.
    '''
    df_sorted = apply_shift_order(df, shift_order, seed)

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

def covariate(df, shift_order, perc, typos_data_path, sort, seed):
    '''
    Appply a covariate shift to the dataframe i.e. a shift that only affects the 
    input distribution p(x).
    '''
    if shift_order == 'cov-typos':
        return covariate_shift_typos(df,  perc, typos_data_path, sort, seed)
    if shift_order == 'cov-sentence-length-asc':
        return covariate_sentence_length(df)
    if shift_order == 'cov-sentence-length-desc':
        return covariate_sentence_length(df).sort_index(ascending=False).reset_index(drop=True)

def covariate_sentence_length(df):
    '''
    Sort the dataframe based on the length of the input sentences.
    '''
    # Apply the function to create a new column with word counts
    df['word_count'] = df['input'].apply(lambda text: len(text.split()))

    df_sorted = df.sort_values(by='word_count')

    df_sorted = df_sorted.drop(columns=['word_count'])
    return df_sorted

def covariate_shift_typos(df, perc, typos_data_path, sort, seed):
    '''
    Replace a percentag of examples of the dataframe with the same examples
    from a typos dataset.
    Args:
        perc: percentage of examples with typos
        typos_data_path: path to the typos dataset
        sort: 'none' for the typos to appear randomly, otherwise typos appear first
        seed: random seed
    '''
    
    df_typos = pd.read_csv(typos_data_path)
    if perc == 1:
        return df_typos
    num_rows_to_replace = int(len(df) * perc)

    np.random.seed(seed)
    # Replace selected inputs with typos
    indices_to_replace = np.random.choice(df.index, size=num_rows_to_replace, replace=False)
    df.loc[indices_to_replace] = df_typos.loc[indices_to_replace].values

    if sort != 'none':
        # Sort so that input with typos appear first
        was_replaced = df.index.isin(indices_to_replace)
        df = df.iloc[was_replaced.argsort()].reset_index(drop=True)
    return df


def apply_shift_order(df, shift_order, seed):
    if shift_order=='ascending' or shift_order == 'descending':
        is_ascending = shift_order=='ascending'
        sort_by = 'gold_hard'  # 'llm_hard' 
        return df.sort_values(by=sort_by, ascending=is_ascending).reset_index(drop=True)
    if shift_order == 'label-agreement':
        return label_agreement(df)
    if shift_order == 'label-dissagreement':
        return label_agreement(df).sort_index(ascending=False).reset_index(drop=True)
    if shift_order =='random':
        # Sort data-points based on labels randomly
        sort_by = 'gold_hard'  # 'llm_hard' 
        unique_values = df[sort_by].unique()
        
        # Shuffle the unique values to get a random order
        np.random.seed(seed)
        np.random.shuffle(unique_values)
        
        # Create a categorical type with the random order
        cat_type = pd.CategoricalDtype(categories=unique_values, ordered=True)
        # Convert the column to the categorical type
        df[sort_by] = df[sort_by].astype(cat_type)

        sorted_df = df.sort_values(by=sort_by)
        return sorted_df
    return df

def label_agreement(df):
    '''
    Sorts the dataframe based on the accuracy of LLM predictions for each class in 'gold_hard'.
    The rows with the 'gold_hard' label that the LLM predicts with higher accuracy appear first.
    Args: 
        df (pd.DataFrame): The input dataframe with 'gold_hard' and 'llm_hard' columns.
    Returns:
        pd.DataFrame: The sorted dataframe.
    '''
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