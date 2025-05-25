# ml_service/src/training/build_dataset.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import config # Import your config file

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Manually Defined Column Names (as you provided)
COLUMN_NAMES = [
    "id", "member_id", "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate", "installment", "grade", 
    "sub_grade", "emp_title", "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d", 
    "loan_status", "pymnt_plan", "url", "desc", "purpose", "title", "zip_code", "addr_state", "dti", 
    "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths", 
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util", 
    "total_acc", "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", 
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", 
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high", 
    "last_fico_range_low", "collections_12_mths_ex_med", "mths_since_last_major_derog", "policy_code", 
    "application_type", "annual_inc_joint", "dti_joint", "verification_status_joint", "acc_now_delinq", 
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", 
    "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", 
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m", "acc_open_past_24mths", 
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt", 
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", 
    "mort_acc", "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq", 
    "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", 
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", 
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", 
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim", 
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit", "revol_bal_joint", 
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", 
    "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il", 
    "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med", 
    "sec_app_mths_since_last_major_derog", "hardship_flag", "hardship_type", "hardship_reason", 
    "hardship_status", "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date", 
    "payment_plan_start_date", "hardship_length", "hardship_dpd", "hardship_loan_status", 
    "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount", 
    "hardship_last_payment_amount", "disbursement_method", "debt_settlement_flag", 
    "debt_settlement_flag_date", "settlement_status", "settlement_date", "settlement_amount", 
    "settlement_percentage", "settlement_term"
]

def preprocess_lending_club_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all preprocessing steps to a chunk of the LendingClub dataset."""
    
    # === Step 1: Define and Map Target Variable ===
    def map_loan_status(status):
        if pd.isna(status): return np.nan
        status_str = str(status).lower()
        bad_statuses = ['charged off', 'default', 'does not meet the credit policy. charged off']
        good_statuses = ['fully paid', 'does not meet the credit policy. fully paid']
        if any(st in status_str for st in bad_statuses): return 1
        elif any(st in status_str for st in good_statuses): return 0
        else: return np.nan # For 'Current', 'In Grace Period', etc.
        
    if 'loan_status' in df.columns:
        df['target'] = df['loan_status'].apply(map_loan_status)
        df.dropna(subset=['target'], inplace=True)
        if df.empty: return pd.DataFrame()
        df['target'] = df['target'].astype(int)
    else:
        logging.warning("'loan_status' column not found in chunk. Cannot create target variable.")
        return pd.DataFrame()

    # === Step 2: Drop Unnecessary, Leaky, and High-Missing/Problematic Columns ===
    cols_to_drop = [
        'member_id', 'desc', 'mths_since_last_record', 'mths_since_last_major_derog', 
        'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'revol_bal_joint', 
        # All sec_app columns (secondary applicant)
        'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line', 
        'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util', 
        'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths', 
        'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog',
        # All hardship columns (often sparse and can be leaky if not handled carefully)
        'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status', 'deferral_term', 'hardship_amount', 
        'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date', 'hardship_length', 
        'hardship_dpd', 'hardship_loan_status', 'orig_projected_additional_accrued_interest', 
        'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
        # All settlement columns (post-default, leaky)
        'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status', 'settlement_date', 
        'settlement_amount', 'settlement_percentage', 'settlement_term',
        # Leaky columns related to payment history
        'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 
        'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'next_pymnt_d',
        # IDs and free text / high cardinality columns
        'id', 'url', 'emp_title', 'zip_code', 'title', # <<< ADDED 'title' HERE
        # Redundant or less useful
        'policy_code', 'funded_amnt', 'funded_amnt_inv', 'sub_grade',
        # Original target
        'loan_status' 
    ]
    # Ensure uniqueness in cols_to_drop and drop them
    df.drop(columns=list(set(cols_to_drop)), inplace=True, errors='ignore')

    # === Step 3: Clean and Engineer Remaining Features ===
    if 'emp_length' in df.columns:
        df['emp_length'].fillna('0 years', inplace=True)
        df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
    
    if 'term' in df.columns: # Convert term to numeric
        # Assuming term is like " 36 months"
        df['term'] = df['term'].str.strip().str.replace(' months', '').astype(float)
    
    if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
        df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
        valid_dates = df['issue_d'].notna() & df['earliest_cr_line'].notna()
        df.loc[valid_dates, 'credit_history_years'] = (df.loc[valid_dates, 'issue_d'] - df.loc[valid_dates, 'earliest_cr_line']).dt.days / 365.25
        df.drop(columns=['earliest_cr_line', 'issue_d'], inplace=True, errors='ignore')

    # === Step 4: Convert to Numeric and Handle Missing Values for Numerical Columns ===
    potential_numeric_cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs", 
        "fico_range_low", "fico_range_high", "inq_last_6mths", "mths_since_last_delinq",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", 
        "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
        "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", 
        "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
        "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
        "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq",
        "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl",
        "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
        "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m",
        "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
        "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
        "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit", # total_bal_il should be here
        "emp_length", "term", "credit_history_years" 
    ]
    # Add total_bal_il to the list if it wasn't mistyped and is intended to be numeric
    if "total_bal_il" not in potential_numeric_cols: # Safety check if you added it elsewhere
        potential_numeric_cols.append("total_bal_il")


    for col in potential_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 
            if df[col].isnull().any(): 
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
    
    # Ensure all remaining object columns are string type, fill their NaNs
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).fillna('Missing')
            
    return df

if __name__ == '__main__':
    chunk_size = 100000 
    processed_chunks = []

    logging.info(f"Starting to process {config.RAW_DATA_PATH} in chunks of {chunk_size}...")
    
    try:
        with pd.read_csv(
            config.RAW_DATA_PATH, 
            chunksize=chunk_size, 
            low_memory=False, 
            skiprows=1, # Assuming first row is disclaimer, second is actual header
            header=None, # We provide names
            names=COLUMN_NAMES,
            # Consider na_filter=False if string 'NA' or 'NaN' should be literals
        ) as reader:
            for i, chunk in enumerate(reader):
                logging.info(f"Processing chunk {i+1}...")
                processed_chunk = preprocess_lending_club_chunk(chunk.copy())
                if not processed_chunk.empty:
                    processed_chunks.append(processed_chunk)
    except FileNotFoundError:
        logging.error(f"Raw data file not found at {config.RAW_DATA_PATH}. Cannot proceed.")
        raise
    except Exception as e:
        logging.error(f"Error during chunk processing: {e}")
        raise

    if not processed_chunks:
        logging.error("No data was processed. Output Parquet file will not be created.")
    else:
        logging.info("Concatenating all processed chunks...")
        final_df = pd.concat(processed_chunks, ignore_index=True)
        
        logging.info("Intelligently handling remaining categorical features (reducing cardinality and one-hot encoding)...")
        # Ensure target is not in this list
        categorical_cols_to_encode = final_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'target' in categorical_cols_to_encode:
            # This should not happen if target is correctly converted to int
            logging.warning("Target column found in categorical columns to encode. Removing it.")
            categorical_cols_to_encode.remove('target')


        for col in categorical_cols_to_encode:
            num_unique = final_df[col].nunique()
            logging.info(f"Column '{col}' has {num_unique} unique values.")
            
            # Adjust cardinality threshold if needed, e.g., 50 or 200
            # Or decide to drop columns like 'addr_state' if it has too many and isn't critical
            if num_unique > 100: 
                logging.info(f"  -> High cardinality found. Grouping rare categories for '{col}'.")
                value_counts = final_df[col].value_counts()
                rare_threshold_count = max(10, int(len(final_df) * 0.0005)) 
                rare_categories = value_counts[value_counts < rare_threshold_count].index
                
                if len(rare_categories) > 0 and col not in ['target']: # defensive check for target
                    final_df[col] = final_df[col].replace(rare_categories, 'Other')
                    logging.info(f"  -> Reduced to {final_df[col].nunique()} unique values for one-hot encoding.")

        logging.info(f"Applying one-hot encoding to: {categorical_cols_to_encode}")
        if categorical_cols_to_encode:
            final_df_encoded = pd.get_dummies(final_df, columns=categorical_cols_to_encode, drop_first=True, dtype=float)
        else:
            final_df_encoded = final_df.copy() # Ensure it's a copy if no encoding happens

        logging.info(f"Saving final cleaned dataset with shape {final_df_encoded.shape}...")
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_df_encoded.to_parquet(config.PROCESSED_DATA_PATH)

        logging.info(f"Process complete. Cleaned data saved to {config.PROCESSED_DATA_PATH}")