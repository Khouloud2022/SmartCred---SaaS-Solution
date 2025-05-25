import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import json
import re # For cleaning 'term' and 'emp_length'

# --- Configuration ---
# Path to your main source CSV file
SOURCE_CSV_PATH = Path("../ml_service/data/loans_full_schema.csv") # YOUR FILENAME

# Output directory for the new CSV files named after your Drizzle tables
OUTPUT_CSV_DIR = Path("output_drizzle_schema_csvs")
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

# Number of rows to process from the source CSV (for faster testing)
# Set to None to process all rows.
PROCESS_N_ROWS = 10000 # Example: process first 10k loans

# --- Helper Functions ---
def map_lc_loan_status_to_actual_outcome(status):
    if pd.isna(status): return None
    status_str = str(status).lower()
    bad_statuses = ['charged off', 'default', 'does not meet the credit policy. charged off', 'late (31-120 days)', 'late (16-30 days)']
    good_statuses = ['fully paid', 'does not meet the credit policy. fully paid']
    active_statuses = ['current', 'in grace period', 'issued']

    if any(st in status_str for st in bad_statuses): return 'DEFAULTED'
    elif any(st in status_str for st in good_statuses): return 'PAID_OFF'
    elif any(st in status_str for st in active_statuses): return 'ACTIVE'
    else: return None

def clean_lc_emp_length(s):
    if pd.isna(s) or str(s).lower() in ['n/a', '< 1 year']:
        return 0
    s_str = str(s).lower()
    if '10+ years' in s_str:
        return 10
    match = re.search(r'(\d+)', s_str)
    return int(match.group(1)) if match else 0

def clean_lc_term(s): # For " 36 months" type strings
    if pd.isna(s):
        return None
    match = re.search(r'(\d+)', str(s))
    return int(match.group(1)) if match else None

def get_random_date_iso(start_date_obj, end_date_obj):
    time_between_dates = end_date_obj - start_date_obj
    days_between_dates = time_between_dates.days
    if days_between_dates < 0: return start_date_obj.date().isoformat()
    random_number_of_days = np.random.randrange(days_between_dates + 1)
    random_date = start_date_obj + timedelta(days=random_number_of_days)
    return random_date.date().isoformat()

# --- Main Processing Logic ---
def main():
    print(f"Starting data processing. Outputting CSVs to: {OUTPUT_CSV_DIR}")

    # Initialize lists for each Drizzle table
    tenants_list = []
    users_list = []
    accounts_list = [] # For Plaid-like accounts
    borrowers_list = []
    csv_upload_jobs_list = []
    loan_applications_list = []
    credit_predictions_list = []
    payments_list = []

    # --- 1. Generate Sample Data for SaaS-Specific Tables ---
    now_dt = datetime.now()
    now_ts = now_dt.isoformat() # For timestamp fields
    today_date = now_dt.date().isoformat() # For date fields

    # Create Sample Tenants
    tenant1_uuid = str(uuid.uuid4())
    tenant2_uuid = str(uuid.uuid4())
    tenants_list.extend([
        {'id': tenant1_uuid, 'name': 'Alpha Finance Corp (Seeded)', 'createdAt': now_ts, 'updatedAt': now_ts},
        {'id': tenant2_uuid, 'name': 'Beta Credit Union (Seeded)', 'createdAt': now_ts, 'updatedAt': now_ts},
    ])
    print(f"Generated {len(tenants_list)} sample tenants.")

    # Create Sample Users (linked to Clerk and Tenants)
    user1_clerk_id = "clerk_user_alpha_manager_seeded"
    user2_clerk_id = "clerk_user_beta_analyst_seeded"
    users_list.extend([
        {'id': user1_clerk_id, 'tenantId': tenant1_uuid, 'email': 'manager.alpha@seeded.com', 'firstName': 'ManagerA', 'lastName': 'Alpha', 'role': 'manager', 'isActive': True, 'createdAt': now_ts, 'updatedAt': now_ts},
        {'id': user2_clerk_id, 'tenantId': tenant2_uuid, 'email': 'analyst.beta@seeded.com', 'firstName': 'AnalystB', 'lastName': 'Beta', 'role': 'analyst', 'isActive': True, 'createdAt': now_ts, 'updatedAt': now_ts},
    ])
    print(f"Generated {len(users_list)} sample users.")

    # Create Sample Accounts
    accounts_list.extend([
        {'id': str(uuid.uuid4()), 'plaidId': 'plaid_id_alpha_001', 'name': 'Alpha Corp Main Checking', 'userId': user1_clerk_id, 'createdAt': now_ts, 'updatedAt': now_ts},
        {'id': str(uuid.uuid4()), 'plaidId': 'plaid_id_beta_001', 'name': 'Beta CU Operations', 'userId': user2_clerk_id, 'createdAt': now_ts, 'updatedAt': now_ts},
    ])
    print(f"Generated {len(accounts_list)} sample accounts.")
    
    # Create Sample CSV Upload Job
    csv_job1_uuid = str(uuid.uuid4())
    csv_upload_jobs_list.append({
        'id': csv_job1_uuid, 'tenantId': tenant1_uuid, 'uploaderUserId': user1_clerk_id,
        'fileName': 'loans_full_schema_seed.csv', 'uploadTimestamp': now_ts, 'status': 'COMPLETED',
        'totalRows': PROCESS_N_ROWS if PROCESS_N_ROWS else 0, # Placeholder
        'processedRows': 0, # Placeholder
        'errorDetails': None, 'createdAt': now_ts, 'updatedAt': now_ts
    })
    print(f"Generated {len(csv_upload_jobs_list)} sample CSV upload job.")

    # --- 2. Process `loans_full_schema.csv` for `borrowers` and `loan_applications` ---
    print(f"Reading source loan CSV: {SOURCE_CSV_PATH}")
    try:
        # ASSUMPTION: loans_full_schema.csv HAS A HEADER ROW.
        # If not, you need to provide 'header=None' and 'names=COLUMN_NAMES_FROM_YOUR_ML_SCRIPT'
        df_source_loans = pd.read_csv(SOURCE_CSV_PATH, nrows=PROCESS_N_ROWS, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: Source CSV not found at {SOURCE_CSV_PATH}. Exiting.")
        return
    except Exception as e:
        print(f"ERROR reading source CSV: {e}. Check file format and header assumption.")
        return
        
    print(f"Loaded {len(df_source_loans)} rows from source loan CSV.")

    processed_loan_count = 0
    for index, source_row in df_source_loans.iterrows():
        # Assign to a tenant (e.g., round-robin or based on some logic)
        current_tenant_id_for_loan = tenant1_uuid if (index % 2 == 0) else tenant2_uuid

        # Create Borrower record for Drizzle 'borrowers' table
        borrower_uuid = str(uuid.uuid4())
        borrowers_list.append({
            'id': borrower_uuid,
            'tenantId': current_tenant_id_for_loan,
            'empTitle': str(source_row.get('emp_title', 'N/A'))[:255],
            'empLength': clean_lc_emp_length(source_row.get('emp_length')), # Use 'emp_length' from LC
            'state': str(source_row.get('addr_state', 'N/A'))[:255], # Use 'addr_state' from LC
            'homeownership': str(source_row.get('home_ownership', 'UNKNOWN'))[:255], # Use 'home_ownership' from LC
            'annualIncome': round(float(source_row.get('annual_inc', 0)), 2), # Use 'annual_inc' from LC
            'verifiedIncome': str(source_row.get('verification_status', 'UNKNOWN'))[:255], # Use 'verification_status'
            'debtToIncome': round(float(source_row.get('dti', 0)), 4), # Use 'dti' from LC
            'createdAt': now_ts, 'updatedAt': now_ts
        })

        # Create Loan Application record for Drizzle 'loanApplications' table
        loan_app_uuid = str(uuid.uuid4())
        actual_outcome_val = map_lc_loan_status_to_actual_outcome(source_row.get('loan_status'))
        
        if actual_outcome_val is None: # Skip loans without a clear final outcome for this seeding
            continue

        loan_term_val = clean_lc_term(source_row.get('term'))
        
        issue_date_val = None
        outcome_date_val = None
        try:
            if pd.notna(source_row.get('issue_d')):
                issue_dt_obj = pd.to_datetime(source_row.get('issue_d'))
                issue_date_val = issue_dt_obj.date().isoformat()
                if actual_outcome_val != 'ACTIVE':
                    months_to_add = loan_term_val if pd.notna(loan_term_val) else 36
                    outcome_dt_obj = issue_dt_obj + pd.DateOffset(months=int(months_to_add))
                    outcome_date_val = outcome_dt_obj.date().isoformat()
        except Exception: # Catch any date parsing errors
            pass

        loan_applications_list.append({
            'id': loan_app_uuid,
            'tenantId': current_tenant_id_for_loan,
            'borrowerId': borrower_uuid,
            'csvUploadJobId': csv_job1_uuid if current_tenant_id_for_loan == tenant1_uuid else None, # Example link
            'rawApplicationData': json.dumps(source_row.iloc[:min(30, len(source_row))].astype(str).to_dict()),
            'applicationIdentifierFromCsv': str(source_row.get('id', 'N/A'))[:255], # Original LC id
            'loanPurpose': str(source_row.get('purpose', 'N/A'))[:255],
            'applicationType': str(source_row.get('application_type', 'Individual'))[:255],
            'loanAmount': str(round(float(source_row.get('loan_amnt', 0)), 2)), # Decimals as string for Drizzle
            'term': loan_term_val,
            'interestRate': str(round(float(str(source_row.get('int_rate', '0%')).replace('%','')), 2)) if pd.notna(source_row.get('int_rate')) else '0.0',
            'installment': str(round(float(source_row.get('installment', 0)), 2)),
            'grade': str(source_row.get('grade', 'N/A'))[:255],
            'subGrade': str(source_row.get('sub_grade', 'N/A'))[:255],
            'issueDate': issue_date_val,
            'systemStatus': 'APPROVED' if actual_outcome_val != 'DEFAULTED' else 'REJECTED', # Simplified
            'actualOutcome': actual_outcome_val,
            'outcomeDate': outcome_date_val,
            'createdAt': now_ts, 'updatedAt': now_ts
        })
        processed_loan_count += 1

        # Create dummy Credit Prediction for Drizzle 'creditPredictions' table
        credit_predictions_list.append({
            'id': str(uuid.uuid4()),
            'loanApplicationId': loan_app_uuid,
            'tenantId': current_tenant_id_for_loan,
            'modelVersion': 'seed_model_v0.1',
            'predictedScore': str(round(np.random.uniform(0.01, 0.95), 4)), # Decimal as string
            'predictedRiskLabel': np.random.choice(['Low Risk', 'Medium Risk', 'High Risk'], p=[0.6, 0.3, 0.1]),
            'predictionTimestamp': now_ts,
            'inputFeaturesToModel': json.dumps({'dti': source_row.get('dti',0), 'annual_inc': source_row.get('annual_inc',0)}),
            'createdAt': now_ts
        })

        # Create simplified Payment for Drizzle 'payments' table
        if actual_outcome_val != 'ACTIVE' and pd.notna(source_row.get('last_pymnt_d')) and pd.notna(source_row.get('last_pymnt_amnt')):
            try:
                payment_dt_val = pd.to_datetime(source_row.get('last_pymnt_d')).date().isoformat()
                payments_list.append({
                    'id': str(uuid.uuid4()),
                    'loanApplicationId': loan_app_uuid,
                    'tenantId': current_tenant_id_for_loan,
                    'paymentDate': payment_dt_val,
                    'paymentAmount': str(round(float(source_row.get('last_pymnt_amnt',0)), 2)), # Decimal as string
                    'paymentStatus': 'PAID' if actual_outcome_val == 'PAID_OFF' else 'PARTIAL',
                    'createdAt': now_ts, 'updatedAt': now_ts
                })
            except Exception:
                pass
    
    print(f"Processed {processed_loan_count} loan applications for table generation.")

    if csv_upload_jobs_list: # Update the sample CSV job count
      if PROCESS_N_ROWS:
          csv_upload_jobs_list[0]['processedRows'] = processed_loan_count
          if len(df_source_loans) < PROCESS_N_ROWS:
             csv_upload_jobs_list[0]['totalRows'] = len(df_source_loans)
      else: # if processing all rows
          csv_upload_jobs_list[0]['totalRows'] = len(df_source_loans)
          csv_upload_jobs_list[0]['processedRows'] = processed_loan_count


    # --- 3. Convert lists of dicts to DataFrames and save to CSVs ---
    # Naming CSVs exactly like your Drizzle table names
    pd.DataFrame(tenants_list).to_csv(OUTPUT_CSV_DIR / 'tenants.csv', index=False)
    pd.DataFrame(users_list).to_csv(OUTPUT_CSV_DIR / 'users.csv', index=False)
    pd.DataFrame(accounts_list).to_csv(OUTPUT_CSV_DIR / 'accounts.csv', index=False)
    pd.DataFrame(borrowers_list).to_csv(OUTPUT_CSV_DIR / 'borrowers.csv', index=False)
    pd.DataFrame(csv_upload_jobs_list).to_csv(OUTPUT_CSV_DIR / 'csv_upload_jobs.csv', index=False)
    pd.DataFrame(loan_applications_list).to_csv(OUTPUT_CSV_DIR / 'loan_applications.csv', index=False)
    pd.DataFrame(credit_predictions_list).to_csv(OUTPUT_CSV_DIR / 'credit_predictions.csv', index=False)
    pd.DataFrame(payments_list).to_csv(OUTPUT_CSV_DIR / 'payments.csv', index=False)

    print(f"All table CSVs matching Drizzle schema saved to {OUTPUT_CSV_DIR}")

if __name__ == "__main__":
    main()