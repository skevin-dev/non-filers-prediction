import pandas as pd
from datetime import datetime

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure date columns are datetime
    if 'reg_date' in df.columns:
        df['reg_date'] = pd.to_datetime(df['reg_date'], errors='coerce')
    if 'close_date' in df.columns:
        df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')

    # --- Group A: Ratio Features ---
    df['purchase_per_supplier'] = df['ebm_tot_purchases'] / (df['suppliers'] + 1e-6)
    df['sales_per_client'] = df['ebm_tot_sales'] / (df['clients'] + 1e-6)
    df['purchase_to_sales_ratio'] = df['ebm_tot_purchases'] / (df['ebm_tot_sales'] + 1e-6)
    df['vat_efficiency_ratio'] = df['ebm_vat_on_sales'] / (df['ebm_vat_on_purchases'] + 1e-6)

    df['sales_per_invoice'] = df['ebm_tot_sales'] / (df['invoice_no_sales'] + 1e-6)
    df['purchases_per_invoice'] = df['ebm_tot_purchases'] / (df['invoice_no_purchases'] + 1e-6)
    df['client_supplier_ratio'] = df['clients'] / (df['suppliers'] + 1e-6)

    # --- Group B: Proportions ---
    df['prop_taxable_sales'] = df['ebm_taxable_sales'] / (df['ebm_tot_sales'] + 1e-6)
    df['prop_exempted_purchases'] = df['ebm_exempted_purchases'] / (df['ebm_tot_purchases'] + 1e-6)
    df['prop_taxable_purchase'] = df['ebm_taxable_purchases'] / (df['ebm_tot_purchases'] + 1e-6)

    # --- Group C: Date-based Features ---
    df['active_days'] = (df['close_date'] - df['reg_date']).dt.days
    df['reg_after_close_flag'] = (df['reg_date'] > df['close_date']).astype(int)
    df['declaration_since_closedate'] = df['year'] - df['close_date'].dt.year
    df['declaration_since_regdate'] = df['year'] - df['reg_date'].dt.year

    current_year = datetime.now().year
    df['is_long_registered'] = ((current_year - df['reg_date'].dt.year) > 5).astype(int)
    df['recently_closed'] = ((current_year - df['close_date'].dt.year) <= 1).astype(int)

    # --- Group D: Behavioral ---
    df['total_invoice_count'] = df['invoice_no_purchases'] + df['invoice_no_sales']
    df['has_import_or_export'] = ((df['import'] > 0) | (df['export'] > 0)).astype(int)
    df['high_purchase_low_sales'] = (
        (df['ebm_tot_purchases'] > df['ebm_tot_purchases'].quantile(0.75)) &
        (df['ebm_tot_sales'] < df['ebm_tot_sales'].quantile(0.25))
    ).astype(int)

    # --- Additional Financial Features ---
    df['gross_margin'] = df['ebm_tot_sales'] - df['ebm_tot_purchases']
    df['net_trade'] = df['export'] - df['import']
    df['high_sales_flag'] = (df['ebm_tot_sales'] > df['ebm_tot_sales'].quantile(0.9)).astype(int)
    df['invoices_with_zero_vat'] = (
        ((df['ebm_vat_on_sales'] == 0) & (df['invoice_no_sales'] > 0)) |
        ((df['ebm_vat_on_purchases'] == 0) & (df['invoice_no_purchases'] > 0))
    ).astype(int)

    # --- New Features (Group E): Difference and Flags ---
    df['diff_sales_purchase'] = df['ebm_tot_sales'] - df['ebm_tot_purchases']
    df['sales_gt_purchase_flag'] = (df['ebm_tot_sales'] > df['ebm_tot_purchases']).astype(int)

    df['diff_vat_sales_purchases'] = df['ebm_vat_on_sales'] - df['ebm_vat_on_purchases']
    df['vat_sales_gt_purchases_flag'] = (df['ebm_vat_on_sales'] > df['ebm_vat_on_purchases']).astype(int)

    df['diff_taxable_sales_total'] = df['ebm_taxable_sales'] - df['ebm_tot_sales']
    df['taxable_sales_gt_total_flag'] = (df['ebm_taxable_sales'] > df['ebm_tot_sales']).astype(int)

    df['diff_taxable_purchases_total'] = df['ebm_taxable_purchases'] - df['ebm_tot_purchases']
    df['taxable_purchases_gt_total_flag'] = (df['ebm_taxable_purchases'] > df['ebm_tot_purchases']).astype(int)

    df['diff_sales_inclusive_exclusive'] = df['ebm_tot_sales'] - df['ebm_total_sales_exclusive']
    df['inclusive_sales_gt_exclusive_flag'] = (df['ebm_tot_sales'] > df['ebm_total_sales_exclusive']).astype(int)

    df['diff_purchases_inclusive_exclusive'] = df['ebm_tot_purchases'] - df['ebm_total_purchases_exclusive']
    df['inclusive_purchases_gt_exclusive_flag'] = (df['ebm_tot_purchases'] > df['ebm_total_purchases_exclusive']).astype(int)

    # Check if tax_type_no_count > 1
    df['multiple_tax_types'] = (df['tax_type_no_count'] > 1).astype(int)

    # Check if district_no is in Kigali districts
    df['is_in_kigali'] = df['district_no'].isin([2, 6, 120]).astype(int)

    # --- Batch Add Final Features ---
    new_cols = pd.DataFrame({
        'prop_exempted_sales': df['ebm_exempted_sales'] / (df['ebm_tot_sales'] + 1e-6),
        'prop_zero_rated_sales': df['ebm_zero_rated_sales'] / (df['ebm_tot_sales'] + 1e-6),
        'prop_zero_rated_purchases': df['ebm_zero_rated_purchases'] / (df['ebm_tot_purchases'] + 1e-6),
    })

    # Add flags based on new proportions
    new_cols['high_exempted_sales_flag'] = (new_cols['prop_exempted_sales'] > 0.5).astype(int)
    new_cols['high_zero_rated_sales_flag'] = (new_cols['prop_zero_rated_sales'] > 0.5).astype(int)
    new_cols['high_zero_rated_purchases_flag'] = (new_cols['prop_zero_rated_purchases'] > 0.5).astype(int)

    # Merge all new columns
    df = pd.concat([df, new_cols], axis=1)

    return df
