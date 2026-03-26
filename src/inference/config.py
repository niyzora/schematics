"""
Feature engineering configuration.
All mappings are aligned with the training notebook.
"""

FEATURE_COLS = [
    "position", "position_inv", "is_position_1",
    "brand_ctr", "brand_ev", "brand_pos1_ctr",
    "brand_credit_ctr", "brand_credit_lift",
    "credit_score_ord", "annual_income_ord", "annual_income_missing",
    "credit_line_ord", "credit_line_missing", "property_value_ord",
    "mortgage_ord", "military_missing", "is_military",
    "traffic_ctr", "traffic_ev",
    "loan_purpose_Home Improvements", "loan_purpose_Investment Opportunities",
    "loan_purpose_Other", "loan_purpose_Retirement Income",
    "device_mobile", "device_tablet",
    "property_type_Multi-Family Home", "property_type_Single-Family Home",
    "property_type_Townhome / Condominium",
    "property_use_Rental Property", "property_use_Secondary/Vacation Home",
]

# --- Ordinal encodings ---

ANNUAL_INCOME_MAP = {
    "Less than $50K": 1,
    "$50K–$75K": 2,
    "$75K–$100K": 3,
    "Over $100K": 4,
}

CREDIT_SCORE_MAP = {
    "Very Poor": 1,
    "Poor": 2,
    "Fair": 3,
    "Good": 4,
    "Excellent": 5,
}

CREDIT_LINE_MAP = {
    "Under $50,000": 1,
    "$50,000-$100,000": 2,
    "$100,000-$200,000": 3,
    "$200,000-$300,000": 4,
    "$300,000-$500,000": 4,
    "Maximum Eligible": 6,
}

PROPERTY_VALUE_MAP = {
    "Under $150,000": 1,
    "Under $200,000": 1,
    "$150,000 - $300,000": 2,
    "$200,000 - $400,000": 2,
    "$300,000 - $500,000": 3,
    "$400,000 - $600,000": 3,
    "$600,000 - $800,000": 7,
    "$500,000 - $1M": 7,
    "Over $800,000": 8,
    "Over $1M": 8,
}

MORTGAGE_MAP = {
    "No": 1,
    "Under $50K": 2,
    "Under $100K": 2,
    "$50K-$150K": 2,
    "$100K-$250K": 3,
    "$150K+": 3,
    "$250K-$400K": 4,
    "$400K+": 5,
}

# --- One-hot categories (drop_first=True, alphabetical) ---
# base categories are NOT listed — they map to all-zeros

LOAN_PURPOSE_DUMMIES = [
    "Home Improvements",
    "Investment Opportunities",
    "Other",
    "Retirement Income",
]  # base = Debt Consolidation

DEVICE_DUMMIES = ["mobile", "tablet"]  # base = desktop

PROPERTY_TYPE_DUMMIES = [
    "Multi-Family Home",
    "Single-Family Home",
    "Townhome / Condominium",
]  # base = Mobile Home

PROPERTY_USE_DUMMIES = [
    "Rental Property",
    "Secondary/Vacation Home",
]  # base = Primary Residence
