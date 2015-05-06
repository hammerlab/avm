from __future__ import print_function, absolute_divide

import pandas as pd
import numpy as np

NaN = float("nan")

def load_data(filename='AVM.xlsx'):
    df = pd.read_excel(filename)
    return create_datasets(df)

def create_datasets(df):
    # split the UVA and Pennsylvania datasets
    (_, uva), (_, pa) = df.groupby('CENTER')
    print("Extracting features for UVA")
    X_uva, Y_uva = create_dataset(uva)
    print("Extracting features for PA")
    X_pa, Y_pa = create_dataset(pa)
    return {"uva": (X_uva, Y_uva), "pa": (X_pa, Y_pa)}

def create_dataset(center_df):
    """
    For each center, split its DataFrame into a set of features
    X and a label series Y.
    """
    df_in_all_rows, input_filter_mask = extract_features(center_df)
    Y_all_rows, censor_mask = survival_outcomes(center_df, obliteration_years=4)
    combined_mask = input_filter_mask & ~censor_mask
    df_in_filtered = df_in_all_rows[combined_mask]
    Y_filtered = np.array(Y_all_rows[combined_mask])
    print(
        "Total patients=%d"
        "Complete patient rows=%d"
        "Patients with sufficient followup=%d"
        "Dataset size=%d" % (
            len(center_df),
            len(df_in_all_rows),
            len(Y_all_rows),
            len(Y_filtered)))
    assert len(Y_filtered) <= len(Y_all_rows) <= len(center_df)
    assert df_in_filtered <= len(df_in_all_rows) <= len(center_df)
    X_filtered = df_in_filtered.as_matrix()
    return X_filtered


# which columns of AVM.xlsx are we using as features for prediction of
# of AVM obliteration
feature_columns = [
    'Sex',
    'age',
    'S_M (size)',
    'S-M (location)',
    'S-M (vein)',
    'Max D',
    'Volume (mL)',
    'Hx of H',
    'Embo',
    'No drainv vein',
    'Draining_Vein_Depth',
    'Associated ',
    'Max dose (Gy)',
    'Peri_MarginalDose(Gy)',
    'Isodose',
    'Shots',
]

# some of the columns are poorly named in the original data file
feature_rename_dict = {
    'Max dose (Gy)': 'Max_Dose',
    'age': 'Age',
    'Volume (mL)': 'Volume',
    'Associated ': 'Aneurysm',
    'Peri_MarginalDose(Gy)': 'Marginal_Dose',
    'Hx of H': 'History_of_Hemorrhage',
    'No drainv vein': 'Number_Draining_Veins',
    'S_M (size)': "SM_Size",
    'S-M (location)': "SM_Location",
    'S-M (vein)': "SM_Vein",
}


def extract_features(df, log_transform_age=True, require_ric_values=False):
    """
    Parameters
    ----------

    df : DataFrame
        DataFrame extracted from AVM.xlsx

    require_ric_values : bool
        Filter rows based on availability of information about radiation
        induced changes

    Returns DataFrame subset of columns, along with boolean mask
    for filtering that DataFrame (and target values corresponding to each row)
    """
    df_in = df[feature_columns].copy()
    df_in.rename(columns=feature_rename_dict, inplace=True)

    if log_transform_age:
        # log transform of age
        df_in['Age'] = np.log(df_in['Age'])

    # remove rows with missing data for
    # - Volume of AVM
    # - Embo: previous embolization
    bad_mask = df_in.Volume.isnull() | df_in.Embo.isnull()

    # drop rows without information about radiation induced changes
    if require_ric_values:
        empty_ric = ~(df['RIC'].str.strip().str.len().isnull())
        ric_not_available = df['RIC'] == 9
        bad_mask |= empty_ric
        bad_mask |= ric_not_available

    good_mask = ~bad_mask
    return df_in, good_mask

survival_outcome_columns = [
    # Was the AVM obliterated? Coded as:
    # 1 = no response
    # 2 = partial obliteration (< 50%, detected by angiogram)
    # 3 = subtotal (> 50% but not obliterated, detected by angiogram)
    # 4 = obliterated, detected by an angiogram
    # 5 = MRI obliterated
    # 6 = MRI subtotal
    # 7 = unknown
    # 8 = not enough followup
    # 9 = less than two years
    'Final_Result',
    # either last time patient had imagining or time until positive imagining
    'K-Mtime yrs',
]

survival_outcome_rename_dict = {
    'K-Mtime yrs': 'Years',
}

def survival_outcomes(df, obliteration_years=4.0):
    """
    Parameters
    ----------

    df : DataFrame
        Raw data extracted from AVM.xlsx

    obliteration_years : float
        Was an obliteration observed within this time period?

    Returns target labels Y and mask of censored data points. Using
    the series Y requires filtering by the negation of the censor mask.
    """
    df_out = df[survival_outcome_columns].rename(
        columns=survival_outcome_rename_dict).copy()

    # only counting total obliteration, as detected by either angiogram or MRI
    obliteration = (df_out['Final_Result'] == 4) | (df_out['Final_Result'] == 5)

    before = df_out['Years'] <= obliteration_years

    # patients which haven't had an obliteration but also haven't been
    # followed long enough should be excluded
    censored_mask = before & ~obliteration

    print("# censored at %f years: %d" % (obliteration_years, censored_mask.sum()))

    # patients whose last followup is before the cutoff and had an
    # obliteration
    Y_all = np.array(before & obliteration)
    return Y_all, censored_mask

ric_columns = [
    'RIC',
    'Degree of RIC',
    'Permanet',
    'S/S from RIC',
]

ric_rename_dict = {
    'Degree of RIC': 'RIC_Degree',
    'Permanet': 'RIC_Permanent',
    'S/S from RIC': 'RIC_Symptoms',
}

def radiation_induced_changes(df):
    """
    Normalized outcomes relating to radiation induced changes for each patient
    """
    ric_df = df[ric_columns].rename(columns=ric_rename_dict).copy()
    ric_df['RIC'] = ric_df['RIC'].convert_objects(convert_numeric=True)

    ric_df['RIC_Symptoms'] = ric_df['RIC_Symptoms'].convert_objects(convert_numeric=True)
    ric_df['RIC_Symptoms'][ric_df.RIC_Symptoms == 0] = NaN

    # if only values should be False/True
    ric_df['RIC_Permanent'] = ric_df['RIC_Permanent'].convert_objects(convert_numeric=True)
    ric_df['RIC_Permanent'][ric_df['RIC_Permanent'].isnull()] = 0
    ric_df['RIC_Permanent'] = ric_df['RIC_Permanent'].astype('bool')

    # absence of data indicated either by 9 or NaN, normalize to only use one
    ric_df['RIC'][ric_df['RIC'] == 9] = np.nan
    ric_df['RIC_Degree'] = ric_df['RIC_Degree'].convert_objects(convert_numeric=True)
    # if patient has an RIC and it's accompanied by permanent symptoms, that's bad!
    ric_df['RIC_Bad'] = ((ric_df.RIC == 1) & ~ric_df.RIC_Symptoms.isnull() & ric_df.RIC_Permanent)
    return ric_df

def adverse_events(df):
    """
    Did each patient experience either a hemorrhage or a permanent symptomatic
    radiation induced change?
    """
    # convert hemorrhage counts to boolean
    hemorrhage = df['Post GK H'] > 0
    ric_df = radiation_induced_changes(df)
    return hemorrhage | ric_df.RIC_Bad
