from __future__ import print_function

import numpy as np
import pandas as pd
from numpy import nan as NaN

FILENAME = 'AVM.xlsx'

# some of the columns are poorly named in the original data file
feature_rename_dict = {
    'Max dose (Gy)': 'Max_Dose',
    'age': 'Age',
    'Volume (mL)': 'Volume',
    'Associated ': 'Aneurysm',
    'Peri_MarginalDose(Gy)': 'Marginal_Dose',
    'Hx of H': 'History_of_Hemorrhage',
    'No drainv vein': 'Number_Draining_Veins',
    'S-M (vein)': 'SM_Vein',
    'OP type': 'Surgery',
    'Max D': 'Max_Diameter'
}

survival_outcome_rename_dict = {
    'K-Mtime yrs': 'Years',
}

ric_rename_dict = {
    'Degree of RIC': 'RIC_Degree',
    'Permanet': 'RIC_Permanent',
    'S/S from RIC': 'RIC_Symptoms',
}

def load_dataframe(filename=FILENAME):
    df = pd.read_excel(filename)
    df = df.rename(
        columns=survival_outcome_rename_dict).copy()
    df = df.rename(
        columns=feature_rename_dict).copy()
    df = df.rename(
        columns=ric_rename_dict).copy()
    return df

def load_datasets(filename=FILENAME):
    df = load_dataframe(filename)
    datasets = create_datasets(df)
    return datasets, df


def combine_output_variables(oblit, adverse):
    """
    Combine two boolean series, one for positive outcomes (obliteration)
    and another for negative outcomes (adverse events), into a single
    categorical variable.
    """
    only_oblit = (oblit > 0) & (adverse == 0)
    # 0 = no obliteration, no adverse event
    # 1 = adverse event (regardless of obliteration)
    # 2 = obliteration, no adverse event
    return 2 * only_oblit.astype(int) + (adverse > 0).astype(int)


def create_datasets(df):
    # split the UVA and Pennsylvania datasets
    (_, uva), (_, pa) = df.groupby('CENTER')
    print("Extracting features for UVA")
    X_uva, Y_uva, oblit_uva, adverse_uva, df_uva = create_dataset(uva)

    cols_uva = tuple(df_uva.columns)

    print("Extracting features for PA")
    X_pa, Y_pa, adverse_pa, oblit_pa, df_pa = create_dataset(pa)

    cols_pa = tuple(df_pa.columns)
    assert cols_pa == cols_uva, set(cols_pa).difference(set(cols_uva))
    return {
        "uva": (X_uva, Y_uva, oblit_uva, adverse_uva, df_uva),
        "pa": (X_pa, Y_pa, oblit_pa, adverse_pa, df_pa),
        "features": cols_pa,
    }

def create_dataset(
        center_df,
        obliteration_years=4,
        ric_years=4):
    """
    For each center, split its DataFrame into a set of features
    X and a label series Y.
    """
    df_in_all_rows, input_filter_mask = extract_features(center_df)

    print("-- Row-wise feature mask len=%d, type=%s, dtype=%s, sum=%d" % (
        len(input_filter_mask),
        type(input_filter_mask),
        input_filter_mask.dtype,
        input_filter_mask.sum()))

    oblit_all_rows, followup_mask = survival_outcomes(
        center_df, obliteration_years=obliteration_years)

    print("-- Followup mask len=%d, type=%s, dtype=%s, sum=%d" % (
        len(followup_mask),
        type(followup_mask),
        followup_mask.dtype,
        followup_mask.sum()))

    combined_mask = input_filter_mask & followup_mask
    df_in_filtered = df_in_all_rows[combined_mask]
    oblit_filtered = oblit_all_rows[combined_mask]
    print(
        "-- Total patients=%d"
        ", Complete patient rows=%d"
        ", Patients with sufficient followup=%d"
        ", Dataset size=%d" % (
            len(center_df),
            len(df_in_all_rows[input_filter_mask]),
            len(oblit_all_rows[followup_mask]),
            len(oblit_filtered)))
    assert len(oblit_filtered) <= len(oblit_all_rows) <= len(center_df)
    assert len(df_in_filtered) <= len(df_in_all_rows) <= len(center_df)
    X_filtered = np.array(df_in_filtered, dtype=float)
    print("X", X_filtered.shape, X_filtered.dtype)
    print("oblit", oblit_filtered.shape, oblit_filtered.dtype)

    adverse_all = adverse_events(center_df)
    adverse_filtered = adverse_all[combined_mask]
    Y_filtered = combine_output_variables(oblit_filtered, adverse_filtered)
    return X_filtered, Y_filtered, oblit_filtered, adverse_filtered, df_in_filtered

# which columns of AVM.xlsx are we using as features for prediction of
# of AVM obliteration
feature_columns = [
    'Sex',
    'age',
    # binarized depth of draining vein
    'S-M (vein)',
    # Location of AVM coded as:
    # 1 = frontal
    # 2 = temporal
    # 3 = parietal
    # 4 = occipital
    # 5 = thalamic
    # 6 = basal ganglia
    # 7 = corpus callosum
    # 8 = brain stem
    # 9 = cerebellum
    # 10 = insula
    'Location',
    # max diamter along any axis of AVM
    'Max D',
    # estimated volume of AVM
    'Volume (mL)',
    # history of hemorrhage
    'Hx of H',
    # was AVM embolized?
    'Embo',
    # associated aneurysm
    'Associated ',
    # max dose of gamma knife
    'Max dose (Gy)',
    # dose of gamma knife at outermost margin of AVM
    'Peri_MarginalDose(Gy)',
    # dose at certain percent distance away from center
    'Isodose',
    # number of target loci of gamma knife
    'Shots',
    # What kind of surgery did the patient undergo before the gamma knife?
    # 0 = none
    # 1 = resection
    # 2 = EVD/shunt
    # 3 = AneurysmRx
    # 4 = 1 and 2
    # 5 = 1 and 3
    'OP type'
]

# columns should be renamed when loaded from AVM excel file
renamed_feature_columns = [
    feature_rename_dict[f] if f in feature_rename_dict else f
    for f in feature_columns
]

def extract_features(
        df,
        log_transform_age=True,
        expand_location_feature=True,
        dose_size_ratios=False):
    """
    Parameters
    ----------

    df : DataFrame
        DataFrame extracted from AVM.xlsx

    log_transform_age : bool
        Take log base 2 of age

    expand_location_feature : bool
        Since location is a categorical variable, should we expand it into
        10 distinct binary variables?

    Returns DataFrame subset of columns, along with boolean mask
    for filtering that DataFrame (and target values corresponding to each row)
    """
    df_in = df[renamed_feature_columns].copy()

    # only care about surgery codes which include a resection
    surgery = df_in['Surgery']
    surgery_mask = (surgery == 1) | (surgery == 4) | (surgery == 5)
    df_in['Surgery'] = surgery_mask

    if log_transform_age:
        # log transform of age
        df_in['Age'] = np.log2(df_in['Age'])

    # remove rows with missing data for
    # - Volume of AVM
    # - Embo: previous embolization
    bad_mask = np.zeros(len(df_in), dtype=bool)
    for column_name in renamed_feature_columns:
        column = df_in[column_name]
        missing = np.array(column.isnull())
        n_missing = missing.sum()
        if n_missing > 0:
            print("-- missing %d/%d feature values for %s" % (
                n_missing, len(missing), column_name))
            bad_mask |= missing

    # expand Location feature after checking that it's not NaN
    if expand_location_feature:
        location = np.array(df_in['Location'])
        del df_in['Location']
        # grouping frontal (code #1) and insula (code #10)
        df_in['Location_frontal'] = (location == 1) | (location == 10)
        # loop over all location codes 2 .. 9
        for location_code in xrange(2, 10):
            column_name = "Location_%d" % location_code
            df_in[column_name] = location == location_code

    if dose_size_ratios:
        for dose_feature in ['Max_Dose', 'Isodose', 'Marginal_Dose']:
            for size_feature in ['Volume']:
                new_column_name = "%s_over_%s" % (dose_feature, size_feature)
                df_in[new_column_name] = df_in[dose_feature] / df_in[size_feature]

    has_all_feature_columns_mask = np.array(~bad_mask)
    print(df_in.dtypes)
    return df_in, has_all_feature_columns_mask

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

renamed_survival_outcome_columns = [
    'Years',
    'Final_Result',
]

def survival_outcomes(df, obliteration_years=4.0):
    """
    Parameters
    ----------

    df : DataFrame
        Raw data extracted from AVM.xlsx

    obliteration_years : float
        Was an obliteration observed within this time period?

    Returns target labels Y and mask for data points with sufficient followup.
    Using the series Y requires filtering by the followup mask.
    """
    df_out = df[renamed_survival_outcome_columns].copy()
    # only counting total obliteration, as detected by either angiogram or MRI
    obliteration = (df_out['Final_Result'] == 4) | (df_out['Final_Result'] == 5)

    before = df_out['Years'] <= obliteration_years

    # patients which haven't had an obliteration but also haven't been
    # followed long enough should be excluded
    censored_mask = before & ~obliteration

    print("# censored at %f years: %d / %d" % (
        obliteration_years,
        censored_mask.sum(),
        len(censored_mask)))

    # patients whose last followup is before the cutoff and had an
    # obliteration
    Y_all = np.array(before & obliteration)
    sufficient_followup_mask = np.array(~censored_mask)
    return Y_all, sufficient_followup_mask

ric_columns = [
    # was there a radiation induced change?
    'RIC',
    'Degree of RIC',
    # was the RIC permanent?
    'Permanet',
    # symptoms categorization
    'S/S from RIC',
]

renamed_ric_columns = [
    'RIC',
    'RIC_Degree',
    'RIC_Permanent',
    'RIC_Symptoms',
]

def radiation_induced_changes(df):
    """
    Normalized outcomes relating to radiation induced changes for each patient
    """

    ric_df = df[renamed_ric_columns].copy()
    ric = ric_df['RIC'].convert_objects(convert_numeric=True)
    ric[ric.isnull()] = 0
    # absence of data indicated either by 9 or NaN, normalize to only use one
    ric[ric == 9] = NaN
    print("RIC", (ric == 1).sum())
    ric_permanent = ric_df['RIC_Permanent'].convert_objects(
        convert_numeric=True)
    ric_permanent[ric_permanent.isnull()] = 0
    print("RIC Permanet", (ric_permanent == 1).sum())
    ric_bad = (ric == 1) & (ric_permanent == 1)
    return ric_bad

def adverse_events(df):
    """
    Did each patient experience either a hemorrhage or a permanent symptomatic
    radiation induced change?
    """
    # convert hemorrhage counts to boolean
    hemorrhage = df['Post GK H'] > 0
    print("H", hemorrhage.sum())
    ric_bad = radiation_induced_changes(df)
    return hemorrhage | ric_bad


def early_ric(df, months=24):
    """
    Returns boolean mask corresponding to which patients experienced
    radiation induced changes within the given threshold.
    """
    # some patients experience radiation induced changes
    # if this is encoded as 1 in 'RIC', then the time
    # until change is in the column 'RIC post GK'.
    # otherwise, look up their last MRI in 'MR FU'
    ric = df['RIC'].astype(bool)
    ric_true_time = df['RIC post GK']
    ric_false_time = df['MR FU']
    ric_time = ric_true_time.copy()
    ric_false_mask = ~ric
    ric_time[ric_false_mask] = ric_false_time[ric_false_mask]

    # cast to a float only once both RIC times and last MRI followup are both in the
    # same series, otherwise we'll get missing values
    ric_time = ric_time.astype(float)
    return ric & (ric_time <= months)
