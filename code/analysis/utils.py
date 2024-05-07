"""
This file contains code for preprocessing of the TCGA datasets. 
This exists so that each further figure can be generated in its own notebook.
"""
import pandas as pd
import numpy as np



# preproc for nuclear features
def _preproc_nuhifs(nuhif_path, agg_methods = ["MEAN", "STD"], cell_class="CANCER"):
    """Preprocessing for nuHIFs: keep only features with correct agg methods from cancer cells.
    Fill NaNs with median."""
    
    all_features = pd.read_excel(nuhif_path)
    
    if "FFPE_ID" in all_features.columns:
        all_features.rename(columns={"FFPE_ID": "H & E_ID"}, inplace=True)
    all_features.set_index("H & E_ID", inplace=True)


    feature_names = list(all_features.columns)
    selected_nuc_features = []
    for feature in feature_names:
        if "NUCLEUS" in feature:
            if ("_MIN_" not in feature) and ("_MAX_" not in feature) and ("ORIENTATION" not in feature):
                if feature.split("[")[0] in agg_methods:
                    if cell_class in feature:
                        selected_nuc_features.append(feature)
    
    # rename "CANCER EPITHELIAL CELL" and "FFPE"
    cancer_nuc_remap = {}
    for col in selected_nuc_features:
        col_new = "CANCER CELL".join(col.split("CANCER EPITHELIAL CELL"))
        if col_new[-4:]=="FFPE":
            col_new = col_new[:-4] + "H & E"
        cancer_nuc_remap[col] = col_new
       
    # rename "CANCER NUCLEUS" to "CANCER CELL NUCLEUS
    cancer_nuc_remap = {}
    for col in selected_nuc_features:
        col_new = "CANCER CELL_NUCLEUS".join(col.split("CANCER_NUCLEUS"))
        if col_new[-4:]=="FFPE":
            col_new = col_new[:-4] + "H & E"
        cancer_nuc_remap[col] = col_new    
        
    # choose which
    standard_features = all_features[selected_nuc_features].copy()
    standard_features.rename(columns=cancer_nuc_remap, inplace=True)
    
    
    with pd.option_context('mode.use_inf_as_na', True):
        standard_features = standard_features.fillna(standard_features.median())
    return standard_features

# preproc meta
def _preproc_meta(meta_path):
    """preproc meta"""
    all_features = pd.read_excel(meta_path)
    
    if "FFPE_ID" in all_features.columns:
        all_features.rename(columns={"FFPE_ID": "H & E_ID"}, inplace=True)
    all_features.set_index("H & E_ID", inplace=True)
    return all_features


def load_pre_post_processing(prefix=""):
    """stereotyped pre-post processing for LUAD, BRCA, PRAD"""
    # how all the data relates

    nuhif_paths = {
        "brca": f"{prefix}data/features/brca_nuhifs.xlsx",
        "luad": f"{prefix}data/features/luad_nuhifs.xlsx",
        "prad": f"{prefix}data/features/prad_nuhifs.xlsx"
    }

    meta_paths = {
        "brca": f"{prefix}data/features/brca_metadata.xlsx",
        "luad": f"{prefix}data/features/luad_metadata.xlsx",
        "prad": f"{prefix}data/features/prad_metadata.xlsx"
    }

    cancer_nuhif_data = {}
    fibroblast_nuhif_data = {}
    lymphocyte_nuhif_data = {}
    meta_data = {}
    
    for key in ["brca", "luad", "prad"]:
        cancer_nuhifs = _preproc_nuhifs(nuhif_paths[key], cell_class="CANCER")
        fibroblast_nuhifs = _preproc_nuhifs(nuhif_paths[key], cell_class="FIBROBLAST")
        lymphocyte_nuhifs = _preproc_nuhifs(nuhif_paths[key], cell_class="LYMPHOCYTE")
        meta = _preproc_meta(meta_paths[key])
        
        assert(all(np.asarray(cancer_nuhifs.index)==np.asarray(fibroblast_nuhifs.index)))
        assert(all(np.asarray(cancer_nuhifs.index)==np.asarray(lymphocyte_nuhifs.index)))
        assert(all(np.asarray(cancer_nuhifs.index)==np.asarray(meta.index)))

        cancer_nuhif_data[key] = cancer_nuhifs
        fibroblast_nuhif_data[key] = fibroblast_nuhifs
        lymphocyte_nuhif_data[key] = lymphocyte_nuhifs
        meta_data[key] = meta
    
    return cancer_nuhif_data, fibroblast_nuhif_data, lymphocyte_nuhif_data, meta_data


### Analysis tools

def qqplot(results_df, uncorr_p_column='p_beta', corr_p_column="p_beta_fdr_bh", out_path=None):
    reject = results_df[corr_p_column] < 0.05
    if np.sum(reject) > 0:
        fig, ax = vis.qq_plot_pvals(results_df[uncorr_p_column], rejected_h0=reject)
    else:
        fig, ax = vis.qq_plot_pvals(results_df[uncorr_p_column])
    ax.set_ylabel("Uncorrected p-values (-log$_{10}$(p))")
    if out_path != None:
        fig.savefig(out_path)
    #fig.show()
    
def generate_KM_plot_and_summary(value, hif_df, survival_df, event_indicator, time_indicator, cox_results):
    
    # drop NaNs in survival DF
    survival_df = survival_df[[event_indicator, time_indicator]].dropna()
    
    # fill nans in hif df
    hif_df = hif_df.loc[list(survival_df.index)]
    with pd.option_context('mode.use_inf_as_na', True):
        hif_df = hif_df.fillna(hif_df.median())
        
    # check that we did this correctly
    assert(all(hif_df.index==survival_df.index)), "Index mismatch between survival df and HIF df."
    
    single_hif_df = hif_df[[value]].copy()

    bin_value = f"{value} BINARY"
    human_bin_value = f"{value} HUMAN"
    single_hif_df.loc[:, bin_value] = np.where(single_hif_df.loc[:, value] > np.median(single_hif_df.loc[:, value]), 1, 0)
    single_hif_df.loc[:, human_bin_value] = np.where(single_hif_df.loc[:, value] > np.median(single_hif_df.loc[:, value]), f"High Feature Value | {value}", f"Low Feature Value | {value}")
    single_hif_df.fillna(single_hif_df.median(), inplace = True)
    
    km_df = single_hif_df.join(survival_df, how="outer")
    km_df[event_indicator] = km_df[event_indicator].astype(int)
    km_df[time_indicator] = km_df[time_indicator].astype(float)
    km_df[time_indicator] = km_df[time_indicator]/30
    
    # print(km_df[time_column])
    fig = plt.figure(figsize=(5.5,4.0))
    ax1 = plt.subplot()

    #run KM Fitter & plot values
    low = (km_df[human_bin_value] == f"Low Feature Value | {value}")
    kmf_low = KaplanMeierFitter()
    ax1 = kmf_low.fit(km_df.loc[low][time_indicator], km_df.loc[low][event_indicator], label=f"Low Value").plot_survival_function(ax=ax1, show_censors=True, ci_show=False, censor_styles={'ms': 6, 'marker': '+'}, color="blue")
    ax1.set_ylim([0,1])

    high = (km_df[human_bin_value] == f"High Feature Value | {value}")
    kmf_high = KaplanMeierFitter()
    ax1 = kmf_high.fit(km_df.loc[high][time_indicator], km_df.loc[high][event_indicator], label=f"High Value").plot_survival_function(ax=ax1, show_censors=True, ci_show=False, censor_styles={'ms': 6, 'marker': '+'}, color="red")

    add_at_risk_counts(kmf_low, kmf_high, ax=ax1)
    ax1.set_xlabel("Time (mo)")
    ax1.legend(bbox_to_anchor=(1.05, 1))
    ax1.set_title(value)
#     plt.tight_layout(**plo.layout_pad)

#     plt.show()

    # collect data to summarize
    kmfs = [kmf_low, kmf_high]
    keys = [f"Low Feature Value | {value}", f"High Feature Value | {value}"]
    col_names = ["Median Survival (mo)", "Median Survival 95% CI", "6mo Survival", "12mo Survival", "24mo Survival", "HR", "HR 95% CI", "HR p"]
    summary_df = pd.DataFrame(columns=col_names)
    for ii, km in enumerate(kmfs):
        name = keys[ii]
        ms = np.round(median_survival_times(km),2)
        ms_ci_df = median_survival_times(km.confidence_interval_)
        ms_ci = f"{np.round(ms_ci_df.values.flatten()[0],2)}, {np.round(ms_ci_df.values.flatten()[1],2)}"

        mo6_12_24 = km.survival_function_at_times([6,12,24])
        mo6 = np.round(mo6_12_24[6],2)
        mo12 = np.round(mo6_12_24[12],2)
        mo24 = np.round(mo6_12_24[24],2)

        hr = np.round(cox_results.loc[name]["exp(coef)"],2)
        ci = f"{np.round(cox_results.loc[name]['exp(coef) lower 95%'],2)}, {np.round(cox_results.loc[name]['exp(coef) upper 95%'],2)}"

        res = [ms, ms_ci, mo6, mo12, mo24, hr, ci, np.round(cox_results.loc[name]["p"], 3)]
        summary_df.loc[name] = res
    
    display(summary_df)

    return summary_df

# robust z-score features
def robust_z_score(data):
    """robust z for numpy array"""
    med = np.median(data)
    mad = np.median(np.abs(data-med))
    return 0.6745 * (data - med) / mad

def robust_z(df, axis=0):
    """computes robust z-score: (x - median) / (1.4826 * MAD)"""
    df_robust = df.copy()
    # apply robust scaling
    assert axis in [0, 1], "Axis must be 0 (columns) or 1 (rows)."
    if axis == 0:
        for column in df_robust.columns:
            df_robust[column] = (df_robust[column] - df_robust[column].median()) / (
                df_robust[column].mad() * 1.4826
            )
    elif axis == 1:
        for index in df_robust.index:
            df_robust.loc[index] = (df_robust.loc[index] - df_robust.loc[index].median()) / (
                df_robust.loc[index].mad() * 1.4826
            )
    return df_robust


