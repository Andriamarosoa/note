"""Label-free audit profiles and transparent paired retention diagnostics."""
from __future__ import annotations

import numpy as np

from scripts.verify_v273_selector_repair_artifacts import named_pairs,require


def audit_profiles(proposals,audits,prediction,baseline):
    """Describe all complete groups giving the selected K; never read truth."""
    require(audits.shape==(*proposals.shape,9),'audit dimensions')
    result=np.full(len(baseline),'keep',dtype='<U55')
    for i in np.flatnonzero(prediction!=baseline):
        a=audits[i,proposals[i]==prediction[i]]
        require(len(a)>0,'selected destination must be proposed')
        positive=a[:,3]>a[:,4];negative=a[:,4]>a[:,3]
        if not np.any(a[:,7]>0):result[i]='no_matching_local_support'
        elif positive.any() and negative.any():result[i]='mixed_by_complete_group'
        elif negative.any():result[i]='local_rates_favor_regression_for_all_groups'
        elif positive.any():result[i]='local_rates_favor_correction_for_all_groups'
        else:result[i]='tied_local_rates'
    return result


def retention(truth,baseline,parent,candidate):
    old_cor=(baseline!=truth)&(parent==truth);new_cor=(baseline!=truth)&(candidate==truth)
    old_reg=(baseline==truth)&(parent!=truth);new_reg=(baseline==truth)&(candidate!=truth)
    return dict(paired=named_pairs(truth,parent,candidate),
        corrections_retained=int((old_cor&new_cor).sum()),corrections_lost=int((old_cor&~new_cor).sum()),
        corrections_added=int((new_cor&~old_cor).sum()),regressions_retained=int((old_reg&new_reg).sum()),
        regressions_avoided=int((old_reg&~new_reg).sum()),regressions_added=int((new_reg&~old_reg).sum()),
        corrections_union=int((old_cor|new_cor).sum()),oracle_union_not_prediction=True)


def profile_retention(profiles,truth,baseline,parent,candidate):
    return {str(profile):dict(rows=int((profiles==profile).sum()),
        parent_vs_freeze=named_pairs(truth[profiles==profile],baseline[profiles==profile],parent[profiles==profile]),
        candidate_vs_freeze=named_pairs(truth[profiles==profile],baseline[profiles==profile],candidate[profiles==profile]),
        retention=retention(truth[profiles==profile],baseline[profiles==profile],parent[profiles==profile],candidate[profiles==profile]))
        for profile in np.unique(profiles)}
