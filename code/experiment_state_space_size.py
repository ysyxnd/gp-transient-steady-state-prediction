from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import experiment_history_selection as base
from transient_response_completion import ScaledScalarRBFGP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "state_space_size_experiment"
WINDOWS = [20, 30, 50, 70, 100]
EVALUABLE_WINDOWS = [50, 70, 100]
N_TESTS = 30
TEST_SEED = 123


def make_state_metadata(df):
    grouped = df.sort_values(["batch_id", "sample_id"]).groupby("batch_id")
    meta = base.batch_metadata(df)
    meta["Cb_initial"] = grouped.Cb.first()
    meta["T_initial"] = grouped.T.first()
    meta["Cb_final"] = grouped.Cb.last()
    meta["T_final"] = grouped.T.last()
    cb_min = meta[["Cb_initial", "Cb_final"]].to_numpy().min()
    cb_max = meta[["Cb_initial", "Cb_final"]].to_numpy().max()
    t_min = meta[["T_initial", "T_final"]].to_numpy().min()
    t_max = meta[["T_initial", "T_final"]].to_numpy().max()
    for c in ["Cb_initial", "Cb_final"]:
        meta[c+"_norm"] = (meta[c]-cb_min)/(cb_max-cb_min)
    for c in ["T_initial", "T_final"]:
        meta[c+"_norm"] = (meta[c]-t_min)/(t_max-t_min)
    ranges = {"Cb_min": cb_min, "Cb_max": cb_max, "T_min": t_min, "T_max": t_max}
    return meta, ranges


def valid_ids(meta, window):
    lower = (1-window/100)/2
    upper = (1+window/100)/2
    cols = ["Cb_initial_norm", "Cb_final_norm", "T_initial_norm", "T_final_norm"]
    valid = meta[cols].ge(lower).all(axis=1) & meta[cols].le(upper).all(axis=1)
    return meta.index[valid].to_numpy(int)


def choose_tests(meta, df):
    valid50 = valid_ids(meta, 50)
    final_time = df.groupby("batch_id").time.max()
    eligible = []
    valid50_set = set(valid50)
    for batch in valid50:
        if (sum(h < batch for h in valid50_set) >= base.HISTORY_SIZE and
                final_time.loc[batch] > 180):
            eligible.append(int(batch))
    rng = np.random.default_rng(TEST_SEED)
    return sorted(rng.choice(eligible, N_TESTS, replace=False).tolist()), eligible


def select_histories(test_batch, window_ids, meta):
    candidates = meta.loc[[b for b in window_ids if b < test_batch]]
    cols = ["u0_frac_A", "u0_Tc_scaled", "delta_frac_A", "delta_Tc_scaled"]
    query = meta.loc[test_batch, cols].to_numpy(float)
    distance = np.linalg.norm(candidates[cols].to_numpy(float)-query, axis=1)
    return candidates.index[np.argsort(distance)[:base.HISTORY_SIZE]].tolist()


def run(anchors, meta, tests, ids_by_window):
    rows, selections, kernels = [], [], []
    for number, test_batch in enumerate(tests, 1):
        test = anchors[anchors.batch_id==test_batch].sort_values("anchor_index").reset_index(drop=True)
        observed = test.iloc[:base.OBSERVED_ANCHOR_COUNT]
        hidden = np.arange(base.OBSERVED_ANCHOR_COUNT,len(test))
        for window in EVALUABLE_WINDOWS:
            history_ids = select_histories(test_batch,ids_by_window[window],meta)
            history = anchors[anchors.batch_id.isin(history_ids)]
            train = pd.concat([history,observed],ignore_index=True)
            selections.append({"test_batch":test_batch,"window_percent":window,
                               "history_batches":",".join(map(str,history_ids))})
            for output in base.OUTPUTS:
                cols=base.model_features(output)
                model=ScaledScalarRBFGP(random_state=base.MODEL_RANDOM_STATE).fit(
                    train[cols].to_numpy(float),train[output].to_numpy(float))
                mean,std=model.predict(test[cols].to_numpy(float)); true=test[output].to_numpy(float)
                err=mean[hidden]-true[hidden]; final=abs(float(mean[-1])-float(true[-1]))
                rows.append({"test_number":number,"test_batch":test_batch,"output":output,
                             "window_percent":window,"hidden_mae":np.mean(np.abs(err)),
                             "hidden_rmse":np.sqrt(np.mean(err**2)),"final_abs_error":final,
                             "final_squared_error":final**2,"final_posterior_std":float(std[-1])})
                kernels.append({"test_batch":test_batch,"output":output,"window_percent":window,
                                "learned_kernel":str(model.gp.kernel_)})
        print(f"Completed test {number:02d}/{len(tests)}: batch {test_batch}",flush=True)
    return pd.DataFrame(rows),pd.DataFrame(selections),pd.DataFrame(kernels)


def summarize(results, counts):
    rows=[]
    for window in WINDOWS:
        if window not in EVALUABLE_WINDOWS:
            rows.append({"output":"not_evaluated","window_percent":window,"n_valid_transitions":counts[window],"n_tests":0})
    for (output,window),g in results.groupby(["output","window_percent"]):
        rows.append({"output":output,"window_percent":window,"n_valid_transitions":counts[window],"n_tests":len(g),
                     "mean_final_mae":g.final_abs_error.mean(),
                     "ci95_final_mae":1.96*g.final_abs_error.std(ddof=1)/np.sqrt(len(g)),
                     "final_rmse":np.sqrt(g.final_squared_error.mean()),
                     "mean_hidden_mae":g.hidden_mae.mean(),
                     "ci95_hidden_mae":1.96*g.hidden_mae.std(ddof=1)/np.sqrt(len(g)),
                     "mean_hidden_rmse":g.hidden_rmse.mean(),
                     "mean_final_posterior_std":g.final_posterior_std.mean()})
    return pd.DataFrame(rows)


def paired(results):
    rows=[]
    for output in base.OUTPUTS:
        d=results[results.output==output]
        for window in [70,100]:
            for metric in ["final_abs_error","hidden_mae","final_posterior_std"]:
                p=d.pivot(index="test_batch",columns="window_percent",values=metric)
                diff=p[window]-p[50]; test=wilcoxon(p[window],p[50])
                rows.append({"output":output,"window_percent":window,"metric":metric,
                             "mean_50":p[50].mean(),"mean_candidate":p[window].mean(),
                             "percent_change_vs_50":100*(p[window].mean()/p[50].mean()-1),
                             "candidate_better_count":int((diff<0).sum()),"window_50_better_count":int((diff>0).sum()),
                             "ties":int((diff==0).sum()),"wilcoxon_statistic":test.statistic,"wilcoxon_p_value":test.pvalue})
    return pd.DataFrame(rows)


def plot(summary,counts):
    fig,axes=plt.subplots(2,2,figsize=(8.2,6.2),sharex=True)
    for col,output in enumerate(base.OUTPUTS):
        d=summary[summary.output==output].set_index("window_percent")
        for row,(metric,ci,title) in enumerate([
            ("mean_final_mae","ci95_final_mae","final-value MAE"),
            ("mean_hidden_mae","ci95_hidden_mae","hidden-trajectory MAE")]):
            ax=axes[row,col]; y=d.loc[EVALUABLE_WINDOWS,metric]; e=d.loc[EVALUABLE_WINDOWS,ci]
            ax.errorbar(EVALUABLE_WINDOWS,y,yerr=e,marker="o",lw=1.8,capsize=3,color="#1764ab")
            symbol=r"$T$" if output=="T" else r"$C_B$"
            ax.set_title(f"{symbol}: {title}",loc="left",fontweight="bold")
            ax.set_ylabel("Mean absolute error (95% CI)"); ax.grid(color="#dddddd",lw=.7); ax.set_axisbelow(True)
            ax.text(-.13,1.03,chr(ord('a')+row*2+col),transform=ax.transAxes,fontweight="bold")
    for ax in axes[-1]: ax.set_xlabel("State-space window (% of domain)"); ax.set_xticks(EVALUABLE_WINDOWS)
    fig.suptitle("Effect of state-space size on GP prediction",fontweight="bold",y=.995)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(OUTPUT_DIR/"state_space_size_performance.png",dpi=600,bbox_inches="tight")
    fig.savefig(OUTPUT_DIR/"state_space_size_performance.pdf",bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,3.8)); vals=[counts[w] for w in WINDOWS]
    ax.bar([str(w) for w in WINDOWS],vals,color="#4c9fbd"); ax.axhline(base.HISTORY_SIZE,color="#d95f02",ls="--",label="30-history requirement")
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*.015,str(v),ha="center")
    ax.set_ylim(0, max(vals)*1.10)
    ax.set_xlabel("State-space window (% of domain)"); ax.set_ylabel("Valid transitions")
    ax.set_title("Data availability within nested state-space windows",fontweight="bold"); ax.legend(frameon=False)
    ax.grid(axis="y",color="#dddddd",lw=.7); ax.set_axisbelow(True); fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"state_space_window_counts.png",dpi=600,bbox_inches="tight")
    fig.savefig(OUTPUT_DIR/"state_space_window_counts.pdf",bbox_inches="tight"); plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); df=pd.read_csv(base.DATA_PATH)
    meta,ranges=make_state_metadata(df); ids={w:valid_ids(meta,w) for w in WINDOWS}; counts={w:len(ids[w]) for w in WINDOWS}
    tests,eligible50=choose_tests(meta,df); anchors=base.build_anchor_dataset(df,max(tests))
    results,selections,kernels=run(anchors,meta,tests,ids)
    summary,comparisons=summarize(results,counts),paired(results)
    results.to_csv(OUTPUT_DIR/"state_space_size_all_tests.csv",index=False)
    selections.to_csv(OUTPUT_DIR/"selected_history_batches.csv",index=False)
    kernels.to_csv(OUTPUT_DIR/"state_space_size_learned_kernels.csv",index=False)
    summary.to_csv(OUTPUT_DIR/"state_space_size_summary.csv",index=False)
    comparisons.to_csv(OUTPUT_DIR/"state_space_size_paired_vs_50.csv",index=False)
    pd.DataFrame({"test_batch":tests}).to_csv(OUTPUT_DIR/"selected_test_batches.csv",index=False)
    with open(OUTPUT_DIR/"experiment_config.json","w",encoding="utf-8") as f:
        json.dump({"windows_percent":WINDOWS,"evaluated_windows":EVALUABLE_WINDOWS,
                   "valid_transition_counts":counts,"normalization_ranges":ranges,
                   "window_definition":"central nested square in normalized Cb-T space; initial and final states both inside",
                   "history_size":base.HISTORY_SIZE,"history_selection":"30 nearest eligible prior transitions",
                   "n_common_tests":len(tests),"test_seed":TEST_SEED,
                   "insufficient_windows":[20,30]},f,indent=2)
    plot(summary,counts); print("\nCounts:",counts); print("\nTests:",tests)
    print("\nSummary:\n",summary.to_string(index=False)); print("\nPaired vs 50%:\n",comparisons.to_string(index=False))


if __name__=="__main__": main()
