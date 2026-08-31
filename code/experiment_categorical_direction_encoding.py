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
OUTPUT_DIR = PROJECT_ROOT / "results" / "categorical_direction_encoding_experiment"
CONFIGURATIONS = ["baseline", "encode_delta_frac_A", "encode_delta_Tc", "encode_both"]
DIRECTION_LABELS = [-1, 0, 1]


def add_direction_columns(anchors):
    for variable, prefix in [("delta_frac_A", "dir_f"), ("delta_Tc_scaled", "dir_tc")]:
        signs = np.sign(anchors[variable].to_numpy())
        for value, suffix in [(-1, "negative"), (0, "zero"), (1, "positive")]:
            anchors[f"{prefix}_{suffix}"] = (signs == value).astype(float)
    return anchors


def model_features(output, configuration):
    columns = ["anchor_index", "u1_frac_A", "u1_Tc_scaled",
               "delta_frac_A", "delta_Tc_scaled", f"{output}_0"]
    if configuration in ["encode_delta_frac_A", "encode_both"]:
        columns += ["dir_f_negative", "dir_f_zero", "dir_f_positive"]
    if configuration in ["encode_delta_Tc", "encode_both"]:
        columns += ["dir_tc_negative", "dir_tc_zero", "dir_tc_positive"]
    return columns


def select_histories(test_batch, meta):
    candidates = meta.loc[meta.index < test_batch]
    columns = ["u0_frac_A", "u0_Tc_scaled", "delta_frac_A", "delta_Tc_scaled"]
    query = meta.loc[test_batch, columns].to_numpy(float)
    distance = np.linalg.norm(candidates[columns].to_numpy(float)-query, axis=1)
    return candidates.index[np.argsort(distance)[:base.HISTORY_SIZE]].tolist()


def run(anchors, meta, tests):
    rows, kernels = [], []
    for number, test_batch in enumerate(tests, 1):
        test = (anchors[anchors.batch_id == test_batch]
                .sort_values("anchor_index").reset_index(drop=True))
        history_ids = select_histories(test_batch, meta)
        history = anchors[anchors.batch_id.isin(history_ids)]
        observed = test.iloc[:base.OBSERVED_ANCHOR_COUNT]
        train = pd.concat([history, observed], ignore_index=True)
        hidden = np.arange(base.OBSERVED_ANCHOR_COUNT, len(test))
        for configuration in CONFIGURATIONS:
            for output in base.OUTPUTS:
                cols = model_features(output, configuration)
                model = ScaledScalarRBFGP(random_state=base.MODEL_RANDOM_STATE).fit(
                    train[cols].to_numpy(float), train[output].to_numpy(float))
                mean, std = model.predict(test[cols].to_numpy(float))
                true = test[output].to_numpy(float)
                err = mean[hidden]-true[hidden]
                final = abs(float(mean[-1])-float(true[-1]))
                rows.append({"test_number": number, "test_batch": test_batch,
                             "output": output, "configuration": configuration,
                             "hidden_mae": np.mean(np.abs(err)),
                             "hidden_rmse": np.sqrt(np.mean(err**2)),
                             "final_abs_error": final, "final_squared_error": final**2,
                             "final_posterior_std": float(std[-1])})
                kernels.append({"test_batch": test_batch, "output": output,
                                "configuration": configuration, "features": ",".join(cols),
                                "learned_kernel": str(model.gp.kernel_)})
        print(f"Completed test {number:02d}/{len(tests)}: batch {test_batch}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(kernels)


def summarize(results):
    rows=[]
    for (output, config), g in results.groupby(["output", "configuration"]):
        rows.append({"output": output, "configuration": config, "n_tests": len(g),
                     "mean_final_mae": g.final_abs_error.mean(),
                     "ci95_final_mae": 1.96*g.final_abs_error.std(ddof=1)/np.sqrt(len(g)),
                     "final_rmse": np.sqrt(g.final_squared_error.mean()),
                     "mean_hidden_mae": g.hidden_mae.mean(),
                     "ci95_hidden_mae": 1.96*g.hidden_mae.std(ddof=1)/np.sqrt(len(g)),
                     "mean_hidden_rmse": g.hidden_rmse.mean(),
                     "mean_final_posterior_std": g.final_posterior_std.mean()})
    return pd.DataFrame(rows)


def paired(results):
    rows=[]
    for output in base.OUTPUTS:
        data=results[results.output==output]
        for config in CONFIGURATIONS[1:]:
            for metric in ["final_abs_error","hidden_mae","final_posterior_std"]:
                p=data.pivot(index="test_batch",columns="configuration",values=metric)
                diff=p[config]-p.baseline
                test=wilcoxon(p[config],p.baseline)
                rows.append({"output":output,"configuration":config,"metric":metric,
                             "mean_baseline":p.baseline.mean(),"mean_candidate":p[config].mean(),
                             "percent_change_vs_baseline":100*(p[config].mean()/p.baseline.mean()-1),
                             "candidate_better_count":int((diff<0).sum()),
                             "baseline_better_count":int((diff>0).sum()),"ties":int((diff==0).sum()),
                             "wilcoxon_statistic":test.statistic,"wilcoxon_p_value":test.pvalue})
    return pd.DataFrame(rows)


def plot(summary):
    labels=["Baseline","Encode\n$\\Delta f_A$ direction","Encode\n$\\Delta T_c$ direction","Encode both\ndirections"]
    colors=["#1764ab","#55a868","#dd8452","#8172b3"]
    fig,axes=plt.subplots(2,2,figsize=(8.8,6.3),sharex=True)
    for col,output in enumerate(base.OUTPUTS):
        d=summary[summary.output==output].set_index("configuration")
        for row,(metric,ci,title) in enumerate([
            ("mean_final_mae","ci95_final_mae","final-value MAE"),
            ("mean_hidden_mae","ci95_hidden_mae","hidden-trajectory MAE")]):
            ax=axes[row,col]
            vals=d.loc[CONFIGURATIONS,metric].to_numpy(); errs=d.loc[CONFIGURATIONS,ci].to_numpy()
            ax.bar(np.arange(4),vals,yerr=errs,capsize=3,color=colors,edgecolor="white")
            symbol=r"$T$" if output=="T" else r"$C_B$"
            ax.set_title(f"{symbol}: {title}",loc="left",fontweight="bold")
            ax.set_ylabel("Mean absolute error (95% CI)")
            ax.grid(axis="y",color="#dddddd",lw=.7); ax.set_axisbelow(True)
            ax.text(-.13,1.03,chr(ord('a')+row*2+col),transform=ax.transAxes,fontweight="bold")
            ax.set_xticks(np.arange(4),labels,fontsize=8.5)
    fig.suptitle("Explicit categorical direction encoding",fontweight="bold",y=.995)
    fig.tight_layout(rect=[0,0,1,.97])
    fig.savefig(OUTPUT_DIR/"categorical_direction_encoding.png",dpi=600,bbox_inches="tight")
    fig.savefig(OUTPUT_DIR/"categorical_direction_encoding.pdf",bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(base.DATA_PATH)
    tests=pd.read_csv(base.TEST_BATCH_PATH).test_batch.astype(int).tolist()
    meta=base.batch_metadata(df)
    anchors=add_direction_columns(base.build_anchor_dataset(df,max(tests)))
    results,kernels=run(anchors,meta,tests)
    summary,comparisons=summarize(results),paired(results)
    results.to_csv(OUTPUT_DIR/"categorical_direction_all_tests.csv",index=False)
    kernels.to_csv(OUTPUT_DIR/"categorical_direction_learned_kernels.csv",index=False)
    summary.to_csv(OUTPUT_DIR/"categorical_direction_summary.csv",index=False)
    comparisons.to_csv(OUTPUT_DIR/"categorical_direction_paired_vs_baseline.csv",index=False)
    with open(OUTPUT_DIR/"experiment_config.json","w",encoding="utf-8") as f:
        json.dump({"configurations":CONFIGURATIONS,"encoding":"full one-hot for negative, zero and positive",
                   "signed_continuous_disturbances_retained":True,"history_size":base.HISTORY_SIZE,
                   "history_selection":"unrestricted nearest neighbours using complete transition descriptor",
                   "n_tests":len(tests)},f,indent=2)
    plot(summary)
    print("\nSummary:\n",summary.to_string(index=False))
    print("\nPaired against baseline:\n",comparisons.to_string(index=False))


if __name__=="__main__":
    main()
