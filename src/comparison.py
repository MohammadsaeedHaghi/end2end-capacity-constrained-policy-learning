"""Cross-method comparison table."""

import numpy as np
import pandas as pd


def make_comparison_table(results, b, N, N_EVAL, T, D):
    rows = []

    for r in results:
        alloc = np.asarray(r["alloc"])

        row = {
            "Method": r["tag"],
            "V_IPW": r["V_IPW"],
            "V_oracle": r["V_oracle"],
            "CapViolation": r["cap_viol_sup"],
            "Cap_OK": r["cap_ok"],
            "LP_Status": r.get("lp_status", "NA"),
            "LP_Time": r.get("lp_time", np.nan),
            "Total_Time": r.get("total_time", np.nan),
        }

        for t in range(T):
            row[f"Alloc_{t}"] = alloc[t]

        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("FINAL COMPARISON TABLE")
    print("=" * 100)
    print(f"N={N}, N_EVAL={N_EVAL}, T={T}, D={D}")
    print(f"Capacities B = {np.array2string(np.asarray(b), precision=3)}")
    print("=" * 100)

    with pd.option_context("display.float_format", lambda x: f"{x: .4f}"):
        print(df.to_string(index=False))

    print("=" * 100)

    return df
