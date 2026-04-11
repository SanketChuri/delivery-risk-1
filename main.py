from data_cleaning import load_data, inspect_data, clean_data
from risk_engine import apply_risk_logic
from llm_agent import generate_ai_brief
from phase1 import build_phase1_operational_view
from langsmith import traceable


@traceable(name="delivery_risk_pipeline")
def main():
    file_path = "data/orders_with_locations.csv"

    df = load_data(file_path)

    print("\n===== RAW DATA =====")
    inspect_data(df)

    df_clean = clean_data(df)

    print("\n===== CLEAN DATA =====")
    inspect_data(df_clean)

    df_final = apply_risk_logic(df_clean)

    ops_df = build_phase1_operational_view(
        df_final,
        telemetry_path="data/driver_locations.csv",
        fallback_region="uk",
    )

    print("\n===== FINAL OUTPUT =====")
    print(
        ops_df[
            [
                "job_id",
                "driver_id",
                "delay",
                "failure_probability",
                "risk_score",
                "risk_level",
                "alert_level",
                "recommended_action",
                "ops_action",
            ]
        ].sort_values(by="failure_probability", ascending=False)
    )

    top_jobs = ops_df.sort_values(by="failure_probability", ascending=False).head(10).copy()
    top_jobs["explanation"] = top_jobs.apply(generate_ai_brief, axis=1)

    print("\n===== LLM EXPLANATIONS =====")
    print(top_jobs[["job_id", "risk_level", "alert_level", "explanation"]])


if __name__ == "__main__":
    main()