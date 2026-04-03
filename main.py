from data_cleaning import load_data, inspect_data, clean_data
from risk_engine import apply_risk_logic
from llm_agent import generate_explanation
from langsmith import traceable

@traceable(name="delivery_risk_pipeline")
def main():
    file_path = "data/dirtyFile.csv"

    df = load_data(file_path)

    print("\n===== RAW DATA =====")
    inspect_data(df)

    df_clean = clean_data(df)

    print("\n===== CLEAN DATA =====")
    inspect_data(df_clean)

    df_final = apply_risk_logic(df_clean)

    print("\n===== FINAL OUTPUT =====")
    print(df_final[['job_id', 'delay', 'risk_score', 'risk_level', 'recommended_action']]
          .sort_values(by='risk_score', ascending=False))

    # Only top 3 to avoid API limits
    top_jobs = df_final.sort_values(by='risk_score', ascending=False).head(3)

    top_jobs['explanation'] = top_jobs.apply(generate_explanation, axis=1)

    print("\n===== LLM EXPLANATIONS =====")
    print(top_jobs[['job_id', 'risk_level', 'explanation']])


if __name__ == "__main__":
    main()