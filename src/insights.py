def generate_insights(df, idx, segment_col):
    row = df.iloc[idx]
    segment = row[segment_col]

    peers = df[df[segment_col] == segment]

    insights = []

    if row["revenue_per_employee"] < peers["revenue_per_employee"].quantile(0.25):
        insights.append(
            "Revenue per employee is low relative to similar companies, indicating potential efficiency risks."
        )

    if row["it_spend_per_employee"] > peers["it_spend_per_employee"].quantile(0.75):
        insights.append(
            "IT spend per employee is high compared to peers, suggesting a digitally intensive operating model."
        )

    if row["infra_density_score"] > peers["infra_density_score"].quantile(0.90):
        insights.append(
            "IT infrastructure density is unusually high, which may increase fixed operating costs."
        )

    if not insights:
        insights.append(
            "The company operates broadly in line with peers across key operational metrics."
        )

    return insights
