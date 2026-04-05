def calculate_delay(df):
    df['delay'] = df['actual_time'] - df['scheduled_time']
    df['delay'] = df['delay'].apply(lambda x: max(x, 0))
    return df


def calculate_risk_score(row):
    score = 0

    if row['delay'] > 10:
        score += 30
    if row['delay'] > 20:
        score += 20

    if row['priority'] == 'high':
        score += 25
    elif row['priority'] == 'medium':
        score += 10

    if row['traffic_level'] == 'heavy':
        score += 15
    elif row['traffic_level'] == 'medium':
        score += 5

    if row['status'] != 'delivered':
        score += 10

    return score


def assign_risk_level(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def recommend_action(row):
    if row['risk_level'] == "High":
        status = str(row.get('status', '')).lower()
        delay = row.get('delay', 0)
        is_picked_and_delayed = (status in {
            'picked_up',
            'in_transit',
            'delayed',
        }) and (delay > 0)
        if is_picked_and_delayed:
            return "Expedite current driver and proactively notify customer"
        return "Reassign driver or notify operations immediately"
    elif row['risk_level'] == "Medium":
        return "Monitor closely and prepare contingency"
    else:
        return "No immediate action required"


def apply_risk_logic(df):
    df = calculate_delay(df)
    df['risk_score'] = df.apply(calculate_risk_score, axis=1)
    df['risk_level'] = df['risk_score'].apply(assign_risk_level)
    df['recommended_action'] = df.apply(recommend_action, axis=1)
    return df