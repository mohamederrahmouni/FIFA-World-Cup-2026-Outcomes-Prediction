import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import itertools


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier



groups={
    "Group A":["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "Group B":["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "Group C":["Brazil", "Morocco", "Haiti", "Scotland"],
    "Group D":["United States", "Paraguay", "Australia", "Turkey"],
    "Group E":["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "Group F":["Netherlands", "Japan", "Sweden", "Tunisia"],
    "Group G":["Belgium", "Iran", "Egypt", "New Zealand"],
    "Group H":["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "Group I":["France", "Senegal", "Norway", "Iraq"],
    "Group J":["Argentina", "Algeria", "Austria", "Jordan"],
    "Group K":["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "Group L":["England", "Croatia", "Ghana", "Panama"]}


features=['home_form_scored','home_form_conceded',
       'home_form_win_rate','away_form_scored','away_form_conceded',
       'away_form_win_rate','home_players_avg','best_home_player_score',
       'home_avg_attack','home_avg_defense','home_avg_pace',
       'home_avg_shooting','home_avg_passing','away_players_avg',
       'best_away_player_score','away_avg_attack','away_avg_defense',
       'away_avg_pace','away_avg_shooting','away_avg_passing',
       'players_avg_diff','attack_diff','defense_diff','home_elo',
       'away_elo','elo_diff','form_scored_diff',
       'form_conceded_diff','avg_passing_diff','avg_shooting_diff',
       'form_win_rate_diff']

st.set_page_config(page_title="FIFA World Cup Predictor", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df=pd.read_csv(r'data/teams_match_features.csv')
    df.rename(columns={"_tournament":"tournament","_date":"date","_home_team":"home_team","_away_team":"away_team","home_avg_overall":"home_players_avg","away_avg_overall":"away_players_avg","home_max_overall":"best_home_player_score","away_max_overall":"best_away_player_score","overall_diff":"players_avg_diff"},inplace=True)
    df["date"]=pd.to_datetime(df["date"])
    df=df.drop(columns=["is_neutral","is_world_cup","is_continental"])

    for col in ["home_avg_pace","home_avg_shooting","home_avg_passing","away_avg_pace","away_avg_shooting","away_avg_passing"]:
     df[col].fillna(df[col].mean())

    def result(data):
      if data['home_goals']>data['away_goals']:
        return 2
      elif data['home_goals']<data['away_goals']:
        return 0   
      else:
        return 1 

    df['result']=df.apply(result,axis=1)
    
    df["form_scored_diff"]=df["home_form_scored"]-df["away_form_scored"]
    df["form_conceded_diff"]=df["home_form_conceded"]-df["away_form_conceded"]
    df["avg_passing_diff"]=df["home_avg_passing"]-df["away_avg_passing"]
    df["avg_shooting_diff"]=df["home_avg_shooting"]-df["away_avg_shooting"]
    df["form_win_rate_diff"]=df["home_form_win_rate"]-df["away_form_win_rate"]
    return df


@st.cache_resource(show_spinner=False)
def train_prediction_model(df):
    X=df[features]
    y=df["result"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # The notebook's final best_model_clf resolves to this logistic model object.
    best_model_clf = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                                        eval_metric='mlogloss', random_state=42, n_jobs=-1)
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model_clf)])
    model_pipeline.fit(X_train,y_train)
    return model_pipeline


@st.cache_data(show_spinner=False)
def build_team_stats(df):
    def get_team_stats(team):
        home_rows=df[df["home_team"]==team].sort_values("date").tail(5)
        away_rows=df[df["away_team"]==team].sort_values("date").tail(5)
        if not home_rows.empty:
            r = home_rows.iloc[-1]
            return {
                "elo": r["home_elo"], "form_scored": r["home_form_scored"],
                "form_conceded": r["home_form_conceded"], "form_win_rate": r["home_form_win_rate"],
                "players_avg": r["home_players_avg"], "avg_attack": r["home_avg_attack"],
                "avg_defense": r["home_avg_defense"], "avg_pace": r["home_avg_pace"],
                "avg_shooting": r["home_avg_shooting"], "avg_passing": r["home_avg_passing"],
                "best_player": r["best_home_player_score"],
            }
        if not away_rows.empty:
            r = away_rows.iloc[-1]
            return {
                "elo": r["away_elo"], "form_scored": r["away_form_scored"],
                "form_conceded": r["away_form_conceded"], "form_win_rate": r["away_form_win_rate"],
                "players_avg": r["away_players_avg"], "avg_attack": r["away_avg_attack"],
                "avg_defense": r["away_avg_defense"], "avg_pace": r["away_avg_pace"],
                "avg_shooting": r["away_avg_shooting"], "avg_passing": r["away_avg_passing"],
                "best_player": r["best_away_player_score"],
            }
        return {
            "elo": 1500.0, "form_scored": 1.2, "form_conceded": 1.2, "form_win_rate": 0.33,
            "players_avg": 72.0, "avg_attack": 70.0, "avg_defense": 70.0,
            "avg_pace": 70.0, "avg_shooting": 65.0, "avg_passing": 68.0, "best_player": 75.0,
        }

    all_teams = [team for group in groups.values() for team in group]
    return {team: get_team_stats(team) for team in all_teams}


def build_match_features(home,away,team_stats):
    h = team_stats.get(home, list(team_stats.values())[0])
    a = team_stats.get(away, list(team_stats.values())[0])
    row = {
        "home_elo": h["elo"], "away_elo": a["elo"], "elo_diff": h["elo"] - a["elo"],
        "home_form_scored": h["form_scored"], "home_form_conceded": h["form_conceded"],
        "home_form_win_rate": h["form_win_rate"],
        "away_form_scored": a["form_scored"], "away_form_conceded": a["form_conceded"],
        "away_form_win_rate": a["form_win_rate"],
        "form_scored_diff": h["form_scored"] - a["form_scored"],
        "form_conceded_diff": h["form_conceded"] - a["form_conceded"],
        "form_win_rate_diff": h["form_win_rate"] - a["form_win_rate"],
        "home_players_avg": h["players_avg"], "away_players_avg": a["players_avg"],
        "players_avg_diff": h["players_avg"] - a["players_avg"],
        "home_avg_attack": h["avg_attack"], "home_avg_defense": h["avg_defense"],
        "home_avg_pace": h["avg_pace"], "home_avg_shooting": h["avg_shooting"],
        "home_avg_passing": h["avg_passing"],
        "away_avg_attack": a["avg_attack"], "away_avg_defense": a["avg_defense"],
        "away_avg_pace": a["avg_pace"], "away_avg_shooting": a["avg_shooting"],
        "away_avg_passing": a["avg_passing"],
        "attack_diff": h["avg_attack"] - a["avg_attack"],
        "defense_diff": h["avg_defense"] - a["avg_defense"],
        "avg_passing_diff": h["avg_passing"] - a["avg_passing"],
        "avg_shooting_diff": h["avg_shooting"] - a["avg_shooting"],
        "best_home_player_score": h["best_player"],
        "best_away_player_score": a["best_player"],
    }
    return pd.DataFrame([row])[features]


def simulate_match(team1,team2,model,team_stats):
    x1 = build_match_features(team1, team2, team_stats)
    proba1 = model.predict_proba(x1)[0]
    x2 = build_match_features(team2, team1, team_stats)
    proba2 = model.predict_proba(x2)[0]

    p_team1_win = (proba1[2] + proba2[0]) / 2
    p_draw = (proba1[1] + proba2[1]) / 2
    p_team1_lose = (proba1[0] + proba2[2]) / 2
    probs = np.array([p_team1_win, p_draw, p_team1_lose], dtype=np.float64)
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum()
    outcome = [1, 0, -1][np.argmax(probs)]
    return outcome, (float(probs[0]), float(probs[1]), float(probs[2]))


@st.cache_data(show_spinner=False)
def simulate_group_stage_cached(_model,team_stats):
    all_matches = []
    group_standings = {}
    for group_name, teams in groups.items():
        standings = {team: {"pts": 0} for team in teams}
        matches = []
        for t1, t2 in itertools.combinations(teams, 2):
            result, proba = simulate_match(t1, t2, _model, team_stats)
            matches.append({
                "Group": group_name, "Team 1": t1, "Team 2": t2, "Result": result,
                "P Team 1 Win": proba[0], "P Draw": proba[1], "P Team 2 Win": proba[2],
                "Predicted Winner": t1 if result == 1 else (t2 if result == -1 else "Draw"),
            })
            if result == 1:
                standings[t1]["pts"] += 3
            elif result == -1:
                standings[t2]["pts"] += 3
            else:
                standings[t1]["pts"] += 1
                standings[t2]["pts"] += 1
        ranked = sorted(standings.items(), key=lambda item: item[1]["pts"], reverse=True)
        group_standings[group_name] = {"ranked": ranked, "matches": matches}
        all_matches.extend(matches)
    return group_standings, pd.DataFrame(all_matches)


def standings_dataframe(group_standings):
    rows = []
    for group_name, data in group_standings.items():
        for rank, (team, stats) in enumerate(data["ranked"], 1):
            rows.append({
                "Group": group_name,
                "Rank": rank,
                "Team": team,
                "Pts": stats["pts"],
                "Qualified": "Yes" if rank <= 2 else ("Best third candidate" if rank == 3 else "No"),
            })
    return pd.DataFrame(rows)


def get_qualified_teams(group_standings):

    top2,top3 ={},[]

    for g,data in group_standings.items():
        ranked=data['ranked']
        top2[g] ={'1st':ranked[0][0],'2nd':ranked[1][0]}
        if len(ranked) >= 3:
            top3.append((g,ranked[2][0],ranked[2][1]))

    top3_sorted=sorted(top3,key=lambda x:(x[2]['pts']),reverse=True)
    best8=top3_sorted[:8]

    print('qualified team:')
    print('\n  1st and 2nd of each group:')
    for g,t in top2.items():
        print(f'{g} : 1st: {t["1st"]} | 2nd: {t["2nd"]}')

    print('\n  Top 3 :')
    for i, (grp, team, stats) in enumerate(best8, 1):
        print(f'- {team} ')

    best8_teams=[t[1] for t in best8]
    best8_groups=[t[0] for t in best8]
    return top2, best8_teams, best8_groups

def assign_best_thirds(best8_thirds, best8_groups):
    slots={
        'M74':{'A','B','C','D','F'},
        'M77':{'C','D','F','G','H'},
        'M79':{'C','E','F','H','I'},
        'M80':{'E','H','I','J','K'},
        'M81':{'B','E','F','I','J'},
        'M82':{'A','E','H','I','J'},
        'M85':{'E','F','G','I','J'},
        'M87':{'D','E','I','J','L'}}
    
    slot_order=['M74','M77','M79','M80','M81','M82','M85','M87']
    assigned={} 
    used_teams=set()

    for slot in slot_order:
        sloted=slots[slot]
        for grp,team in zip(best8_groups, best8_thirds):
            key=grp.replace('Group ', '')   
            if key in sloted and team not in used_teams:
                assigned[slot]=team
                used_teams.add(team)
                break
        if slot not in assigned:
            assigned[slot] = 'Senegal' 

    return assigned


def build_round32_bracket(top2,third_slots):
    
    g={grp[-1]:{'1st':t['1st'],'2nd':t['2nd']} for grp,t in top2.items()}
    
    bracket=[('M73',(g['B']['2nd'],g['A']['2nd']))]
    
    bracket=[
        ('M73',(g['A']['2nd'],g['B']['2nd'])),
        ('M74',(g['E']['1st'],third_slots['M74'])),
        ('M75',(g['F']['1st'],g['C']['2nd'])),
        ('M76',(g['C']['1st'],g['F']['2nd'])),
        ('M77',(g['I']['1st'],third_slots['M77'])),
        ('M78',(g['E']['2nd'],g['I']['2nd'])),
        ('M79',(g['A']['1st'],third_slots['M79'])),
        ('M80',(g['L']['1st'],third_slots['M80'])),
        ('M81',(g['D']['1st'],third_slots['M81'])),
        ('M82',(g['G']['1st'],third_slots['M82'])),
        ('M83',(g['K']['2nd'],g['L']['2nd'])),
        ('M84',(g['H']['1st'],g['J']['2nd'])),
        ('M85',(g['B']['1st'],third_slots['M85'])),
        ('M86',(g['J']['1st'],g['H']['2nd'])),
        ('M87',(g['K']['1st'],third_slots['M87'])),
        ('M88',(g['D']['2nd'],g['G']['2nd']))]
    return bracket


def simulate_round(matches, stage, model, team_stats):
    winners, rows = [], []
    for match_id, (team1, team2) in matches:
        result, proba = simulate_match(team1, team2, model, team_stats)
        winner = team1 if result == 1 else team2
        loser = team2 if winner == team1 else team1
        rows.append({
            "Stage": stage, "Match": match_id, "Team 1": team1, "Team 2": team2, "Winner": winner, "Loser": loser,
            "P Team 1 Win": proba[0], "P Draw": proba[1], "P Team 2 Win": proba[2],
        })
        winners.append(winner)
    return winners, rows


@st.cache_data(show_spinner=False)
def simulate_knockout_cached(_model, team_stats, group_standings):
    top2, best8_thirds, best8_groups = get_qualified_teams(group_standings)
    third_slots = assign_best_thirds(best8_thirds, best8_groups)
    all_rows = []

    r32_bracket = build_round32_bracket(top2, third_slots)
    _, r32_rows = simulate_round(r32_bracket, "Round of 32", _model, team_stats)
    all_rows.extend(r32_rows)
    winners = {row["Match"]: row["Winner"] for row in r32_rows}

    r16 = [("M89", (winners["M74"], winners["M77"])), ("M90", (winners["M73"], winners["M75"])),
           ("M91", (winners["M76"], winners["M78"])), ("M92", (winners["M79"], winners["M80"])),
           ("M93", (winners["M83"], winners["M84"])), ("M94", (winners["M81"], winners["M82"])),
           ("M95", (winners["M86"], winners["M88"])), ("M96", (winners["M85"], winners["M87"]))]
    _, rows = simulate_round(r16, "Round of 16", _model, team_stats)
    all_rows.extend(rows); winners.update({row["Match"]: row["Winner"] for row in rows})

    qf = [("M97", (winners["M89"], winners["M90"])), ("M98", (winners["M93"], winners["M94"])),
          ("M99", (winners["M91"], winners["M92"])), ("M100", (winners["M95"], winners["M96"]))]
    _, rows = simulate_round(qf, "Quarter final", _model, team_stats)
    all_rows.extend(rows); winners.update({row["Match"]: row["Winner"] for row in rows})

    sf = [("M101", (winners["M97"], winners["M98"])), ("M102", (winners["M99"], winners["M100"]))]
    _, sf_rows = simulate_round(sf, "Semi final", _model, team_stats)
    all_rows.extend(sf_rows); winners.update({row["Match"]: row["Winner"] for row in sf_rows})

    sf_losers = [row["Loser"] for row in sf_rows]
    _, third_rows = simulate_round([("M103", (sf_losers[0], sf_losers[1]))], "Third place", _model, team_stats)
    all_rows.extend(third_rows)

    _, final_rows = simulate_round([("M104", (winners["M101"], winners["M102"]))], "Final", _model, team_stats)
    all_rows.extend(final_rows)

    bracket_df = pd.DataFrame(all_rows)
    podium = {
        "Champion": final_rows[0]["Winner"],
        "Runner-up": final_rows[0]["Loser"],
        "Third place": third_rows[0]["Winner"],
        "Fourth place": third_rows[0]["Loser"],
    }
    return bracket_df, podium


@st.cache_data(show_spinner=False)
def favorite_scores(_model, team_stats):
    teams = [team for group in groups.values() for team in group]
    rows = []
    for team in teams:
        scores = []
        for opponent in teams:
            if team == opponent:
                continue
            _, proba = simulate_match(team, opponent, _model, team_stats)
            scores.append(proba[0])
        rows.append({"Team": team, "Win Probability": np.mean(scores)})
    fav = pd.DataFrame(rows).sort_values("Win Probability", ascending=False)
    fav["Win Probability"] = fav["Win Probability"] / fav["Win Probability"].sum()
    return fav


def probability_bar(labels, values, title):
    fig = go.Figure(go.Bar(x=labels, y=[v * 100 for v in values], marker_color=["#16a34a", "#64748b", "#dc2626"]))
    fig.update_layout(title=title, yaxis_title="Probability (%)", xaxis_title="", height=380, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def format_probability_columns(df):
    out = df.copy()

    for col in ["P Team 1 Win", "P Draw", "P Team 2 Win", "Win Probability"]:
        if col in out.columns:
            out[col] = (out[col] * 100).map(lambda x: f"{x:.1f}%" if pd.notnull(x) else x)

    for col in ["Team", "Team 1", "Team 2", "Winner", "Loser"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: team_label(x) if isinstance(x, str) else x)

    return out


def team_label(team):
    return str(team).upper()

def render_group_prediction(model, team_stats):
    st.subheader("Group Prediction")
    selected_group = st.selectbox("Select a group", list(groups.keys()))
    teams = groups[selected_group]
    cols = st.columns(4)

    for i, team in enumerate(teams):
     with cols[i % 4]:
        st.metric(label="",value=team_label(team))
        style_metric_cards(background_color="#1522E3",border_left_color="red",box_shadow=True)


    col1, col2 = st.columns(2)
    team1 = col1.selectbox("Team 1", teams, format_func=team_label)
    team2 = col2.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0, format_func=team_label)

    if team1 == team2:
        st.warning("Choose different teams")
        return

    if st.button("Predict Match Outcome", type="primary", use_container_width=True):
        result, proba = simulate_match(team1, team2, model, team_stats)
        labels = [f"{team1} win", "Draw", f"{team2} win"]
        cols = st.columns(3)
        cols[0].metric(f"{team_label(team1)} win", f"{proba[0] * 100:.1f}%")
        cols[1].metric("Draw", f"{proba[1] * 100:.1f}%")
        cols[2].metric(f"{team_label(team2)} win", f"{proba[2] * 100:.1f}%")
        style_metric_cards(background_color="#1522E3",border_left_color="red",box_shadow=True)
        predicted = team1 if result == 1 else (team2 if result == -1 else "Draw")
        st.success(f"Predicted outcome: {team_label(predicted) if predicted != 'Draw' else 'Draw'}")
        st.plotly_chart(probability_bar(labels, proba, "Predicted match probabilities"), use_container_width=True)



def render_knockout(bracket_df, podium):
    st.subheader("Knockout Bracket")

    cols = st.columns(4)
    cols[0].metric("🏆 Champion",team_label(podium["Champion"]))
    cols[1].metric("🥈 Runner-up",team_label(podium["Runner-up"]))
    cols[2].metric("🥉 Third place",team_label(podium["Third place"]))
    cols[3].metric("4️⃣  Fourth place",team_label(podium["Fourth place"]))
    style_metric_cards(background_color="#1522E3",border_left_color="yellow",box_shadow=True)


    st.divider()

    stage_order = ["Round of 32","Round of 16","Quarter final","Semi final","Third place","Final"]
    stage_labels = {
        "Round of 32":"Round of 32 Match Probabilities",
        "Round of 16":"Round of 16 Match Probabilities",
        "Quarter final":"Quarter Finals Match Probabilities",
        "Semi final":"Semi Finals Match Probabilities",
        "Third place":"Third Place Match Probabilities",
        "Final":"Final Match Probabilities"}
    
    selected_stages = st.multiselect("Phases",options=stage_order,
                                     default=stage_order,format_func=lambda s: stage_labels[s])

    view = bracket_df[bracket_df["Stage"].isin(selected_stages)]
    with st.expander("Results", expanded=False):
        st.dataframe(format_probability_columns(view), hide_index=True, use_container_width=True)

    st.divider()

    colors={"P Team 1 Win":"#16a34a","P Draw":"#f59e0b","P Team 2 Win":"#dc2626"}
    outcome_labels={"P Team 1 Win": "Win team 1","P Draw":"Match Nul","P Team 2 Win": "Win team 2"}

    for stage in stage_order:
        if stage not in selected_stages:
            continue

        stage_data = bracket_df[bracket_df["Stage"] == stage].copy()
        if stage_data.empty:
            continue

        with st.container():
            st.subheader(stage_labels[stage])

            stage_data["Matchup"] = (stage_data["Team 1"].apply(team_label)+ " vs "+ stage_data["Team 2"].apply(team_label))

            melted = stage_data.melt(
                id_vars=["Match", "Matchup", "Team 1", "Team 2", "Winner"],
                value_vars=["P Team 1 Win", "P Draw", "P Team 2 Win"],
                var_name="Outcome",
                value_name="Probability")
            melted["Probability %"] = (melted["Probability"] * 100).round(1)
            melted["Outcome Label"] = melted["Outcome"].map(outcome_labels)

            fig = go.Figure()

            for outcome_key, outcome_label in outcome_labels.items():
                subset = melted[melted["Outcome"] == outcome_key]
                fig.add_trace(go.Bar(
                    name=outcome_label,
                    x=subset["Matchup"],
                    y=subset["Probability %"],
                    marker_color=colors[outcome_key],
                    text=[f"{v:.1f}%" for v in subset["Probability %"]],
                    textposition="outside"))

            n_matches = len(stage_data)
            chart_height = max(380, 260 + n_matches * 55)

            fig.update_layout(barmode="group",height=chart_height,
                yaxis=dict(title="Probabilité (%)",range=[0, 115],ticksuffix="%",gridcolor="#f1f5f9",),
                xaxis=dict(title="", tickangle=-20 if n_matches > 4 else 0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=60, b=60),
                font=dict(size=13))
            fig.update_xaxes(showgrid=False)

            st.plotly_chart(fig, use_container_width=True)

            winner_cols = st.columns(min(n_matches,4))
            for i, (_, row) in enumerate(stage_data.iterrows()):
                col_idx = i % min(n_matches, 4)
                winner_cols[col_idx].caption(f"**{row['Match']}** —  {team_label(row['Winner'])}")

            st.divider()





def main():
    col=st.columns(3)
    with col[0]:
        st.image(r"pictures/wc-logo.png")
    with col[1]:
        st.markdown("<h1 style='text-align: center;'>FIFA World Cup Predictor</h1>",unsafe_allow_html=True)
    with col[2]:
        st.image(r"pictures/pic2.png")
    df = load_and_prepare_data()
    model = train_prediction_model(df)
    team_stats = build_team_stats(df)
    group_standings, _ = simulate_group_stage_cached(model, team_stats)
    standings_df = standings_dataframe(group_standings)
    bracket_df, podium = simulate_knockout_cached(model, team_stats, group_standings)
    fav_df = favorite_scores(model, team_stats)


    page = st.segmented_control("",["Group Prediction","Knockout Bracket"],default="Group Prediction")

    if page == "Group Prediction":
        render_group_prediction(model, team_stats)
    elif page == "Knockout Bracket":
        render_knockout(bracket_df, podium)


if __name__ == "__main__":
    main()



