import io
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ABC Logistics – Delay Predictor", page_icon="🚚", layout="wide")

@st.cache_resource
def load():
    try:
        bundle = joblib.load("delay_model.joblib")
        return bundle["tree"], bundle["columns"]
    except Exception:
        # Saved model unreadable (e.g. different scikit-learn version): retrain from the data file
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        df = pd.read_excel("delivery_delay.xlsx")
        X, y = df.drop(columns="Delivery_Delay"), df["Delivery_Delay"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        return tree, list(X.columns)

tree, COLS = load()
WEATHER = {"Clear": 1, "Rainy": 2, "Stormy": 3}
SLOT = {"Morning": 1, "Afternoon": 2, "Evening": 3}

def advice(r):
    """Suggested actions from the drivers found in our analysis."""
    tips = []
    if r["Warehouse_Processing_Time"] > 90:
        tips.append("Warehouse dwell above ~90 min: in our data every such parcel was late. Expedite sorting and dispatch.")
    if r["Vehicle_Age"] > 7:
        tips.append("Vehicle older than 7 years: in our data every such parcel was late. Reassign to a newer vehicle.")
    if r["Traffic_Congestion"] >= 4:
        tips.append("Heavy traffic on route: re-sequence the route or move to a less congested slot.")
    if r["Weather_Condition"] >= 2:
        tips.append("Rain or storm: add buffer time and warn the customer early.")
    if r["Delivery_Distance"] > 35:
        tips.append("Long distance: consider splitting the route or prioritising this parcel.")
    return tips

def tree_path(row):
    """The yes/no questions the tree asked for this parcel."""
    t = tree.tree_
    steps = []
    for n in tree.decision_path(row).indices[:-1]:
        f, th = COLS[t.feature[n]], t.threshold[n]
        v = row.iloc[0][f]
        steps.append(f"{f} = {v:g}  →  {'≤' if v <= th else '>'} {th:.1f}")
    return steps

st.title("🚚 Delivery Delay Predictor")
st.caption("ABC Logistics · decision-tree model · scores a parcel at dispatch using only information known before the van leaves.")

tab1, tab2 = st.tabs(["Single parcel", "Batch (upload file)"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        dist = st.number_input("Delivery distance (km)", 1.0, 50.0, 25.0, 0.5)
        traffic = st.slider("Traffic congestion (1 low – 5 high)", 1, 5, 3)
        weather = st.selectbox("Weather", list(WEATHER))
        slot = st.selectbox("Delivery slot", list(SLOT))
    with c2:
        drv = st.number_input("Driver experience (years)", 1, 19, 10)
        stops = st.number_input("Stops before delivery", 1, 9, 5)
        veh = st.number_input("Vehicle age (years)", 1, 9, 5)
        road = st.slider("Road condition (1 poor – 5 excellent)", 1, 5, 3)
    with c3:
        wt = st.number_input("Package weight (kg)", 0.5, 50.0, 25.0, 0.5)
        fuel = st.number_input("Fuel efficiency (km/l)", 5.0, 20.0, 12.0, 0.5)
        proc = st.number_input("Warehouse processing time (min)", 10, 119, 60)

    row = pd.DataFrame([{
        "Delivery_Distance": dist, "Traffic_Congestion": traffic, "Weather_Condition": WEATHER[weather],
        "Delivery_Slot": SLOT[slot], "Driver_Experience": drv, "Num_Stops": stops, "Vehicle_Age": veh,
        "Road_Condition_Score": road, "Package_Weight": wt, "Fuel_Efficiency": fuel,
        "Warehouse_Processing_Time": proc}])[COLS]

    if st.button("Predict delay risk", type="primary"):
        delayed = int(tree.predict(row)[0]) == 1
        st.metric("Prediction", "🔴 DELAYED" if delayed else "🟢 ON TIME")
        if delayed:
            st.error("Flagged: likely to miss its delivery window.")
            tips = advice(row.iloc[0])
            if tips:
                st.subheader("Suggested actions")
                for t in tips:
                    st.write("• " + t)
        else:
            st.success("Not flagged: likely to be delivered on time.")
        with st.expander("Why? The tree's decision path for this parcel"):
            for s in tree_path(row):
                st.write("• " + s)

with tab2:
    st.write("Upload a CSV or Excel file with these columns: " + ", ".join(COLS))
    f = st.file_uploader("Parcels file", type=["csv", "xlsx"])
    if f is not None:
        data = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        missing = [c for c in COLS if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            out = data.copy()
            out["Prediction"] = tree.predict(data[COLS])
            out["Prediction"] = out["Prediction"].map({1: "Delayed", 0: "On time"})
            out = out.sort_values("Prediction")
            st.metric("Parcels flagged as delayed", f"{(out['Prediction'] == 'Delayed').sum()} of {len(out)}")
            st.dataframe(out, use_container_width=True)
            buf = io.BytesIO(); out.to_csv(buf, index=False)
            st.download_button("Download scored parcels", buf.getvalue(), "scored_parcels.csv", "text/csv")
