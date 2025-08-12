# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.datasets import load_iris

# load dataset + model
iris = load_iris()
model = joblib.load('iris_pipeline.pkl')
scaler = model.named_steps['standardscaler']
pca = model.named_steps['pca']
clf = model.named_steps['logisticregression']

# prepare PCA-transformed dataset for plotting
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)
labels = iris.target

st.set_page_config(layout="wide", page_title="Iris 3D Classifier")
st.title("Iris Flower Classifier — 3D Animated")

# Sidebar controls
st.sidebar.header("Input features")
def feature_slider(col):
    mn, mx = float(X[col].min()), float(X[col].max())
    return st.sidebar.slider(col, mn, mx, float(X[col].median()))

sepal_length = feature_slider('sepal length (cm)')
sepal_width  = feature_slider('sepal width (cm)')
petal_length = feature_slider('petal length (cm)')
petal_width  = feature_slider('petal width (cm)')

# predict
user_X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred = model.predict(user_X)[0]
probs = model.predict_proba(user_X)[0]
pred_name = iris.target_names[pred]
st.subheader("Prediction")
st.write(f"**Predicted species:** {pred_name}")
st.write("**Probabilities:**")
prob_df = pd.DataFrame([probs], columns=iris.target_names).T
prob_df.columns = ['probability']
st.dataframe(prob_df.style.format({'probability': '{:.2f}'}))

# make an animated 3D Plotly figure of PCA space rotating
def make_rotating_figure(coords, labels, user_point=None, n_frames=60):
    species = iris.target_names
    frames = []
    # ensure consistent species order in traces
    species_unique = np.unique(labels)
    for i, angle in enumerate(np.linspace(0, 2*np.pi, n_frames)):
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [0, 0, 1]])
        coords_rot = coords.dot(rot.T)
        data = []
        for s in species_unique:
            mask = labels == s
            data.append(go.Scatter3d(
                x=coords_rot[mask, 0],
                y=coords_rot[mask, 1],
                z=coords_rot[mask, 2],
                mode='markers',
                marker=dict(size=6),
                name=species[s],
                hovertemplate=species[s] + '<extra></extra>'
            ))
        # add user point as last trace (so it stays on top)
        if user_point is not None:
            up_rot = user_point.dot(rot.T)
            data.append(go.Scatter3d(
                x=[up_rot[0]], y=[up_rot[1]], z=[up_rot[2]],
                mode='markers+text',
                marker=dict(size=10, symbol='diamond'),
                name='Your input',
                text=[pred_name],
                textposition='top center',
                hovertemplate='Your input: ' + pred_name + '<extra></extra>'
            ))
        frames.append(go.Frame(data=data, name=str(i)))

    # initial figure
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        margin=dict(l=0, r=0, t=30, b=0),
        updatemenus=[dict(type='buttons',
                          showactive=False,
                          y=1,
                          x=1.05,
                          xanchor='right',
                          yanchor='top',
                          pad=dict(t=0, r=10),
                          buttons=[dict(label='Play',
                                        method='animate',
                                        args=[None, {'frame': {'duration': 50, 'redraw': True},
                                                     'fromcurrent': True, 'transition': {'duration': 0}}])])])
    return fig

# compute user_point in PCA coords
user_scaled = scaler.transform(user_X)
user_pca = pca.transform(user_scaled)[0]

st.subheader("3D PCA visualization (animated)")
fig = make_rotating_figure(X_pca, labels, user_point=user_pca, n_frames=80)
st.plotly_chart(fig, use_container_width=True)

# batch CSV upload
st.subheader("Batch predictions (CSV upload)")
csv = st.file_uploader("Upload CSV with columns: " + ", ".join(iris.feature_names), type=['csv'])
if csv is not None:
    df_upload = pd.read_csv(csv)
    # basic validation
    if all(col in df_upload.columns for col in iris.feature_names):
        preds = model.predict(df_upload[iris.feature_names].values)
        df_upload['predicted_species'] = [iris.target_names[p] for p in preds]
        st.dataframe(df_upload)
        csv_out = df_upload.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv_out, file_name='predictions.csv')
    else:
        st.error("CSV must contain columns: " + ", ".join(iris.feature_names))

# show model accuracy info
st.sidebar.header("Model info")
st.sidebar.write("Test accuracy (from notebook):")
# small interface: load the precomputed accuracy or compute quickly:
# compute on the full dataset (quick estimate)
acc = np.mean(model.predict(X) == labels)
st.sidebar.write(f"{acc:.3f} (approx on full dataset)")
st.sidebar.write("Pipeline steps:")
st.sidebar.write(list(model.named_steps.keys()))
