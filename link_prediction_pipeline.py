import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ✅ Load CSV files and rename columns consistently
nodes_df = pd.read_csv("nodes.csv")
nodes_df.rename(columns={'Node ID': 'id'}, inplace=True)

# ✅ Encode categorical columns to numeric
for col in nodes_df.columns:
    if nodes_df[col].dtype == 'object' and col != 'id':
        le = LabelEncoder()
        nodes_df[col] = le.fit_transform(nodes_df[col].astype(str))

edges_df = pd.read_csv("edges.csv")
edges_df.rename(columns={
    'Source Node': 'source',
    'Target Node': 'target'
}, inplace=True)

target_df = pd.read_csv("link_prediction_target.csv")
target_df.rename(columns={
    'Source Node': 'source',
    'Target Node': 'target',
    'Label': 'label'
}, inplace=True)

# ✅ Build Graph using source and target columns
G = nx.Graph()
G.add_edges_from(edges_df[['source', 'target']].values)

# ✅ Feature Engineering Function
def generate_features(G, df_pairs, nodes_df):
    features = []
    for _, row in df_pairs.iterrows():
        u, v = row['source'], row['target']

        # Topological features — only if both nodes exist in graph
        if u in G.nodes() and v in G.nodes():
            common_neighbors = len(list(nx.common_neighbors(G, u, v)))
            jaccard = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
            pref_attach = list(nx.preferential_attachment(G, [(u, v)]))[0][2]
            adamic_adar = list(nx.adamic_adar_index(G, [(u, v)]))[0][2]
        else:
            common_neighbors, jaccard, pref_attach, adamic_adar = 0, 0, 0, 0

        # Non-topological node features
        feat_u = nodes_df[nodes_df['id'] == u].drop(columns='id').values.flatten() if u in nodes_df['id'].values else np.zeros(nodes_df.shape[1]-1)
        feat_v = nodes_df[nodes_df['id'] == v].drop(columns='id').values.flatten() if v in nodes_df['id'].values else np.zeros(nodes_df.shape[1]-1)

        # Combine all features
        feat_vector = [common_neighbors, jaccard, pref_attach, adamic_adar]
        feat_vector.extend(feat_u)
        feat_vector.extend(feat_v)

        features.append(feat_vector)
    return np.array(features)

# ✅ Prepare Features and Labels
X = generate_features(G, target_df[['source', 'target']], nodes_df)
y = target_df['label'].values

# ✅ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Define ML Models
svm_model = SVC(probability=True, C=1.0, kernel='rbf')
knn_model = KNeighborsClassifier(n_neighbors=5)
nn_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000)

# ✅ Train Models
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# ✅ Evaluate AUC and Accuracy
models = {'SVM': svm_model, 'k-NN': knn_model, 'NeuralNet': nn_model}
results = {}

for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = {'AUC': auc, 'Accuracy': acc}

# ✅ Print Results
for name, metrics in results.items():
    print(f"{name} ➝ AUC: {metrics['AUC']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")

# ✅ Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['AUC']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Link Prediction Models")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Confusion Matrix for Best Model
best_model_name = max(results, key=lambda x: results[x]['AUC'])
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix for {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ✅ Classification Report
print(classification_report(y_test, y_pred))

# ✅ Interactive Network Graph with Plotly
edge_x, edge_y = [], []
pos = nx.spring_layout(G, seed=42)
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

node_x, node_y = zip(*[pos[node] for node in G.nodes()])

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
                         hoverinfo='none', mode='lines'))
fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                         marker=dict(size=5, color='skyblue'),
                         text=list(G.nodes()), hoverinfo='text'))
fig.update_layout(title_text="Interactive Network Graph", showlegend=False)
fig.show()
