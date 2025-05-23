import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import tkinter as tk
from tkinter import messagebox, ttk
import os

# --- Reproducibility ---
def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

# --- Feature Extraction ---
def generate_features(G, link_pairs, nodes_df=None, use_node_features=True):
    features = []
    attr_len = 0 if nodes_df is None else nodes_df.shape[1] - 1
    for u, v in link_pairs:
        if u in G and v in G:
            cn = len(list(nx.common_neighbors(G, u, v)))
            jc = next(nx.jaccard_coefficient(G, [(u, v)]))[2]
            pa = next(nx.preferential_attachment(G, [(u, v)]))[2]
            aa = next(nx.adamic_adar_index(G, [(u, v)]))[2]
            deg_u, deg_v = G.degree(u), G.degree(v)
            cluster_u, cluster_v = nx.clustering(G, u), nx.clustering(G, v)
        else:
            cn = jc = pa = aa = deg_u = deg_v = cluster_u = cluster_v = 0

        feat_vector = [cn, jc, pa, aa, deg_u, deg_v, cluster_u, cluster_v]

        if use_node_features and nodes_df is not None:
            feat_u = nodes_df.loc[nodes_df['id'] == u].drop(columns='id').values.flatten()
            feat_v = nodes_df.loc[nodes_df['id'] == v].drop(columns='id').values.flatten()
            if feat_u.size == 0:
                feat_u = np.zeros(attr_len)
            if feat_v.size == 0:
                feat_v = np.zeros(attr_len)
            feat_vector.extend(feat_u)
            feat_vector.extend(feat_v)

        features.append(feat_vector)
    return np.array(features)

# --- DBLP Dataset Loader ---
def load_dblp_data(use_node_features=False):
    with open("com-dblp.ungraph.txt", 'r') as f:
        dblp_edges = [tuple(map(int, line.strip().split())) for line in f if not line.startswith("#")]
    G = nx.Graph()
    G.add_edges_from(dblp_edges)
    pos_links = random.sample(dblp_edges, 25000)
    nodes = list(G.nodes())
    neg_links = set()
    while len(neg_links) < 25000:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and u != v:
            neg_links.add((u, v))
    neg_links = list(neg_links)
    links = pos_links + neg_links
    labels = [1]*len(pos_links) + [0]*len(neg_links)
    nodes_df = pd.DataFrame({'id': nodes}) if use_node_features else None
    X = generate_features(G, links, nodes_df, use_node_features)
    y = np.array(labels)
    return X, y, G, nodes_df

# --- CSV Data Loader ---
def load_csv_data(use_node_features=True):
    nodes_df = pd.read_csv("nodes.csv")
    nodes_df.rename(columns={'Node ID': 'id'}, inplace=True)
    for col in nodes_df.columns:
        if nodes_df[col].dtype == 'object' and col != 'id':
            nodes_df[col] = LabelEncoder().fit_transform(nodes_df[col].astype(str))
    edges_df = pd.read_csv("edges.csv")
    edges_df.rename(columns={'Source Node': 'source', 'Target Node': 'target'}, inplace=True)
    labels_df = pd.read_csv("link_prediction_target.csv")
    labels_df.rename(columns={'Source Node': 'source', 'Target Node': 'target', 'Label': 'label'}, inplace=True)
    G = nx.Graph()
    G.add_edges_from(edges_df[['source', 'target']].values)
    if not use_node_features:
        nodes_df = None
    X = generate_features(G, labels_df[['source', 'target']].values, nodes_df, use_node_features)
    y = labels_df['label'].values
    return X, y, G, nodes_df

# --- Training & Evaluation ---
def train_models(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=25, random_state=42),
        'SVM': SVC(probability=True, kernel='rbf', C=10.0, random_state=42),
        'k-NN': KNeighborsClassifier(n_neighbors=5),
        'NeuralNet': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=2000, early_stopping=True, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        results[name] = {'model': model, 'auc': auc, 'accuracy': acc, 'y_pred': preds, 'probs': probs}
    return results, X_test, y_test

# --- Plotting ---
def plot_graphs(results, y_test, best_model_name, best_res, nodes_df):
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['probs'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    cm = confusion_matrix(y_test, best_res['y_pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, best_res['y_pred']))

    if best_model_name == 'RandomForest':
        feature_names = ["CN", "JC", "PA", "AA", "Deg_u", "Deg_v", "Cluster_u", "Cluster_v"]
        if nodes_df is not None:
            attr_names = nodes_df.columns.drop('id').tolist()
            feature_names += [f"{n}_u" for n in attr_names]
            feature_names += [f"{n}_v" for n in attr_names]
        importances = best_res['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 8))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.show()

    precision, recall, _ = precision_recall_curve(y_test, best_res['probs'])
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"{best_model_name}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.show()

# --- GUI ---
def run_gui():
    def on_run(dataset_type, feature_type):
        use_node_features = feature_type.get() == "With Node Features"
        if dataset_type == 'CSV':
            X, y, G, nodes_df = load_csv_data(use_node_features)
        else:
            X, y, G, nodes_df = load_dblp_data(use_node_features)
        results, X_test, y_test = train_models(X, y)
        best_model_name = max(results, key=lambda k: results[k]['auc'])
        best_res = results[best_model_name]
        output_text.set(f"Best Model: {best_model_name}\n\n" +
                        "\n".join([f"{name}: AUC={res['auc']:.4f}, Accuracy={res['accuracy']:.4f}" for name, res in results.items()]))
        plot_graphs(results, y_test, best_model_name, best_res, nodes_df)

    root = tk.Tk()
    root.title("Link Prediction Thesis GUI")
    root.geometry("600x450")

    title_label = ttk.Label(root, text="Link Prediction with Feature Options", font=("Helvetica", 16))
    title_label.pack(pady=20)

    frame1 = ttk.Frame(root)
    frame1.pack(pady=10)
    ttk.Label(frame1, text="Choose Dataset:").pack(side="left", padx=10)
    dataset_combo = ttk.Combobox(frame1, values=["CSV", "DBLP"])
    dataset_combo.pack(side="left")
    dataset_combo.set("CSV")

    frame2 = ttk.Frame(root)
    frame2.pack(pady=10)
    ttk.Label(frame2, text="Feature Type:").pack(side="left", padx=10)
    feature_combo = ttk.Combobox(frame2, values=["Topological Only", "With Node Features"])
    feature_combo.pack(side="left")
    feature_combo.set("With Node Features")

    run_btn = ttk.Button(root, text="Run Prediction", command=lambda: on_run(dataset_combo.get(), feature_combo))
    run_btn.pack(pady=20)

    output_text = tk.StringVar()
    output_label = ttk.Label(root, textvariable=output_text, wraplength=500)
    output_label.pack(pady=10)

    root.mainloop()

# --- Entry Point ---
if __name__ == "__main__":
    run_gui()
