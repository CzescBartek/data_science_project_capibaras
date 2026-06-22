import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('../data/features.csv')
df = data.drop(['img_id'], axis=1)
df = df.dropna(axis=0)

x = df.drop(['Cancerous'], axis=1)
y = df[['Cancerous']]

dev_x, test_x, dev_y, test_y = train_test_split(
    x, y, stratify=y, random_state=0, test_size=0.2,
)

scaler = StandardScaler()
kfold = KFold(n_splits=5, random_state=0, shuffle=True)
n_neighbors = [1, 3, 5, 7, 9, 11, 15, 21, 31, 40, 50]

cv_results = []

for k in n_neighbors:
    fold_aucs = []
    for train_idx, val_idx in kfold.split(dev_x):
        k_train_x, k_val_x = dev_x.iloc[train_idx], dev_x.iloc[val_idx]
        k_train_y, k_val_y = dev_y.iloc[train_idx], dev_y.iloc[val_idx]
        
        k_train_x_scaled = scaler.fit_transform(k_train_x)
        k_val_x_scaled = scaler.transform(k_val_x)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(k_train_x_scaled, np.ravel(k_train_y))
        
        pred = knn.predict_proba(k_val_x_scaled)[:, 1]
        fold_aucs.append(roc_auc_score(k_val_y, pred))
    
    mean_auc = np.mean(fold_aucs)
    cv_results.append(mean_auc)
    print(f"k={k}, Mean CV AUC: {mean_auc:.4f}")

absolute_best_k_idx = np.argmax(cv_results)
best_k_score = cv_results[absolute_best_k_idx]
k_tolerance = 0.005

best_k = n_neighbors[absolute_best_k_idx]
for i, mean_score in enumerate(cv_results):
    if mean_score >= (best_k_score - k_tolerance):
        best_k = n_neighbors[i]
        break

print(f"Best k from Cross-Validation (with tolerance): {best_k}")

dev_x_scaled = scaler.fit_transform(dev_x)
test_x_scaled = scaler.transform(test_x)

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(dev_x_scaled, np.ravel(dev_y))

test_probs = final_model.predict_proba(test_x_scaled)[:, 1]
final_auc = roc_auc_score(test_y, test_probs)

print(f"Final Test ROC AUC with k={best_k}: {final_auc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(n_neighbors, cv_results, label='CV Mean AUC', marker='s', color='green')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.xlabel('Neighbors (k)')
plt.ylabel('ROC AUC Score')
plt.title('Cross-Validation Results')
plt.legend()
plt.grid(True)
plt.savefig('../result/figures/KNN_Cross_Validation', dpi=300, bbox_inches='tight') 
plt.show()
plt.close() 

test_preds = final_model.predict(test_x_scaled)
joblib.dump(final_model, '../result/models/KNN_Model.pkl')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

cm = confusion_matrix(test_y, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax1, cmap='Blues')
ax1.set_title(f'Confusion Matrix (k={best_k})')

fpr, tpr, _ = roc_curve(test_y, test_probs)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {final_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Receiver Operating Characteristic (ROC)')
ax2.legend(loc="lower right")
ax2.grid(True)

plt.tight_layout()
plt.savefig('../result/figures/KNN_CM_ROC_with', dpi=300, bbox_inches='tight') 
plt.show()
plt.close()