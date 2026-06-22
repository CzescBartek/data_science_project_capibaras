import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split, KFold, learning_curve, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, 
                             ConfusionMatrixDisplay, accuracy_score, 
                             recall_score, precision_score, f1_score)

def RandomForest_model(features_path, prediction_results_path, model_path, feature_names_path, X_test_path, y_test_path):
    # 1. Data Loading and Preparation
    df = pd.read_csv(features_path).dropna(axis=0)
    if 'img_id' in df.columns:
        df = df.drop(['img_id'], axis=1)

    X = df.drop(['Cancerous'], axis=1)
    y = df['Cancerous']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1907, stratify=y
    )

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare directories for results based on provided paths
    figures_dir = os.path.join(os.path.dirname(model_path), '../figures')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prediction_results_path), exist_ok=True)

    kfold = KFold(n_splits=5, random_state=1907, shuffle=True)

    # 2. Hyperparameter Tuning: Max Depth (With Overfitting Protection)
    base_n = 100
    depth_range = range(1, 11)
    depth_means = []
    depth_stds = []

    print("Analyzing Max Depth...")
    for d in depth_range:
        fold_aucs = []
        for train_idx, val_idx in kfold.split(X_train_scaled):
            rf = RandomForestClassifier(n_estimators=base_n, max_depth=d, min_samples_leaf=5, random_state=1907, n_jobs=-1)
            rf.fit(X_train_scaled[train_idx], np.ravel(y_train.iloc[train_idx]))
            probs = rf.predict_proba(X_train_scaled[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_train.iloc[val_idx], probs))
        depth_means.append(np.mean(fold_aucs))
        depth_stds.append(np.std(fold_aucs))

    # Tolerance Rule for Max Depth
    absolute_best_d_idx = np.argmax(depth_means)
    best_d_score = depth_means[absolute_best_d_idx]
    depth_tolerance = 0.005 
    
    best_d_from_graph = depth_range[absolute_best_d_idx]
    for i, mean_score in enumerate(depth_means):
        if mean_score >= (best_d_score - depth_tolerance):
            best_d_from_graph = depth_range[i]
            break

    plt.figure(figsize=(10, 6))
    plt.errorbar(depth_range, depth_means, yerr=depth_stds, fmt='-o', label='Mean AUC ±1 std')
    plt.title(f'Cross-Validation: Max Depth (Selected Optimal: {best_d_from_graph})')
    plt.xlabel('max_depth')
    plt.ylabel('Mean ROC AUC')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'CV_Max_Depth_Plot.png'))
    plt.show()

    # 3. Hyperparameter Tuning: N Estimators (With Overfitting & Complexity Protection)
    n_test_range = [10, 50, 100, 200, 300, 400, 500, 600]
    n_means = []
    n_stds = []

    print(f"Analyzing N_Estimators with depth={best_d_from_graph}...")
    for n in n_test_range:
        fold_aucs = []
        for train_idx, val_idx in kfold.split(X_train_scaled):
            rf = RandomForestClassifier(n_estimators=n, max_depth=best_d_from_graph, min_samples_leaf=5, random_state=1907, n_jobs=-1)
            rf.fit(X_train_scaled[train_idx], np.ravel(y_train.iloc[train_idx]))
            probs = rf.predict_proba(X_train_scaled[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_train.iloc[val_idx], probs))
        n_means.append(np.mean(fold_aucs))
        n_stds.append(np.std(fold_aucs))

    # Tolerance Rule for N Estimators
    absolute_best_n_idx = np.argmax(n_means)
    best_n_score = n_means[absolute_best_n_idx]
    n_estimators_tolerance = 0.003 
    
    best_n_from_graph = n_test_range[absolute_best_n_idx]
    for i, mean_score in enumerate(n_means):
        if mean_score >= (best_n_score - n_estimators_tolerance):
            best_n_from_graph = n_test_range[i]
            break

    plt.figure(figsize=(10, 6))
    plt.errorbar(n_test_range, n_means, yerr=n_stds, fmt='-s', color='red', label='Mean AUC ±1 std')
    plt.title(f'Cross-Validation: N Estimators (Selected Optimal: {best_n_from_graph})')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean ROC AUC')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'CV_N_Estimators_Plot.png'))
    plt.show()

    # 4. Final Model Training
    print(f"Final training: n_estimators={best_n_from_graph}, max_depth={best_d_from_graph}")
    classifier = RandomForestClassifier(
        n_estimators=best_n_from_graph,
        max_depth=best_d_from_graph,
        min_samples_leaf=5,
        random_state=0,
        oob_score=True,
        n_jobs=-1
    )
    classifier.fit(X_train_scaled, y_train)

    # 5. Saving Artifacts using provided variable paths
    joblib.dump(classifier, model_path)
    joblib.dump(X_test_scaled, X_test_path) 
    joblib.dump(feature_names, feature_names_path) 
    joblib.dump(y_test, y_test_path)

    # 6. Evaluation
    y_pred = classifier.predict(X_test_scaled)
    testprobs = classifier.predict_proba(X_test_scaled)[:, 1]
    
    # Save predictions to CSV
    pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Probability': testprobs}).to_csv(prediction_results_path, index=False)

    final_auc = roc_auc_score(y_test, testprobs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')

    fpr, tpr, _ = roc_curve(y_test, testprobs)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {final_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'RandomForest_CM_ROC.png'), dpi=300) 
    plt.show()

    # 7. Learning Curve Plot
    print("Generating Learning Curve...")
    max_train_samples = int(len(X_train_scaled) * (4 / 5))
    train_sizes_param = np.linspace(10, max_train_samples, 10, dtype=int)
    
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=RandomForestClassifier(
            n_estimators=best_n_from_graph, 
            max_depth=best_d_from_graph, 
            min_samples_leaf=5, 
            random_state=1907, 
            n_jobs=-1
        ),
        X=X_train_scaled,
        y=y_train,
        train_sizes=train_sizes_param,
        cv=kfold,
        scoring='roc_auc',
        n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.figure(figsize=(11, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='#1f77b4', label='Training AUC')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.15, color='#1f77b4')
    plt.plot(train_sizes, validation_scores_mean, 'o-', color='#ff7f0e', label='Validation AUC')
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.15, color='#ff7f0e')
    
    plt.title(f'Learning Curve (Random Forest, depth={best_d_from_graph})', fontsize=13)
    plt.xlabel('Training Set Size', fontsize=11)
    plt.ylabel('AUC Score', fontsize=11)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'Learning_Curve_Plot.png'), dpi=300)
    plt.show()
    
    # Calculate variation (Standard Deviation) using Cross-Validation on the final model setup
    scoring_metrics = ['accuracy', 'recall', 'roc_auc', 'precision', 'f1']
    cv_results = cross_validate(
        estimator=RandomForestClassifier(
            n_estimators=best_n_from_graph,
            max_depth=best_d_from_graph,
            min_samples_leaf=5,
            random_state=1907,
            n_jobs=-1
        ),
        X=X_train_scaled,
        y=y_train,
        cv=kfold,
        scoring=scoring_metrics,
        n_jobs=-1
    )

    # Final Metrics Output with ± Variation (Standard Deviation)
    print("\n" + "—"*40)
    print(f"• Accuracy:  {np.mean(cv_results['test_accuracy']):.4f} ± {np.std(cv_results['test_accuracy']):.4f}")
    print(f"• Recall:    {np.mean(cv_results['test_recall']):.4f} ± {np.std(cv_results['test_recall']):.4f}")
    print(f"• AUC:       {np.mean(cv_results['test_roc_auc']):.4f} ± {np.std(cv_results['test_roc_auc']):.4f}")
    print(f"• Precision: {np.mean(cv_results['test_precision']):.4f} ± {np.std(cv_results['test_precision']):.4f}")
    print(f"• F1 Score:  {np.mean(cv_results['test_f1']):.4f} ± {np.std(cv_results['test_f1']):.4f}")
    print("—"*40)

if __name__ == "__main__":
    RandomForest_model(
        features_path = "../data/features.csv",
        prediction_results_path = "../result/predictions/predictions.csv",
        model_path = "../result/models/RandomForest_Model.pkl",
        feature_names_path = '../result/models/feature_names.pkl',
        X_test_path = '../result/models/X_test.pkl',
        y_test_path = '../result/models/Y_test.pkl',
    )