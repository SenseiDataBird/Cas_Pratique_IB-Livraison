# =======================================================================================================================================
# ImmoBird - Pipeline Complet de Modélisation
# Prétraitement → Entraînement XGBoost → Rapport
# ---------------------------------------------------------------------------------------------------------------------------------------
# Pipeline pédagogique pour CDP Data & IA : démonstration bout-en-bout
# Sans red flags, propre, commenté pour apprentissage
# =======================================================================================================================================

import os                          # Gestion des chemins et fichiers système
import hashlib                     # Calcul de checksums pour détection de drift
import json                        # Sérialisation des données pour rapports
from datetime import datetime      # Horodatage des exécutions

import pandas as pd                                                 # Manipulation des données tabulaires
import numpy as np                                                  # Calculs numériques et métriques
from sklearn.preprocessing import StandardScaler, OneHotEncoder     # Normalisation et encodage
from sklearn.model_selection import train_test_split                # Division train/test
from sklearn.metrics import mean_squared_error, r2_score            # Métriques d'évaluation
import xgboost as xgb                                               # Algorithme de machine learning (gradient boosting)
import joblib                                                       # Sérialisation des modèles ML


# =======================================================================================================================================
# CONFIGURATION GLOBALE
# =======================================================================================================================================

DATA_DIR = "/opt/airflow/data"      # Répertoire des données (brutes et transformées)
REPORT_DIR = "/opt/airflow/reports" # Répertoire des rapports générés


# =======================================================================================================================================
# FONCTIONS UTILITAIRES
# =======================================================================================================================================

def sha256_file(path: str) -> str:
    """Calcule le hash SHA256 d'un fichier pour détecter les changements de données."""
    h = hashlib.sha256()                               # Initialisation du hash SHA256
    with open(path, "rb") as f:                    # Ouverture en mode binaire
        for chunk in iter(lambda: f.read(8192), b""):  # Lecture par chunks de 8KB
            h.update(chunk)                            # Mise à jour du hash
    return h.hexdigest()                               # Retour du hash en hexadécimal


# =======================================================================================================================================
# ÉTAPE 1 : PRÉTRAITEMENT DES DONNÉES
# =======================================================================================================================================

def preprocess_data():
    """Étape 1: Nettoyage, transformation et préparation des données brutes pour le ML."""
    print("=== ÉTAPE 1: PRÉTRAITEMENT ===")
    
    # Chargement des données brutes depuis le fichier CSV
    df = pd.read_csv(os.path.join(DATA_DIR, 'house_pred.csv'))
    df_ml = df.copy()                              # Copie de travail pour éviter de modifier l'original
    
    # Diagnostic : vérification des valeurs manquantes
    print("Valeurs manquantes par colonne :")
    print(df_ml.isnull().sum())
    
    # Stratégie de traitement des valeurs manquantes
    # Pour les colonnes numériques : remplacement par la médiane (robuste aux outliers)
    numeric_columns = ['Area', 'YearBuilt']
    for col in numeric_columns:
        median_value = df_ml[col].median()         # Calcul de la médiane
        df_ml[col].fillna(median_value, inplace=True)  # Remplacement des NaN
    
    # Pour les colonnes catégorielles : remplacement par le mode (valeur la plus fréquente)
    categorical_columns = ['Location', 'Garage', 'Condition']
    for col in categorical_columns:
        mode_value = df_ml[col].mode()[0]          # Récupération de la valeur la plus fréquente
        df_ml[col].fillna(mode_value, inplace=True)    # Remplacement des NaN
    
    # Normalisation des variables numériques (centrer-réduire)
    scaler = StandardScaler()                      # Initialisation du normalisateur
    df_ml[['Area', 'YearBuilt']] = scaler.fit_transform(df_ml[['Area', 'YearBuilt']])
    
    # Encodage des variables catégorielles en variables binaires (One-Hot Encoding)
    ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' évite la multicolinéarité
    all_cat_encoded = ohe.fit_transform(df_ml[['Location', 'Garage', 'Condition']])
    
    # Construction des noms de colonnes pour les variables encodées
    location_categories = ohe.categories_[0][1:]   # Récupération des catégories (sans la première)
    garage_categories = ohe.categories_[1][1:]
    condition_categories = ohe.categories_[2][1:]
    new_columns = [f'Location_{cat}' for cat in location_categories] + \
                  [f'Garage_{cat}' for cat in garage_categories] + \
                  [f'Condition_{cat}' for cat in condition_categories]
    
    # Création d'un DataFrame avec les variables encodées
    encoded_df = pd.DataFrame(all_cat_encoded, columns=new_columns)
    
    # Fusion des données : suppression des colonnes catégorielles originales + ajout des encodées
    df_ml = pd.concat([
        df_ml.drop(['Location', 'Garage', 'Condition'], axis=1),  # Suppression des originales
        encoded_df                                 # Ajout des variables binaires
    ], axis=1)
    
    # Sauvegarde du jeu de données prétraité pour l'étape suivante
    output_path = os.path.join(DATA_DIR, 'house_pred_for_ml.csv')
    df_ml.to_csv(output_path, index=False)         # Sauvegarde sans l'index
    
    # Affichage des résultats pour validation
    print("\nPrétraitement terminé !")
    print("\nAperçu des données prétraitées :")
    print(df_ml.head())                            # Affichage des 5 premières lignes
    print("\nInformations sur les colonnes :")
    print(df_ml.info())                            # Structure du DataFrame
    
    return df_ml


# =======================================================================================================================================
# ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE
# =======================================================================================================================================

def train_model():
    """Étape 2: Entraînement d'un modèle XGBoost et évaluation des performances."""
    print("\n=== ÉTAPE 2: ENTRAÎNEMENT ===")
    
    # Chargement des données prétraitées depuis l'étape précédente
    df = pd.read_csv(os.path.join(DATA_DIR, 'house_pred_for_ml.csv'))
    
    # Séparation des variables explicatives (X) et de la variable cible (y)
    X = df.drop('Price', axis=1)                   # Features : toutes les colonnes sauf Price
    y = df['Price']                                # Target : la colonne Price à prédire
    
    # Division stratifiée du dataset (80% entraînement, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,                             # 20% pour le test
        random_state=42                            # Reproductibilité des résultats
    )
    
    # Configuration et initialisation du modèle XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,                          # Nombre d'arbres (plus = mieux mais plus lent)
        learning_rate=0.1,                         # Taux d'apprentissage (compromis vitesse/précision)
        max_depth=5,                               # Profondeur max des arbres (évite le surapprentissage)
        min_child_weight=1,                        # Poids minimum par feuille (régularisation)
        subsample=0.8,                             # Fraction d'échantillons par arbre (évite le surapprentissage)
        colsample_bytree=0.8,                      # Fraction de features par arbre (diversification)
        random_state=42                            # Reproductibilité
    )
    
    # Entraînement du modèle sur les données d'entraînement
    model.fit(X_train, y_train)                   # Apprentissage des patterns prix/features
    
    # Prédiction sur l'ensemble de test (données non vues pendant l'entraînement)
    y_pred = model.predict(X_test)                # Prédictions du modèle
    
    # Calcul des métriques de performance
    mse = mean_squared_error(y_test, y_pred)      # Erreur quadratique moyenne
    rmse = np.sqrt(mse)                           # Racine de l'erreur quadratique (même unité que Price)
    r2 = r2_score(y_test, y_pred)                 # Coefficient de détermination (% variance expliquée)
    
    # Affichage des résultats d'évaluation
    print("\nRésultats de l'évaluation du modèle :")
    print(f"RMSE : {rmse:.2f}")                   # Plus c'est bas, mieux c'est
    print(f"R² : {r2:.4f}")                       # Plus c'est proche de 1, mieux c'est
    
    # Sauvegarde des métriques dans un fichier pour traçabilité
    metrics_path = os.path.join(DATA_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"RMSE: {rmse:.2f}\nR2: {r2:.4f}\n")  # Format structuré pour parsing
    
    # Analyse de l'importance des variables (quelles features influencent le plus le prix)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,                      # Noms des features
        'Importance': model.feature_importances_   # Scores d'importance XGBoost
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)  # Tri décroissant
    print("\nImportance des features :")
    print(feature_importance)                     # Aide à comprendre les drivers de prix
    
    # Persistance du modèle entraîné pour réutilisation future
    model_path = os.path.join(DATA_DIR, 'immo_model.joblib')
    joblib.dump(model, model_path)                # Sérialisation du modèle
    
    # Démonstration : exemple de prédiction sur un cas réel
    test_features = X_test.iloc[0].to_dict()                           # Premier exemple du jeu de test
    predicted_price = model.predict(pd.DataFrame([test_features]))[0]  # Prédiction
    actual_price = y_test.iloc[0]                                      # Prix réel correspondant 
    
    print("\nExemple de prédiction :")
    print(f"Prix prédit : {predicted_price:.2f}")
    print(f"Prix réel : {actual_price:.2f}")
    
    return rmse, r2, feature_importance


# =======================================================================================================================================
# ÉTAPE 3 : GÉNÉRATION DE RAPPORT
# =======================================================================================================================================

def generate_report():
    """Étape 3: Génération d'un rapport Markdown avec métriques et recommandations CDP."""
    print("\n=== ÉTAPE 3: RAPPORT ===")
    
    # Création du répertoire de rapports s'il n'existe pas
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Récupération des métadonnées d'exécution depuis les variables d'environnement Airflow
    dag_run_id = os.environ.get("AIRFLOW_CTX_DAG_RUN_ID", "manual")                               # ID de l'exécution DAG
    execution_date = os.environ.get("AIRFLOW_CTX_EXECUTION_DATE", datetime.utcnow().isoformat())  # Date d'exécution
    image = os.environ.get("IMAGE_NAME", "immo-demo:latest")                                      # Image Docker utilisée
    git_sha = os.environ.get("GIT_SHA", "unknown")                                                # Hash du code source
    
    # Définition des chemins vers les fichiers de données
    raw_csv = os.path.join(DATA_DIR, "house_pred.csv")                      # Données brutes
    ml_csv = os.path.join(DATA_DIR, "house_pred_for_ml.csv")                # Données prétraitées
    metrics_path = os.path.join(DATA_DIR, "metrics.txt")                    # Métriques du modèle
    
    # Chargement des datasets pour analyse
    df_raw = pd.read_csv(raw_csv)                                           # Dataset original
    df_ml = pd.read_csv(ml_csv)                                             # Dataset transformé
    
    # Calcul des métadonnées du dataset
    dataset_size_bytes = os.path.getsize(raw_csv)                                      # Taille du fichier en bytes
    num_features = df_ml.shape[1] - 1 if "Price" in df_ml.columns else df_ml.shape[1]  # Nombre de features
    
    # Analyse de qualité des données
    missing_counts = df_raw.isnull().sum().to_dict()                        # Comptage des valeurs manquantes
    target = df_raw.get("Price")                                            # Variable cible
    target_mean = float(target.mean()) if target is not None else None      # Moyenne des prix
    target_std = float(target.std()) if target is not None else None        # Écart-type des prix
    
    # Calcul du checksum pour détecter les changements de données (data drift)
    checksum = sha256_file(raw_csv)                                         # Hash du fichier de données
    drift_score = float(abs((target_mean or 0) / (target_std or 1)))        # Score de drift simplifié
    
    # Lecture des métriques de performance du modèle
    rmse = None                                                             # Root Mean Square Error
    r2 = None                                                               # Coefficient de détermination
    if os.path.exists(metrics_path):                                        # Si le fichier de métriques existe
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:                                                  # Parsing ligne par ligne
                if line.startswith("RMSE:"):                                # Extraction du RMSE
                    rmse = float(line.split(":")[1].strip())
                if line.startswith("R2:") or line.startswith("R²:"):        # Extraction du R²
                    try:
                        r2 = float(line.split(":")[1].strip())
                    except Exception:
                        pass                                                # Gestion des erreurs de parsing
    
    # Identification des top-5 features (simplifiée pour la démo)
    top5_features = [c for c in df_ml.columns if c != "Price"][:5]          # Premières features (hors Price)
    
    # Génération du nom de fichier rapport avec timestamp
    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")                   # Format: YYYYMMDD_HHMMSS
    report_name = f"report_{now_str}_{dag_run_id}.md"                       # Nom unique du rapport
    report_path = os.path.join(REPORT_DIR, report_name)                     # Chemin complet
    
    # Recommandations pro-forma pour les CDP (exemples pédagogiques)
    decisions = [
        "à surveiller: stabilité de la variable cible",                     # Monitoring continu
        "action recommandée: planifier un rafraîchissement des données",    # Maintenance données
        "prochain incrément: ajouter des features temporelles",             # Évolution modèle
    ]
    
    # Construction du contenu Markdown du rapport
    content = []
    content.append(f"# Rapport d'exécution — {dag_run_id}")                # Titre avec ID d'exécution
    content.append("")
    content.append("## Métadonnées")                                       # Section métadonnées
    content.append(f"- dag_run_id: `{dag_run_id}`")                        # Identifiant de l'exécution
    content.append(f"- date/heure: `{execution_date}`")                    # Horodatage
    content.append(f"- image Docker: `{image}`")                           # Version de l'environnement
    content.append(f"- hash du code (GIT_SHA): `{git_sha}`")               # Version du code
    content.append(f"- taille dataset (bytes): `{dataset_size_bytes}`")    # Taille des données
    content.append(f"- nb de features: `{num_features}`")                  # Dimensionnalité
    content.append("")
    content.append("## Qualité & dérive")                                                      # Section qualité données
    content.append(f"- nb de valeurs manquantes (raw): `{json.dumps(missing_counts)}`")        # Données manquantes
    content.append(f"- cible (Price) — moyenne: `{target_mean}`, écart-type: `{target_std}`")  # Stats cible
    content.append(f"- checksum CSV: `{checksum}`")                                            # Empreinte des données
    content.append(f"- drift score (pédagogique): `{drift_score:.4f}`")                        # Indicateur de dérive
    content.append("")
    content.append("## Résultats modèle")                                  # Section performance
    content.append(f"- RMSE: `{rmse}`")                                    # Erreur de prédiction
    content.append(f"- R²: `{r2}`")                                        # Qualité d'ajustement
    content.append(f"- Top-5 features: `{top5_features}`")                 # Variables importantes
    content.append("")
    content.append("## Décisions CDP (pro-forma)")                          # Section recommandations
    for d in decisions:                                                     # Ajout de chaque recommandation
        content.append(f"- {d}")
    
    # Écriture du rapport dans le fichier Markdown
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")                                 # Assemblage et sauvegarde
    
    print(f"Report generated at: {report_path}")                           # Confirmation de génération
    
    # Fonctionnalité optionnelle : push vers un repository GitHub
    repo_url = os.environ.get("REPORTS_REPO_URL")                              # URL du repo (si définie)
    repo_branch = os.environ.get("REPORTS_REPO_BRANCH", "main")                # Branche cible
    if repo_url:                                                               # Si un repo est configuré
        os.system("git config --global user.email 'airflow-bot@example.com'")  # Config Git
        os.system("git config --global user.name 'airflow-bot'")
        os.chdir("/opt/airflow")                                                # Changement de répertoire
        os.system("git init")                                                   # Initialisation Git
        os.system("git checkout -B " + repo_branch)                             # Création/switch branche
        os.system("git add reports/")                                           # Ajout des rapports
        os.system("git commit -m 'Add report '" + report_name + " || true")     # Commit (ignore échecs)
        os.system("git remote remove origin 2>/dev/null || true")               # Suppression remote existant
        os.system("git remote add origin '" + repo_url + "'")                   # Ajout nouveau remote
        os.system("git push -u origin " + repo_branch + " --force")             # Push forcé


# =======================================================================================================================================
# ORCHESTRATION PRINCIPALE
# =======================================================================================================================================

def main():
    """Orchestration complète du pipeline : prétraitement → entraînement → rapport."""
    print("DÉMARRAGE DU PIPELINE IMMOBIRD MODELING")
    print("="*80)
    
    try:
        # Étape 1: Nettoyage et préparation des données brutes
        print("🔄 Lancement de l'étape 1...")
        df_processed = preprocess_data()                    # Transformation des données brutes
        
        # Étape 2: Entraînement du modèle de machine learning
        print("🔄 Lancement de l'étape 2...")
        rmse, r2, feature_importance = train_model()        # Entraînement et évaluation XGBoost
        
        # Étape 3: Génération du rapport de pilotage pour les CDP
        print("🔄 Lancement de l'étape 3...")
        generate_report()                                   # Création du rapport Markdown
        
        # Résumé final pour validation
        print("\n" + "="*80)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print(f"📊 Métriques finales: RMSE={rmse:.2f}, R²={r2:.4f}")  # Affichage des performances
        print("📋 Rapport généré dans le répertoire /opt/airflow/reports/")
        print("="*80)
        
    except Exception as e:
        # Gestion des erreurs avec information détaillée
        print("\n" + "="*80)
        print(f"❌ ERREUR DANS LE PIPELINE: {e}")
        print("🔍 Vérifiez les logs ci-dessus pour diagnostiquer le problème")
        print("="*80)
        raise                                               # Re-lancement de l'exception pour Airflow


# =======================================================================================================================================
# POINT D'ENTRÉE DU SCRIPT
# =======================================================================================================================================

if __name__ == "__main__":
    # Exécution du pipeline complet si le script est lancé directement
    main()
