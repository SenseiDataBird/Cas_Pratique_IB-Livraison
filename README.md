# Cas_Pratique_IB — Livraison

### Ce dépôt contient le matériel du cas pratique "Livraison".
---
#### Votre rôle : ajuster la **planification** (Airflow) du projet ImmoBird, en cohérence avec le cadrage.
---
#### Votre mission : **lire** le repo puis **corriger la planification** (jour/heure/catchup/...) directement dans le code du DAG.
---

## Contexte de planification

### Suite à la phase de cadrage et à la validation technique des scripts Python...

### Vous vous êtes concerté avec l'équipe Produit d'ImmoBird, qui demande une industrialisation du projet.

#### **Vous avez donc réalisé un plan d'orchestration que le DevOps de votre équipe à retranscrit techniquement sur Git, Docker et Airflow.**

*Vous constatez plusieurs incohérences simples entre le cadrage et l'implémentation actuelle :*

### Calendrier (jour / heure) :
- [ ] **Jour/Heure** — Le DAG doit se déclencher **tous les lundis à 09:00**.  
      Question : la valeur de `schedule_interval` est-elle `0 9 * * 1` (lundi) et non `0 9 * * 2` (mardi) ?

- [ ] **Description lisible** — La `description` du DAG **mentionne clairement** :  
      “Planifié **Lundi 09:00**”.  
      Question : la description reflète-t-elle bien le **lundi 09:00** ?

- [ ] **Commentaires alignés** — Les commentaires du fichier **ne se contredisent pas** (plus de “mardi” si le cron est lundi).  
      Question : les commentaires et le cron disent-ils **la même chose** ?

### Robustesse (retries / délais) :
- [ ] **Tentatives d’échec** — En cas d’échec, on veut **3 tentatives**.  
      Question : `default_args["retries"] == 3` ?

- [ ] **Délai entre tentatives** — Le délai doit être **10 minutes**.  
      Question : `default_args["retry_delay"] == timedelta(minutes=10)` ?

### Historique (rattrapage) :
- [ ] **Rattrapage activé** — “Aucun trigger ne doit manquer” : on **rattrape** les exécutions manquées.  
      Question : `catchup == True` pour garder une **trace** de chaque run ?

- [ ] **Ancrage contrôlé** — Pour éviter l’“avalanche” d’anciens runs, la `start_date` doit être **récente** (par ex. le **lundi** de la semaine en cours à **09:00**).  
      Question : `start_date` pointe-t-elle vers un **lundi récent 09:00** plutôt que **2024-01-01** ?

### Preuve / lisibilité :
- [ ] **Bloc “Planification”** — Un **commentaire court** en haut du DAG résume :
      - Jour/heure : **Lundi 09:00**
      - `retries=3`, `retry_delay=10min`
      - `catchup=True`, `start_date=<lundi récent 09:00>`
      Question : ce bloc est-il présent et **à jour** ?

#### Dernière étape : Vérification des rapports d'exécution automatique disponible sous `Git/repport_to_evaluate/...` :

*À l'aide de ses rapports fréquents, vous serez capable de visualiser les performances techniques du modèle et de faire des recommandations !*

**La boucle est bouclée.**

## Livrables attendus

1) **Lecture** : cochez chaque point de la checklist et notez les écarts repérés.  
2) **Modification** : corrigez le fichier `pipeline_immo.py` en conséquence (via l’UI GitHub → *Edit* → *Commit changes*).
