# 🎯 Résumé - Projet NLP : Analyse de Sentiment Multi-dimensionnelle

## 📌 Vue d'Ensemble

**Objectif** : Créer un système d'analyse de sentiment avancé qui détecte non seulement le positif/négatif, mais aussi les émotions fines (joie, colère, peur, etc.) et l'ironie dans des textes français (tweets, commentaires).

**Catégorie** : Machine Learning Avancé & Deep Learning

---

## 🛠️ Technologies Principales

- **CamemBERT** : Modèle BERT spécialisé pour le français
- **HuggingFace Transformers** : Framework pour les modèles
- **PyTorch** : Framework deep learning
- **Python 3.8+**

---

## 📁 Structure du Projet

```
projet-nlp/
├── data/                  # Données brutes et traitées
├── notebooks/             # Exploration et expérimentations
├── src/                   # Code source
│   ├── data/             # Chargement et preprocessing
│   ├── models/           # Modèles (CamemBERT multi-tâches)
│   ├── training/         # Entraînement
│   └── evaluation/       # Métriques et analyse
├── models/               # Modèles sauvegardés
└── results/              # Résultats et visualisations
```

---

## 🎯 Les 3 Tâches à Résoudre

### 1. Classification d'Émotions (7 classes)
- Joie, Tristesse, Colère, Peur, Surprise, Dégoût, Neutre

### 2. Analyse de Sentiment (3 classes)
- Positif, Négatif, Neutre

### 3. Détection d'Ironie (2 classes)
- Ironique, Non-ironique

---

## 📊 Méthodologie en 5 Phases

### **Phase 1 : Données**
1. Collecter des datasets français (DEFT, Allocine, GoEmotions traduit)
2. Nettoyer les textes (URLs, mentions) **MAIS garder emojis et ponctuation !**
3. Séparer : 70% train / 15% validation / 15% test

**Key point** : Stratifier le split pour garder la distribution des classes

### **Phase 2 : Exploration**
1. Statistiques descriptives (longueur des textes, distribution des classes)
2. Visualisations (word clouds, distributions)
3. Identifier les déséquilibres de classes

### **Phase 3 : Baseline**
1. TF-IDF + Logistic Regression
2. Établir la performance minimum à battre
3. F1-Score attendu : ~0.50-0.60

**Pourquoi ?** Pour prouver que le deep learning apporte vraiment un gain

### **Phase 4 : CamemBERT Multi-tâches**

**Architecture** :
```
Texte → CamemBERT (encodeur partagé) → 3 têtes de classification
                                      ├─→ Tête Émotions (7 classes)
                                      ├─→ Tête Sentiment (3 classes)
                                      └─→ Tête Ironie (2 classes)
```

**Entraînement** :
- Learning rate : 2e-5 (encodeur) / 1e-4 (têtes)
- Batch size : 16 (ou 8 avec gradient accumulation)
- Époques : 3-5
- Dropout : 0.3
- Early stopping sur validation F1

**Loss** : Somme pondérée des 3 tâches
```
Loss_totale = Loss_émotion + 0.5 × Loss_sentiment + 0.3 × Loss_ironie
```

### **Phase 5 : Évaluation**
1. Métriques : **F1-Score** (macro et weighted), Accuracy, Precision, Recall
2. Matrices de confusion par tâche
3. Analyse des erreurs (regarder 50-100 exemples mal classés)
4. Visualisations (courbes d'apprentissage, t-SNE des embeddings)

---

## 📈 Objectifs de Performance

| Tâche | Métrique | Objectif Minimum | Objectif Optimal |
|-------|----------|------------------|------------------|
| Émotions | F1 (macro) | 0.65 | 0.75+ |
| Sentiment | Accuracy | 0.80 | 0.88+ |
| Ironie | F1 | 0.60 | 0.70+ |

---

## ⚡ Points Critiques à Ne Pas Rater

### ✅ À FAIRE ABSOLUMENT

1. **Fixer les seeds** pour la reproductibilité
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   ```

2. **Stratifier** le split train/val/test

3. **Garder les emojis** dans le preprocessing (porteurs d'émotion !)

4. **Utiliser F1-Score** comme métrique principale (pas accuracy)

5. **Faire une baseline simple** avant CamemBERT

6. **Early stopping** pour éviter l'overfitting

7. **Analyser les erreurs** (pas juste reporter les chiffres)

### ❌ À NE PAS FAIRE

1. ❌ Prétraiter avant de splitter (data leakage)
2. ❌ Supprimer les emojis ou la ponctuation excessive
3. ❌ Se fier uniquement à l'accuracy avec classes déséquilibrées
4. ❌ Tuner les hyperparamètres sur le test set
5. ❌ Oublier de documenter l'environnement
6. ❌ Ignorer les déséquilibres de classes

---

## 🔧 Solutions aux Problèmes Courants

### Problème : Classes déséquilibrées
**Solution** : 
- Class weighting dans la loss
- Ou over-sampling (SMOTE)

### Problème : Overfitting
**Solution** :
- Dropout (0.3-0.5)
- Early stopping (patience 2-3 époques)
- Data augmentation (back-translation)

### Problème : GPU insuffisant
**Solution** :
- Gradient accumulation (batch effectif = 8 × 4 = 32)
- Google Colab gratuit
- Réduire la longueur max des séquences (128 tokens)

### Problème : Ironie difficile à détecter
**Solution** :
- Utiliser un dataset spécialisé ironie
- Attention aux emojis (🙄, 😏) et ponctuation (!!!, ???)
- C'est normal que ce soit la tâche la plus difficile

---

## 📝 Livrables Finaux

### 1. Code
- Scripts Python bien documentés
- Notebooks Jupyter clairs
- README avec instructions

### 2. Modèle
- Modèle final sauvegardé (.pt)
- Fichier de config

### 3. Présentation (15-20 slides)
- Contexte → Méthode → Résultats → Discussion

- Introduction et problématique
- Méthodologie
- Résultats avec tableaux et graphiques
- Analyse d'erreurs
- Discussion et limites
- Conclusion

### 4. Démo
- Interface Gradio/Streamlit pour tester le modèle

---

## 📅 Planning Recommandé

| Période | Tâches |
|---------|--------|
| **Jour 1-2** | Setup + collecte données + exploration |
| **Jour 3** | Preprocessing + statistiques |
| **Jour 4** | Baseline (TF-IDF + ML) |
| **Jour 5-6** | Implémentation CamemBERT multi-tâches |
| **Jour 7-9** | Fine-tuning + optimisation hyperparamètres |
| **Jour 10-11** | Évaluation complète + analyse erreurs |
| **Jour 12-14** | Rédaction rapport |
| **Jour 15** | Présentation + démo |

---

## 📚 Ressources Essentielles

### Datasets
- DEFT 2017 (tweets français)
- Allocine (critiques films)
- GoEmotions traduit (émotions)

### Outils
- Google Colab (GPU gratuit)
- Weights & Biases (tracking expériences)
- Gradio (démo rapide)

---

## Checklist Finale

**Données** :
- [ ] Corpus collecté (>5000 exemples)
- [ ] Preprocessing validé
- [ ] Split train/val/test stratifié

**Modèles** :
- [ ] Baseline implémentée et évaluée
- [ ] CamemBERT multi-tâches fonctionnel
- [ ] Hyperparamètres optimisés
- [ ] Objectifs de performance atteints

**Évaluation** :
- [ ] F1-Score calculé pour chaque tâche
- [ ] Matrices de confusion générées
- [ ] Analyse d'erreurs réalisée
- [ ] Visualisations créées

**Livrables** :
- [ ] Code documenté et testé
- [ ] Rapport rédigé
- [ ] Présentation prête
- [ ] Démo fonctionnelle (optionnel)

**Rigueur** :
- [ ] Reproductibilité garantie
- [ ] Environnement documenté
- [ ] Résultats validés
- [ ] Limites discutées

---

## En Résumé : Les 3 Choses Essentielles

1. **Méthodologie rigoureuse** : Baseline → Exploration → Modèle avancé → Analyse
2. **CamemBERT multi-tâches** : Un encodeur, trois têtes de classification
3. **Évaluation critique** : F1-Score, matrices de confusion, analyse d'erreurs


---
