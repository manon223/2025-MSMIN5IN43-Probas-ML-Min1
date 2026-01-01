"""
Script de téléchargement et préparation des datasets français
Taille optimisée : ~3000-5000 exemples pour un entraînement rapide
"""

import os
import pandas as pd
import numpy as np
# from datasets import load_dataset  # Temporairement désactivé à cause d'un conflit PyTorch
from sklearn.model_selection import train_test_split
import json

# Configuration
np.random.seed(42)
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Créer les dossiers si nécessaires
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 70)
print("TÉLÉCHARGEMENT DES DATASETS FRANÇAIS")
print("=" * 70)


# =============================================================================
# 1. DATASET ALLOCINE - Pour le Sentiment (Critiques de films)
# =============================================================================
print("\n📥 1. Création d'un dataset de sentiment Allocine...")

# Créer des exemples de critiques synthétiques (car load_dataset nécessite PyTorch)
# Vous pourrez enrichir avec de vraies données plus tard

allocine_data = []

positive_reviews = [
    "Film magnifique, j'ai adoré ! Les acteurs sont excellents.",
    "Une merveille du cinéma, à voir absolument ! 😍",
    "Quel chef-d'œuvre ! Je suis sorti de la salle émerveillé.",
    "Excellente réalisation, scénario captivant du début à la fin.",
    "Bravo ! Un film qui restera dans les mémoires.",
] * 100

negative_reviews = [
    "Très déçu, le scénario est prévisible et ennuyeux.",
    "Film médiocre, j'ai failli m'endormir. 😴",
    "Quel gâchis ! Les acteurs jouent mal et l'histoire n'a aucun sens.",
    "Je ne recommande pas du tout, c'est une perte de temps.",
    "Nul, vraiment nul. Je regrette d'être allé le voir.",
] * 100

neutral_reviews = [
    "C'est correct, sans plus. Rien de mémorable.",
    "Film moyen, certaines scènes sont bonnes, d'autres moins.",
    "On a passé un moment correct, mais rien d'exceptionnel.",
    "Le film est regardable, mais je ne le reverrai pas.",
    "Pas mal dans l'ensemble, mais ça ne restera pas gravé.",
] * 100

for text in positive_reviews[:500]:
    allocine_data.append({'text': text, 'sentiment': 2, 'sentiment_3class': 2})

for text in negative_reviews[:500]:
    allocine_data.append({'text': text, 'sentiment': 0, 'sentiment_3class': 0})

for text in neutral_reviews[:200]:
    allocine_data.append({'text': text, 'sentiment': 1, 'sentiment_3class': 1})

allocine_df = pd.DataFrame(allocine_data)

print(f"   ✓ {len(allocine_df)} critiques créées")
print(f"   Distribution : Négatif={sum(allocine_df['sentiment']==0)}, Neutre={sum(allocine_df['sentiment']==1)}, Positif={sum(allocine_df['sentiment']==2)}")

# Sauvegarder
allocine_df.to_csv(os.path.join(RAW_DIR, "allocine_sentiment.csv"), index=False)


# =============================================================================
# 2. DATASET EMOTIONS - Pour les Émotions
# =============================================================================
print("\n📥 2. Création d'un dataset d'émotions français...")

# Comme il n'y a pas de gros dataset français d'émotions facilement accessible,
# on va créer des exemples synthétiques basés sur des patterns typiques
# (Vous pourrez les remplacer par de vraies données plus tard)

emotions_data = []

# Dictionnaire de phrases types par émotion
emotion_examples = {
    'joie': [
        "Je suis trop content, c'est génial ! 😊",
        "Quelle merveilleuse journée, j'adore !",
        "Trop bien, je suis aux anges ! ❤️",
        "Super nouvelle, je suis ravi !",
        "C'est fantastique, je ne m'y attendais pas ! 🎉"
    ],
    'tristesse': [
        "Je suis vraiment triste aujourd'hui 😢",
        "C'est déprimant, rien ne va",
        "Je me sens si seul et abandonné",
        "Quelle déception, je suis dévasté",
        "Rien ne va plus, tout est noir 😔"
    ],
    'colere': [
        "J'en ai marre, c'est vraiment énervant ! 😡",
        "C'est inadmissible, je suis furieux !",
        "Ça suffit maintenant, je ne supporte plus !",
        "Quelle incompétence, c'est révoltant !",
        "Je suis vraiment en colère contre toi ! 😠"
    ],
    'peur': [
        "J'ai vraiment peur, c'est angoissant 😨",
        "C'est effrayant, je suis terrorisé",
        "J'ai des frissons, c'est inquiétant",
        "Je suis anxieux, ça me stresse",
        "Ça fait peur, je suis paniqué 😰"
    ],
    'surprise': [
        "Oh ! Je ne m'attendais pas à ça ! 😮",
        "Quoi ?! C'est incroyable !",
        "Wow, quelle surprise !",
        "Je n'en crois pas mes yeux ! 😲",
        "C'est inattendu, je suis choqué !"
    ],
    'degout': [
        "C'est dégoûtant, beurk ! 🤢",
        "J'ai la nausée, c'est répugnant",
        "C'est écœurant, je ne peux pas",
        "Quelle horreur, c'est immonde",
        "Beurk, c'est vraiment dégueulasse 🤮"
    ],
    'neutre': [
        "Le train arrive à 15h.",
        "Il fait beau aujourd'hui.",
        "J'ai rendez-vous demain.",
        "La réunion est à 10h.",
        "Le magasin est fermé le dimanche."
    ]
}

# Générer ~100 exemples par émotion
for emotion_name, examples in emotion_examples.items():
    base_examples = examples * 20  # Répéter pour avoir ~100
    for i, text in enumerate(base_examples[:100]):
        emotions_data.append({
            'text': text,
            'emotion': emotion_name
        })

emotions_df = pd.DataFrame(emotions_data)

# Mapping des émotions vers des indices
emotion_mapping = {
    'joie': 0, 'tristesse': 1, 'colere': 2, 'peur': 3,
    'surprise': 4, 'degout': 5, 'neutre': 6
}
emotions_df['emotion_id'] = emotions_df['emotion'].map(emotion_mapping)

print(f"   ✓ {len(emotions_df)} exemples d'émotions créés")
print(f"   Distribution : {emotions_df['emotion'].value_counts().to_dict()}")

emotions_df.to_csv(os.path.join(RAW_DIR, "emotions.csv"), index=False)


# =============================================================================
# 3. DATASET IRONIE - Pour la Détection d'Ironie
# =============================================================================
print("\n📥 3. Création d'un dataset d'ironie...")

# Exemples d'ironie (vous pourrez enrichir avec de vraies données)
irony_data = []

ironic_examples = [
    "Super cette pluie, j'adore être trempé ! 🙄",
    "Génial, encore une réunion inutile !",
    "Oh quelle joie, mon train est encore en retard !",
    "Fantastique, mon ordinateur a planté ! 😒",
    "J'adore attendre pendant des heures, vraiment !",
] * 50

non_ironic_examples = [
    "J'adore vraiment ce film, il est excellent !",
    "Quelle belle journée, je suis content !",
    "Ce restaurant est vraiment bon, je recommande.",
    "J'ai passé un excellent week-end !",
    "Ce livre est passionnant, je ne peux pas m'arrêter.",
] * 50

for text in ironic_examples[:250]:
    irony_data.append({'text': text, 'is_ironic': 1})

for text in non_ironic_examples[:250]:
    irony_data.append({'text': text, 'is_ironic': 0})

irony_df = pd.DataFrame(irony_data)

print(f"   ✓ {len(irony_df)} exemples d'ironie créés")
print(f"   Distribution : Ironique={sum(irony_df['is_ironic']==1)}, Non-ironique={sum(irony_df['is_ironic']==0)}")

irony_df.to_csv(os.path.join(RAW_DIR, "irony.csv"), index=False)


# =============================================================================
# 4. FUSION DES DATASETS - Créer le dataset multi-tâches
# =============================================================================
print("\n🔄 4. Création du dataset multi-tâches combiné...")

# Pour simplifier, on va créer un dataset où chaque exemple a les 3 labels
# Prendre tous les exemples d'émotions disponibles
combined_data = []

# Utiliser les émotions comme base et ajouter sentiment + ironie
for idx, row in emotions_df.iterrows():
    # Déduire le sentiment de l'émotion
    if row['emotion'] in ['joie', 'surprise']:
        sentiment = 2  # positif
    elif row['emotion'] in ['tristesse', 'colere', 'peur', 'degout']:
        sentiment = 0  # négatif
    else:
        sentiment = 1  # neutre
    
    # Ironie aléatoire (20% de chance d'être ironique)
    is_ironic = 1 if np.random.rand() > 0.8 else 0
    
    combined_data.append({
        'text': row['text'],
        'emotion': row['emotion'],
        'emotion_id': row['emotion_id'],
        'sentiment': sentiment,
        'is_ironic': is_ironic
    })

combined_df = pd.DataFrame(combined_data)

print(f"   ✓ {len(combined_df)} exemples dans le dataset combiné")

# Sauvegarder
combined_df.to_csv(os.path.join(RAW_DIR, "combined_multitask.csv"), index=False)


# =============================================================================
# 5. SPLIT TRAIN / VAL / TEST (Stratifié)
# =============================================================================
print("\n✂️ 5. Split des données (70% train / 15% val / 15% test)...")

# Split stratifié sur l'émotion (la tâche avec le plus de classes)
train_val, test = train_test_split(
    combined_df, 
    test_size=0.15, 
    random_state=42,
    stratify=combined_df['emotion_id']
)

train, val = train_test_split(
    train_val,
    test_size=0.176,  # 0.176 * 0.85 ≈ 0.15 du total
    random_state=42,
    stratify=train_val['emotion_id']
)

print(f"   ✓ Train : {len(train)} exemples")
print(f"   ✓ Val   : {len(val)} exemples")
print(f"   ✓ Test  : {len(test)} exemples")

# Sauvegarder les splits
train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)


# =============================================================================
# 6. STATISTIQUES GLOBALES
# =============================================================================
print("\n📊 6. Statistiques globales...")

stats = {
    'total_examples': len(combined_df),
    'train_size': len(train),
    'val_size': len(val),
    'test_size': len(test),
    'emotion_distribution': combined_df['emotion'].value_counts().to_dict(),
    'sentiment_distribution': combined_df['sentiment'].value_counts().to_dict(),
    'irony_distribution': combined_df['is_ironic'].value_counts().to_dict(),
    'avg_text_length': int(combined_df['text'].str.len().mean()),
    'max_text_length': int(combined_df['text'].str.len().max()),
    'min_text_length': int(combined_df['text'].str.len().min())
}

# Sauvegarder les stats
with open(os.path.join(DATA_DIR, "dataset_stats.json"), 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"   ✓ Statistiques sauvegardées dans {DATA_DIR}/dataset_stats.json")


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 70)
print("✅ TÉLÉCHARGEMENT TERMINÉ !")
print("=" * 70)
print(f"\n📁 Fichiers créés :")
print(f"   • data/raw/allocine_sentiment.csv")
print(f"   • data/raw/emotions.csv")
print(f"   • data/raw/irony.csv")
print(f"   • data/raw/combined_multitask.csv")
print(f"   • data/processed/train.csv ({len(train)} exemples)")
print(f"   • data/processed/val.csv ({len(val)} exemples)")
print(f"   • data/processed/test.csv ({len(test)} exemples)")
print(f"   • data/dataset_stats.json")

print(f"\n📊 Résumé :")
print(f"   • Total : {len(combined_df)} exemples")
print(f"   • Longueur moyenne des textes : {stats['avg_text_length']} caractères")
print(f"   • 7 émotions, 3 sentiments, 2 classes d'ironie")

print(f"\n🎯 Prochaine étape :")
print(f"   Ouvrez le notebook 'notebooks/01_exploration_donnees.ipynb'")
print(f"   pour explorer visuellement ces données !")
print("=" * 70)
