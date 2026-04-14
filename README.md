#  Le Generative AI en Computer Vision
### Expliqué comme si tu avais 5 ans

*Par **Ammar JERADA** *

![Computer Vision](https://img.shields.io/badge/Computer%20Vision-GenAI-purple?style=flat-square)
![Language](https://img.shields.io/badge/Langue-Français-blue?style=flat-square)

---

##  Table des matières

- [Introduction](#-introduction)
- [C'est quoi le Generative AI en CV ?](#-cest-quoi-le-generative-ai-en-computer-vision-)
- [Pourquoi utiliser le GenAI ?](#-pourquoi-utiliser-le-generative-ai-)
- [Les 3 architectures principales](#️-les-3-architectures-principales)
  - [GAN](#1--gan---generative-adversarial-network)
  - [VAE](#2--vae---variational-autoencoder)
  - [Diffusion Models](#3-️-diffusion-models)
- [Comparaison](#-comparaison-des-3-architectures)
- [Conclusion](#-conclusion)
- [Sources](#-sources)

---

##  Introduction

Imagine que tu regardes un million de photos de chats. Après ça, tu es capable de dessiner un chat que tu n'as jamais vu. Ton cerveau a **appris les règles** : 4 pattes, des oreilles pointues, des moustaches…

Le **Generative AI** fait exactement ça, mais avec un ordinateur.

---

##  C'est quoi le Generative AI en Computer Vision ?

Le **Generative AI** (IA Générative) en Computer Vision, c'est la capacité d'une machine à **créer de nouvelles images** — des visages, des paysages, de l'art — qui n'ont jamais existé, mais qui semblent réels.

Ce n'est pas de la magie. C'est des mathématiques.

Le modèle apprend à partir de millions d'images existantes pour comprendre **les règles cachées** : comment la lumière fonctionne sur un visage, comment les textures s'organisent, comment les couleurs se combinent.

> 💬 *"Plutôt que de copier une image, l'IA apprend à en inventer de nouvelles qui respectent les mêmes règles."*

### Exemples concrets

| Outil | Ce qu'il fait |
|---|---|
| **Midjourney / DALL-E** | Génèrent des images à partir de descriptions texte |
| **Face App** | Transforme un visage en le vieillissant |
| **Stable Diffusion** | Crée des images artistiques à la demande |
| **Super-résolution** | Améliore la qualité d'une image floue |

---

##  Pourquoi utiliser le Generative AI ?

| Problème | Solution GenAI |
|---|---|
| Tu n'as que 100 photos d'une maladie rare | GenAI génère 10 000 images supplémentaires réalistes |
| Tu veux tester une voiture sous la neige sans sortir | GenAI simule des milliers de scénarios |
| Tu veux restaurer une vieille photo abîmée | GenAI reconstruit les parties manquantes |
| Tu crées un jeu vidéo et tu as besoin de 500 textures | GenAI les génère en quelques secondes |

### Les 3 grands avantages

1.  **Créer des données** là où il n'en existe pas — très utile en médecine, en sécurité, en science
2.  **Augmenter la créativité** — artistes, designers, cinéastes utilisent ces outils
3.  **Réduire les coûts** — pas besoin de photographier ou d'illustrer chaque variante

---

##  Les 3 architectures principales

---

### 1.  GAN — Generative Adversarial Network

#### L'analogie : Le faussaire et le détective

Imagine un **faussaire** (le Générateur) qui essaie de peindre une copie d'un tableau de Picasso.
Et un **détective** (le Discriminateur) qui essaie de détecter si le tableau est un faux ou un vrai.

Au début, le faussaire est nul. Le détective les repère tous.
Mais le faussaire apprend de ses erreurs et s'améliore.
Le détective aussi doit s'améliorer.

Après des milliers de rounds, le faussaire est **si bon** que même le détective ne voit plus la différence.

```
Bruit aléatoire  →  Générateur  →  Fausse image ──┐
                                                    ├──→ Discriminateur → Vrai / Faux ?
Images réelles  ────────────────────────────────────┘
                             ↑
                  Retour d'erreur vers le Générateur
```

#### Comment ça marche

- Le **Générateur** commence avec du bruit aléatoire et essaie de créer des images réalistes
- Le **Discriminateur** reçoit des vraies ET fausses images, et apprend à les distinguer
- Le résultat d'erreur du Discriminateur est **renvoyé** au Générateur pour qu'il s'améliore

####  Avantages
- Produit des images **extrêmement réalistes** (ex : visages humains fake)
- Très rapide à l'utilisation une fois entraîné
- Utilisé dans StyleGAN, Pix2Pix, CycleGAN

####  Limites
- **Mode collapse** : le générateur trouve UNE seule image qui trompe le discriminateur et arrête de varier
- Très difficile à entraîner (l'équilibre entre les deux est délicat)
- Peu de contrôle sur ce que l'on génère

---

### 2.  VAE — Variational Autoencoder

#### L'analogie : Le résumé et le roman

Imagine que tu lis un roman de 500 pages sur un chat.
Tu en fais un résumé de 10 mots : *"Chat orange, paresseux, vit en appartement, aime le thon".*
Maintenant, quelqu'un lit ton résumé et réécrit un tout nouveau roman à partir de ces 10 mots.

C'est ça un VAE.

L'**Encodeur** fait le résumé → l'image devient quelques chiffres (= **espace latent**).
Le **Décodeur** relit le résumé → il génère une image à partir de ces chiffres.

```
Image → Encodeur → z (espace latent) → Décodeur → Image reconstruite/générée
                         ↑
              z est une distribution de probabilité
              Changer z légèrement = changer l'image légèrement
```

#### Comment ça marche

La **magie** du VAE : l'espace latent est **organisé et continu**.
Si tu changes légèrement `z`, l'image change légèrement aussi.

> `z = 0.5` → chat normal  
> `z = 0.9` → chat plus poilu  
> `z = 0.1` → chat plus petit  

####  Avantages
- Espace latent **interprétable et contrôlable**
- Permet d'interpoler entre deux images (chat → chien progressivement)
- Plus stable à entraîner qu'un GAN
- Utilisé pour la détection d'anomalies

####  Limites
- Images souvent **floues** (moins net que GAN ou Diffusion)
- Plus difficile de générer des images ultra-réalistes
- L'encodeur peut perdre des détails importants

---

### 3.  Diffusion Models

#### L'analogie : La sculpture dans le brouillard

Imagine une belle sculpture. Tu la couvres de brouillard, petit à petit, jusqu'à ce qu'on ne voit plus rien.

Un Diffusion Model apprend à faire **l'inverse** : partir du brouillard total et enlever le brouillard petit à petit, pour révéler une belle sculpture.

```
Entraînement :
Image nette → +bruit → +bruit → +bruit → Bruit pur  (t = 0 → 1000)

Génération :
Bruit pur → -bruit → -bruit → -bruit → Image nette  (t = 1000 → 0)
              ↑
   Réseau U-Net prédit quel bruit enlever à chaque étape
```

#### Comment ça marche

- **Phase 1 (entraînement)** : on prend des images et on y ajoute du bruit progressivement jusqu'à avoir du bruit pur
- **Phase 2 (génération)** : on part de bruit pur et on demande au réseau d'enlever le bruit, étape par étape

Le réseau apprend à chaque étape `t` :
> *"Étant donné cette image bruyante au step t, quel bruit dois-je enlever pour aller vers t-1 ?"*

Avec le **texte conditionnel** (DALL-E, Stable Diffusion) : la description textuelle guide chaque étape de débruitage → l'image correspond à la description.

####  Avantages
- **Meilleure qualité d'image** des trois architectures
- Très **contrôlable** (texte, style, composition)
- Stable à entraîner
- Utilisé dans : DALL-E 2/3, Stable Diffusion, Midjourney, Sora (vidéo)

####  Limites
- **Lent** : 1000 étapes de débruitage = long à générer
- Nécessite beaucoup de puissance de calcul
- Plus difficile à comprendre théoriquement

---

##  Comparaison des 3 architectures

| Critère |  GAN |  VAE |  Diffusion |
|---|:---:|:---:|:---:|
| **Idée centrale** | Faussaire vs Détective | Compression + Reconstruction | Enlever du bruit progressivement |
| **Contrôle** |  Difficile |  Bon |  Excellent |
| **Vitesse** |  Très rapide |  Rapide |  Lent |
| **Stabilité d'entraînement** |  Difficile |  Moyen |  Facile |
| **Usage principal** | Visages, transfert de style | Anomalie, interpolation | Génération texte → image |
| **Modèles connus** | StyleGAN, CycleGAN | Détection médicale | DALL-E, Midjourney |

---

##  Conclusion

Le Generative AI en Computer Vision, c'est une révolution dans la façon dont les ordinateurs **comprennent et créent des images**.

- Les **GANs** nous ont appris que deux réseaux en compétition peuvent créer quelque chose d'extraordinaire
- Les **VAEs** nous ont appris qu'on peut comprendre une image en la compressant en quelques chiffres significatifs
- Les **Diffusion Models** nous ont appris qu'apprendre à défaire le chaos permet de créer de l'ordre

Ces trois familles ne s'excluent pas : elles sont souvent **combinées** dans les systèmes modernes.

La prochaine fois que tu utilises DALL-E ou Midjourney, tu sauras que derrière l'image générée, il y a des milliers d'étapes de débruitage guidées par ta description… ou peut-être un faussaire qui essaie de tromper son détective intérieur. 

---

##  Sources

- Goodfellow et al. (2014) — *Generative Adversarial Networks*
- Kingma & Welling (2013) — *Auto-Encoding Variational Bayes*
- Ho et al. (2020) — *Denoising Diffusion Probabilistic Models (DDPM)*
