# architecture-experimentale-de-reseau-neuronal-distributed
Architecture expérimentale de réseau neuronal distribué


# 🧠 Réseau neuronal synchrone bidirectionnel avec CUDA

> Réseau neuronal expérimental construit en **Go**, organisé sous forme de **Nodes communicants par channels synchrones**, avec un **anneau bidirectionnel Forward / Backward** et une accélération des calculs numériques par **CUDA sur GPU**.

---

## 📌 Présentation

Ce projet explore une manière différente de construire un réseau neuronal.

Au lieu de regrouper toute la logique dans une fonction classique comme :

```text
train()
    ↓
forward()
    ↓
loss()
    ↓
backpropagation()
    ↓
update()
```

le réseau est construit comme un **ensemble de Nodes indépendants**.

Chaque Node réalise une étape précise du traitement et communique avec les autres grâce à des **channels Go non bufferisés**.

Le réseau forme ainsi un **anneau synchrone bidirectionnel** :

```text
                         FORWARD
                            ↓

       ┌──────────┐
       │  INPUT   │
       └────┬─────┘
            │
            ▼
     ┌──────────────┐
     │  NEURONNE    │
     └──────┬───────┘
            │
            ▼
       ┌─────────┐
       │   SUM   │
       └────┬────┘
            │
            ▼
       ┌─────────┐
       │  BIAIS  │
       └────┬────┘
            │
            ▼
     ┌────────────┐
     │  SIGMOID   │
     └──────┬─────┘
            │
            ▼
       ┌─────────┐
       │ OUTPUT  │
       └────┬────┘
            │
            │
            ▼
        ERREUR
            │
            │
            ▼
       BACKWARD
            │
            ▼
     SIGMOID → BIAIS → SUM → NEURONNE → INPUT
```

Le même message `Neuronne` effectue donc un **tour complet** dans le réseau.

---

# 🏗️ Architecture générale

Le projet est composé de plusieurs niveaux.

```text
                 ┌─────────────────────────────┐
                 │          APPLICATION        │
                 │             Go              │
                 └──────────────┬──────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │   Anneau neuronal   │
                    │                     │
                    │ Input → ... → Output│
                    │ ↑                  ↓ │
                    │ └──── Backward ────┘ │
                    └──────────┬────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
              CPU                         GPU
                                           │
                                           ▼
                                         CUDA
```

Il existe donc une séparation entre :

* **communication**
* **structure du réseau**
* **calcul neuronal**
* **apprentissage**
* **accélération matérielle**

---

# 🔄 L'anneau bidirectionnel

## Pourquoi un anneau ?

Chaque liaison entre deux Nodes est bidirectionnelle.

Elle possède deux channels :

```go
type Liaison struct {
    Forward  chan Neuronne
    Backward chan Neuronne
}
```

Les channels sont créés sans buffer :

```go
make(chan Neuronne)
```

Cela signifie que la communication est **synchrone**.

Un Node qui écrit dans le channel attend qu'un autre Node soit prêt à recevoir.

---

## Forward

Pendant le Forward, le message avance :

```text
Input
  │
  ▼
Neuronne
  │
  ▼
Sum
  │
  ▼
Biais
  │
  ▼
Sigmoid
  │
  ▼
Output
```

Le message contient notamment :

```go
type Neuronne struct {
    Entrees []float64
    Cible   []float64

    HiddenZ []float64
    HiddenA []float64

    OutputZ []float64
    OutputA []float64

    DeltaHidden []float64
    DeltaOutput []float64

    Erreur float64

    Backward bool
}
```

Il transporte donc progressivement les informations nécessaires au calcul et à l'apprentissage.

---

# 🔙 Backward

Lorsque `Node_Output` obtient le résultat, il calcule l'erreur.

Le message change alors de direction :

```go
n.Backward = true
```

Il repart dans l'autre sens :

```text
Output
   │
   ▼
Sigmoid
   │
   ▼
Biais
   │
   ▼
Sum
   │
   ▼
Neuronne
   │
   ▼
Input
```

Le Backward permet alors de calculer les gradients et de modifier les paramètres du réseau.

---

# 🧬 Le réseau neuronal

Le modèle actuel est un réseau :

```text
2 → 2 → 1
```

Il possède :

```text
2 entrées
   ↓
2 neurones cachés
   ↓
1 neurone de sortie
```

Visuellement :

```text
        Couche cachée
       ┌───────┐
x₁ ───►│   h₁  │───┐
       └───────┘   │
                   ├──► y
       ┌───────┐   │
x₂ ───►│   h₂  │───┘
       └───────┘
```

Ce petit réseau est suffisant pour expérimenter avec l'apprentissage du **XOR**.

---

# 🧩 Les Nodes

## `Node_Input`

Responsabilité :

* recevoir un exemple ;
* injecter l'exemple dans l'anneau ;
* récupérer le message après le Backward.

Il ne réalise aucun calcul neuronal.

```text
extérieur
   │
   ▼
Node_Input
   │
   ▼
anneau
```

---

## `Node_Neuronne`

`Node_Neuronne` est volontairement très simple.

Il **ne réalise pas toute la logique du neurone**.

Il ne calcule pas :

* la somme pondérée ;
* le biais ;
* la fonction d'activation ;
* les gradients ;
* les mises à jour.

Son rôle est de **préparer et représenter la couche neuronale** pour les étapes suivantes.

```text
Input
  │
  ▼
Node_Neuronne
  │
  ▼
Node_Sum
```

Cette séparation permet de ne pas transformer un Node en bloc monolithique.

---

# ➕ `Node_Sum`

`Node_Sum` réalise la partie linéaire :

$$
z_i = \sum_j x_j w_{ij} + b_i
$$

Pour la couche cachée :

```text
x₁ ──┐
     ├──► somme pondérée ──► z₁
x₂ ──┘

x₁ ──┐
     ├──► somme pondérée ──► z₂
x₂ ──┘
```

Dans la version CUDA, cette opération est exécutée sur le GPU.

---

# ➕ `Node_Biais`

Le biais est un paramètre indépendant associé à chaque neurone.

Mathématiquement :

$$
z = Wx + b
$$

Pendant le Backward, le biais est également mis à jour :

$$
b_{new} = b_{old} - \eta \delta
$$

où :

* `η` est le taux d'apprentissage ;
* `δ` est le gradient.

---

# σ `Node_Sigmoid`

La fonction d'activation utilisée actuellement est la sigmoid :

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Elle transforme la valeur du neurone en une valeur comprise entre `0` et `1`.

Sa dérivée est :

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

Elle est particulièrement importante pendant le Backward.

Le Node utilise donc CUDA pour :

```text
Forward
    ↓
Sigmoid

Backward
    ↓
Dérivée de Sigmoid
    ↓
Gradient
```

---

# 🎯 `Node_Output`

`Node_Output` représente la sortie du réseau.

Il reçoit :

```text
prediction
```

et connaît :

```text
target
```

L'erreur actuelle utilise une fonction MSE :

$$
E = \frac{1}{2}(y-\hat y)^2
$$

avec :

* `y` : valeur prédite ;
* `ŷ` : valeur attendue.

Après avoir calculé l'erreur, `Node_Output` déclenche le Backward.

---

# 🔙 Rétropropagation

Le gradient de sortie est calculé avec :

$$
\delta_{out}
=
(\hat y-y)
\cdot
\sigma'(\hat y)
$$

Puis le gradient est propagé vers la couche cachée :

$$
\delta_h =
\delta_{out}
\cdot
W_{out}
\cdot
\sigma'(h)
$$

Les paramètres sont ensuite modifiés.

Pour les poids :

$$
W_{new}
=
W_{old}
-
\eta
\frac{\partial E}{\partial W}
$$

Pour les biais :

$$
b_{new}
=
b_{old}
-
\eta
\frac{\partial E}{\partial b}
$$

---

# 🧠 Apprentissage

L'apprentissage suit donc ce cycle :

```text
          ┌───────────────────────┐
          │                       │
          ▼                       │
       INPUT                      │
          │                       │
          ▼                       │
       FORWARD                    │
          │                       │
          ▼                       │
        OUTPUT                   │
          │                       │
          ▼                       │
        ERREUR                   │
          │                       │
          ▼                       │
       BACKWARD                  │
          │                       │
          ▼                       │
      GRADIENTS                  │
          │                       │
          ▼                       │
   MISE À JOUR PARAMÈTRES        │
          │                       │
          └───────────────────────┘
```

Chaque exemple fait donc un tour complet dans l'anneau avant que l'exemple suivant ne soit envoyé.

Cela conserve le caractère **synchrone** du modèle.

---

# ⚡ CUDA

Le CPU ne réalise pas nécessairement tous les calculs.

Les opérations numériques importantes peuvent être déléguées au GPU.

L'architecture est :

```text
             Go
              │
              │ CGO
              ▼
          API C/CUDA
              │
              ▼
          CUDA Runtime
              │
              ▼
             GPU
              │
              ▼
        CUDA Kernels
```

Le fichier CUDA contient notamment des kernels pour :

```text
linear
sigmoid
update_weights
update_bias
```

---

# 🚀 Pourquoi utiliser CUDA ?

Un CPU possède relativement peu de cœurs généralistes.

Un GPU possède un très grand nombre de cœurs adaptés à l'exécution massive d'opérations similaires.

Un calcul neuronal ressemble souvent à :

$$
Wx
$$

avec énormément de multiplications et additions indépendantes.

Cela correspond particulièrement bien au calcul parallèle.

Par exemple :

```text
CPU

w₀x₀ ──┐
w₁x₁ ──┼──► addition
w₂x₂ ──┤
w₃x₃ ──┘
```

Sur GPU, de nombreux éléments peuvent être calculés en parallèle :

```text
GPU

thread 0 → w₀x₀
thread 1 → w₁x₁
thread 2 → w₂x₂
thread 3 → w₃x₃
thread 4 → w₄x₄
...
```

---

# 🔌 CUDA n'est pas un Node

C'est une distinction importante dans cette architecture.

CUDA ne remplace pas :

```text
Node_Sum
Node_Biais
Node_Sigmoid
```

CUDA est le **moteur d'exécution** utilisé par ces Nodes.

On peut représenter cela ainsi :

```text
                 NODE_SUM
                    │
                    ▼
              ┌───────────┐
              │   CUDA    │
              └─────┬─────┘
                    │
                    ▼
                   GPU
```

Le Node conserve donc sa responsabilité logique.

Le GPU fournit simplement la puissance de calcul.

---

# 🧱 Architecture logicielle

Le projet est organisé autour de trois niveaux.

## Niveau 1 — Communication

```go
type Liaison struct {
    Forward  chan Neuronne
    Backward chan Neuronne
}
```

Responsabilité :

```text
Transport synchrone
```

---

## Niveau 2 — Nodes

```text
Node_Input
Node_Neuronne
Node_Sum
Node_Biais
Node_Sigmoid
Node_Output
```

Responsabilité :

```text
Logique du réseau
```

---

## Niveau 3 — Calcul matériel

```text
CPU
 │
 └── Go

GPU
 │
 └── CUDA
```

Responsabilité :

```text
Exécution numérique
```

---

# 🔄 Vue complète

```text
                         ┌───────────────────┐
                         │       GPU         │
                         │      CUDA         │
                         └───────▲───────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    │          Calcul         │
                    │            │            │
                    │            ▼            │
                    │       Node_Sum          │
                    │       Node_Biais        │
                    │       Node_Sigmoid      │
                    │                         │
                    └─────────────────────────┘


                         ANNEAU BIDIRECTIONNEL

                              FORWARD
                                 ↓

                    ┌──────────────────┐
                    │    Node_Input    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Node_Neuronne   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Node_Sum     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Node_Biais    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Node_Sigmoid    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Node_Output    │
                    └────────┬─────────┘
                             │
                           ERREUR
                             │
                             ▼

                    ┌──────────────────┐
                    │  Node_Sigmoid    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Node_Biais    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Node_Sum     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Node_Neuronne   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Node_Input    │
                    └──────────────────┘

                              BACKWARD
```

---

# 🔒 Synchronisation

Les channels sont **non bufferisés**.

Par exemple :

```go
Forward: make(chan Neuronne)
Backward: make(chan Neuronne)
```

Il n'existe donc pas de file d'attente entre deux Nodes.

Un envoi :

```go
sortie.Forward <- n
```

attend qu'un Node soit prêt à recevoir.

Cette propriété donne au réseau un comportement de **pipeline synchrone**.

---

# 🔁 Un exemple complet

Pour l'exemple XOR :

```text
Entrée :
[0, 1]

Cible :
[1]
```

le message effectue :

```text
[0,1]
  │
  ▼
Input
  │
  ▼
Neuronne
  │
  ▼
Sum ─────────► GPU
  │
  ▼
Biais ───────► GPU
  │
  ▼
Sigmoid ─────► GPU
  │
  ▼
Output
  │
  ▼
ŷ = 0.73
  │
  ▼
Erreur
  │
  ▼
Sigmoid ─────► GPU
  │
  ▼
Biais ───────► GPU
  │
  ▼
Sum ─────────► GPU
  │
  ▼
Neuronne
  │
  ▼
Input
```

Les paramètres sont alors légèrement modifiés.

L'opération est répétée sur les différents exemples :

```text
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

Au fil des itérations, le réseau ajuste ses poids et ses biais.

---

# 📁 Organisation du projet

```text
xor-cuda/
│
├── main.go
│
├── cuda_nn.h
│
├── cuda_nn.cu
│
├── Makefile
│
└── go.mod
```

### `main.go`

Contient :

* les Nodes ;
* les channels ;
* l'anneau ;
* le modèle ;
* l'entraînement ;
* l'interface Go/CUDA.

### `cuda_nn.h`

Déclare l'API CUDA accessible depuis Go.

### `cuda_nn.cu`

Contient :

* les kernels CUDA ;
* la gestion de la mémoire GPU ;
* les opérations exécutées sur le GPU.

### `Makefile`

Compile :

```text
CUDA → .so
       ↓
     Go + CGO
```

---

# 🛠️ Compilation

Vérifier CUDA :

```bash
nvcc --version
```

Puis :

```bash
nvidia-smi
```

Compiler :

```bash
make
```

Exécuter :

```bash
make run
```

Pour une NVIDIA RTX A6000, l'architecture CUDA correspondante est `sm_86`.

---

# ⚠️ Limite actuelle

Le modèle actuel utilise CUDA de manière volontairement simple.

Pour chaque opération, le prototype peut effectuer :

```text
CPU
 ↓
cudaMemcpy
 ↓
GPU
 ↓
kernel
 ↓
cudaMemcpy
 ↓
CPU
```

C'est excellent pour **démontrer l'architecture**, mais ce n'est pas encore optimal pour les performances.

Pour un vrai réseau de grande taille, cette architecture devra évoluer vers :

```text
CPU
 │
 │ transfert initial
 ▼
GPU
 │
 ├── Forward
 │
 ├── Forward
 │
 ├── Forward
 │
 ├── Backward
 │
 ├── Backward
 │
 └── mise à jour
 │
 ▼
CPU
```

Autrement dit, les données et paramètres devront rester **résidents en VRAM** pendant plusieurs opérations.

---

# 🚀 Évolutions possibles

Le prototype actuel permet ensuite d'expérimenter avec :

### 📦 Batchs

Au lieu de :

```text
un exemple
```

utiliser :

```text
batch
 ├── exemple 1
 ├── exemple 2
 ├── exemple 3
 ├── ...
 └── exemple N
```

Cela permettrait au GPU de travailler sur beaucoup plus de données simultanément.

### 🧮 Grandes couches

Passer de :

```text
2 → 2 → 1
```

à :

```text
784 → 1024 → 1024 → 512 → 10
```

ou davantage.

### 🧠 Différentes fonctions d'activation

Ajouter :

```text
ReLU
Tanh
Sigmoid
Softmax
GELU
```

### ⚡ Mémoire GPU persistante

Éviter les transferts :

```text
CPU ↔ GPU
```

à chaque opération.

### 🔀 Plusieurs GPU

L'architecture par Nodes pourrait permettre d'imaginer :

```text
Node_Sum_1     → GPU 0
Node_Sum_2     → GPU 1
Node_Sigmoid   → GPU 0
Node_Backward  → GPU 1
```

### 🌐 Nodes distribués

À terme, la même idée pourrait même être étendue à :

```text
Node
 │
 ├── channel local
 │
 ├── GPU local
 │
 └── communication réseau
```

Le Node deviendrait alors une unité de calcul indépendante.

---

# 🎯 Philosophie du projet

Le but n'est pas uniquement de reproduire un réseau neuronal classique.

Le projet expérimente une autre manière de voir un réseau neuronal :

```text
       Communication
             +
       Calcul distribué
             +
       Apprentissage
             +
       Accélération GPU
```

Le réseau peut être considéré comme un **système de Nodes communicants**, où :

* les channels représentent les liaisons ;
* le Forward représente la propagation de l'information ;
* le Backward représente la propagation de l'apprentissage ;
* les Nodes représentent les étapes de traitement ;
* CUDA représente le moteur de calcul parallèle ;
* le GPU fournit la puissance de calcul ;
* le message `Neuronne` circule dans l'ensemble du système.

---

# 🧠 Résumé

```text
                  RÉSEAU NEURONAL
                         │
                         ▼
              ┌────────────────────┐
              │   ANNEAU SYNCHRONE │
              └─────────┬──────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
           FORWARD             BACKWARD
              │                   │
              ▼                   ▼
        Calcul neuronal       Gradients
              │                   │
              └─────────┬─────────┘
                        │
                        ▼
                     CUDA
                        │
                        ▼
                       GPU
```

**L'idée centrale :**

> Le réseau neuronal n'est pas une grosse fonction. C'est un ensemble de Nodes spécialisés, reliés par des communications synchrones bidirectionnelles, capables de faire circuler aussi bien les données que l'apprentissage, tandis que les calculs lourds peuvent être délégués au GPU via CUDA.
