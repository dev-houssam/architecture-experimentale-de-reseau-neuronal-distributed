# Réseau neuronal distribué Go + CUDA

## 1. Vision du projet

Ce projet expérimente une architecture de réseau neuronal dans laquelle :

* **Go** orchestre les nœuds de calcul ;
* les **goroutines** représentent les nœuds du réseau ;
* des **channels synchrones** forment un anneau bidirectionnel ;
* **CUDA** réalise les opérations numériques lourdes ;
* les données intermédiaires restent autant que possible **résidentes sur le GPU** ;
* le réseau ne fait plus circuler de gros tableaux entre les nœuds ;
* les nœuds font circuler de petites **références de calcul** permettant d'identifier les données à traiter.

L'objectif n'est donc pas simplement de « mettre du CUDA dans un réseau neuronal Go ».

L'objectif est de séparer clairement :

```text
                    PLAN DE CONTRÔLE
                         Go
                          │
                          │
              ┌───────────▼───────────┐
              │       Anneau          │
              │                       │
              │  Jetons / Références  │
              │  États / Synchronisation
              └───────────┬───────────┘
                          │
                          │ ID
                          ▼
                    PLAN DE DONNÉES
                         CUDA
                          │
              ┌───────────▼───────────┐
              │         VRAM          │
              │                       │
              │ Entrées               │
              │ Hidden Z              │
              │ Hidden A              │
              │ Output Z              │
              │ Output A              │
              │ Delta Output          │
              │ Delta Hidden          │
              │ Poids                 │
              │ Biais                 │
              └───────────────────────┘
```

---

# 2. Le problème du modèle CPU → GPU → CPU

Une première implémentation peut naturellement faire ceci :

```text
CPU
 │
 │ données
 ▼
GPU
 │
 │ résultat
 ▼
CPU
 │
 │ nouvelles données
 ▼
GPU
```

Cela fonctionne, mais devient rapidement inefficace lorsque chaque opération CUDA provoque :

* une allocation GPU ;
* une copie CPU → GPU ;
* un calcul ;
* une copie GPU → CPU ;
* une libération GPU.

Le problème n'est alors plus nécessairement le calcul neuronal lui-même.

Le problème devient le **déplacement permanent des données**.

---

# 3. Nouvelle approche : les données restent sur le GPU

Dans cette architecture, les données numériques sont conservées dans un espace mémoire GPU.

Le CPU ne transporte plus les tenseurs entre les Nodes.

Il transporte principalement une référence :

```text
Reference
    │
    ├── ID
    ├── état
    └── séquence
```

Par exemple :

```go
type Reference struct {
    ID       uint64
    Etat     Etat
    Sequence uint64
}
```

Le message qui traverse le ring peut donc être extrêmement petit.

Au lieu de transmettre :

```text
[0.32, 0.81, 0.12, 0.73, ...]
```

on transmet :

```text
ID = 42
Etat = HIDDEN_A
```

Le Node comprend alors :

> « Je dois travailler sur les données associées à l'identifiant 42, actuellement dans l'état HIDDEN_A. »

---

# 4. Le registre GPU

Les identifiants sont associés aux ressources présentes sur le GPU par un registre.

Conceptuellement :

```text
              GPU Registry
           ┌─────────────────┐
           │ ID 42            │
           │                  │
           │ → Entrées        │
           │ → HiddenZ        │
           │ → HiddenA        │
           │ → OutputZ        │
           │ → OutputA        │
           │ → DeltaOutput    │
           │ → DeltaHidden    │
           └─────────────────┘
```

Un autre calcul peut utiliser :

```text
ID 43
```

sans interférer avec :

```text
ID 42
```

Le registre devient ainsi une couche d'abstraction entre les Nodes Go et les pointeurs réels présents en VRAM.

---

# 5. Pourquoi utiliser un identifiant plutôt qu'un pointeur brut ?

Le système pourrait théoriquement faire circuler un pointeur CUDA.

Cependant, le projet privilégie une référence logique :

```go
type Reference struct {
    ID uint64
}
```

plutôt qu'un :

```go
unsafe.Pointer
```

L'identifiant permet au gestionnaire GPU de conserver le contrôle sur les ressources.

Une entrée du registre peut conceptuellement contenir :

```text
ID 42
 │
 ├── adresse GPU
 ├── taille
 ├── type
 ├── état actuel
 ├── ressources associées
 └── synchronisation
```

Le Node n'a donc pas besoin de connaître directement l'organisation physique de la VRAM.

---

# 6. Le jeton de calcul

Le message circulant dans l'anneau devient un **jeton de calcul**.

```go
type Jeton struct {
    ID       uint64
    Etat     Etat
    Sequence uint64
}
```

Le jeton ne contient pas nécessairement les données du réseau.

Il indique simplement :

1. quelle instance de données doit être traitée ;
2. dans quel état elle se trouve ;
3. éventuellement quelle séquence de calcul elle appartient.

On peut donc voir le ring comme un système faisant circuler des **instructions de progression** plutôt que des données volumineuses.

---

# 7. L'automate à états

Le réseau neuronal devient également un automate distribué.

Chaque Node réalise une transition :

```text
État entrant
     │
     ▼
  opération
     │
     ▼
État sortant
```

Par exemple :

```text
ENTREES
   │
   │ SumHidden
   ▼
HIDDEN_Z
   │
   │ BiaisHidden
   ▼
HIDDEN_Z_BIASED
   │
   │ SigmoidHidden
   ▼
HIDDEN_A
   │
   │ SumOutput
   ▼
OUTPUT_Z
   │
   │ BiaisOutput
   ▼
OUTPUT_Z_BIASED
   │
   │ SigmoidOutput
   ▼
OUTPUT_A
```

Puis le retour d'apprentissage :

```text
OUTPUT_A
   │
   │ erreur
   ▼
DELTA_OUTPUT
   │
   │ rétropropagation
   ▼
DELTA_HIDDEN
   │
   │ mise à jour
   ▼
POIDS
```

Chaque Node devient donc responsable d'une transition bien définie.

---

# 8. Exemple d'états

L'automate peut être représenté par un type énuméré :

```go
type Etat uint8

const (
    EtatEntrees Etat = iota

    EtatHiddenZ
    EtatHiddenBiais
    EtatHiddenA

    EtatOutputZ
    EtatOutputBiais
    EtatOutputA

    EtatDeltaOutput
    EtatDeltaHidden

    EtatTermine
)
```

Le nombre exact d'états pourra évoluer avec l'architecture du réseau.

L'important est que l'état indique **quelles données sont actuellement valides et quelle opération peut être effectuée ensuite**.

---

# 9. Les Nodes

Les Nodes ne transportent plus les matrices numériques.

Ils manipulent une référence vers une ressource GPU.

## Node_Input

Le Node_Input :

* crée ou récupère une entrée ;
* place les données initiales dans la VRAM ;
* obtient un identifiant ;
* injecte le jeton dans l'anneau.

```text
CPU
 │
 │ création de l'exemple
 ▼
GPU
 │
 │ ID = 42
 ▼
Node_Input
 │
 ▼
Jeton #42
```

---

## Node_Neuronne

`Node_Neuronne` ne réalise pas toute la logique neuronale.

Il sert principalement à préparer ou organiser les ressources nécessaires au traitement.

Il peut notamment vérifier que les buffers nécessaires existent :

```text
ID 42
 │
 ├── Entrées
 ├── HiddenZ
 ├── HiddenA
 ├── OutputZ
 ├── OutputA
 ├── DeltaOutput
 └── DeltaHidden
```

Il fait ensuite progresser le jeton.

---

## Node_Sum

Le Node_Sum réalise les produits matriciels.

Exemple :

```text
HiddenZ = W_hidden × X
```

ou :

```text
OutputZ = W_output × HiddenA
```

Le calcul est exécuté par CUDA.

Mais le Node ne reçoit pas `X`.

Il reçoit :

```text
ID = 42
```

et demande au gestionnaire GPU les ressources correspondant à cette référence.

---

## Node_Biais

Le Node_Biais ajoute les biais directement dans les buffers GPU.

```text
HiddenZ += BiasHidden
```

ou :

```text
OutputZ += BiasOutput
```

Aucune copie intermédiaire des vecteurs n'est nécessaire.

---

## Node_Sigmoid

Le Node_Sigmoid applique l'activation directement sur le GPU :

```text
HiddenA = sigmoid(HiddenZ)
```

puis :

```text
OutputA = sigmoid(OutputZ)
```

Le jeton passe simplement à l'état suivant.

---

## Node_Output

Le Node_Output termine la propagation avant.

Il calcule l'erreur et déclenche ensuite le retour dans l'autre sens de l'anneau.

```text
Forward
───────►
Input
  ↓
...
  ↓
Output
  │
  │ erreur
  ▼
Backward
◄───────
```

---

# 10. Anneau bidirectionnel

Le réseau conserve une communication bidirectionnelle.

Chaque liaison possède deux directions :

```go
type Liaison struct {
    Forward  chan Jeton
    Backward chan Jeton
}
```

Les channels restent synchrones :

```go
make(chan Jeton)
```

Le calcul avance donc par synchronisation entre les Nodes.

Architecture :

```text
                    FORWARD
                       ───────►

       ┌───────────────────────────────────────┐
       │                                       │
       ▼                                       │
     Input → Neuronne → Sum → Biais → Sigmoid │
       │                                       │
       │                                       ▼
       │                                  Sum Output
       │                                       │
       │                                       ▼
       │                                  Biais Output
       │                                       │
       │                                       ▼
       │                                  Sigmoid Output
       │                                       │
       │                                       ▼
       │                                     Output
       │                                       │
       └───────────────────────────────────────┘
                    ◄───────
                     BACKWARD
```

Le même jeton peut donc traverser le réseau dans les deux directions.

---

# 11. Forward

Pour un exemple `X` :

```text
ID = 42
```

le jeton avance :

```text
ID 42
EtatEntrees
      │
      ▼
SumHidden
      │
      ▼
HiddenZ
      │
      ▼
BiaisHidden
      │
      ▼
HiddenZ_Biased
      │
      ▼
SigmoidHidden
      │
      ▼
HiddenA
      │
      ▼
SumOutput
      │
      ▼
OutputZ
      │
      ▼
BiaisOutput
      │
      ▼
OutputZ_Biased
      │
      ▼
SigmoidOutput
      │
      ▼
OutputA
```

Les grandes quantités numériques ne quittent pas la VRAM.

---

# 12. Backward

Après le calcul de l'erreur :

```text
OutputA
   │
   ▼
DeltaOutput
```

le même identifiant repart dans l'autre sens :

```text
DeltaOutput
     │
     ▼
mise à jour biais output
     │
     ▼
calcul DeltaHidden
     │
     ▼
mise à jour poids output
     │
     ▼
DeltaHidden
     │
     ▼
mise à jour biais hidden
     │
     ▼
mise à jour poids hidden
     │
     ▼
Input
```

La boucle est alors complète.

---

# 13. Plusieurs exemples peuvent coexister

Cette architecture permet de ne plus limiter le GPU à un seul exemple.

On peut avoir :

```text
ID 41 → exemple XOR #1
ID 42 → exemple XOR #2
ID 43 → exemple XOR #3
ID 44 → exemple XOR #4
```

Tous les exemples peuvent rester en VRAM.

Le ring fait circuler :

```text
#41
#42
#43
#44
```

plutôt que leurs vecteurs.

Cela ouvre naturellement la voie au traitement par lots.

```text
             GPU
 ┌───────────────────────────────┐
 │                               │
 │ ID 41 ─┐                      │
 │ ID 42 ─┼──► Batch CUDA        │
 │ ID 43 ─┤                      │
 │ ID 44 ─┘                      │
 │                               │
 └───────────────────────────────┘
```

Le modèle de communication peut donc rester identique alors que l'implémentation CUDA évolue vers le batching.

---

# 14. Séparation contrôle / données

L'architecture repose finalement sur deux niveaux.

## Plan de contrôle

Géré par Go :

```text
Goroutines
Channels
Jetons
États
Synchronisation
Ordonnancement
```

## Plan de données

Géré par CUDA :

```text
Tenseurs
Poids
Biais
Activations
Gradients
Buffers
VRAM
```

Cette séparation est fondamentale.

```text
                 CPU / Go
        ┌─────────────────────┐
        │                     │
        │  « traite ID 42 »   │
        │          │          │
        └──────────┼──────────┘
                   │
                   │ référence
                   ▼
        ┌─────────────────────┐
        │         CUDA        │
        │                     │
        │  données de l'ID 42 │
        │                     │
        └─────────────────────┘
```

---

# 15. Principe fondamental

Le principe directeur du projet devient :

> **On ne déplace pas les données quand on peut déplacer une référence vers les données.**

Le ring ne transporte donc pas le calcul.

Il transporte **l'identité et l'état du calcul**.

CUDA possède les données et réalise les opérations numériques.

Go orchestre la progression.

---

# 16. Architecture globale

```text
                         GO
          ┌─────────────────────────────┐
          │                             │
          │       AUTOMATE DISTRIBUÉ    │
          │                             │
          │   ┌─────┐       ┌─────┐    │
          │   │Node │──────►│Node │    │
          │   └─────┘       └──┬──┘    │
          │      ▲              │       │
          │      └──────────────┘       │
          │                             │
          │       Jetons / États        │
          └─────────────┬───────────────┘
                        │
                        │ ID
                        ▼
          ┌─────────────────────────────┐
          │            CUDA             │
          │                             │
          │       GPU Registry          │
          │              │              │
          │              ▼              │
          │            VRAM             │
          │                             │
          │   X    Hidden    Output     │
          │   Δ       W       Bias      │
          │                             │
          └─────────────────────────────┘
```

---

# 17. Évolution prévue

L'architecture pourra évoluer progressivement vers :

* buffers GPU persistants ;
* gestionnaire centralisé des ressources CUDA ;
* réutilisation des buffers ;
* plusieurs exemples simultanés ;
* batching ;
* CUDA Streams ;
* synchronisation fine entre opérations ;
* gestion explicite des dépendances entre états ;
* exécution de plusieurs parties du graphe en parallèle ;
* généralisation à plusieurs couches ;
* séparation complète entre orchestration et calcul.

L'objectif final est de conserver un principe simple :

```text
         GO
          │
          │ référence
          ▼
       AUTOMATE
          │
          │
          ▼
         CUDA
          │
          ▼
         VRAM
```

**Les données restent là où elles sont utiles.
Le ring ne transporte que ce qui est nécessaire pour faire progresser le calcul.**
