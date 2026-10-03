package main

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -lcuda_nn -L/usr/local/cuda/lib64 -lcudart

#include "cuda_nn.h"
*/
import "C"

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"
	"unsafe"
)

/*
================================================================
                    CONFIGURATION DU RÉSEAU
================================================================
*/

const (
	NombreEntrees = 2

	NombreCaches = 2

	NombreSorties = 1

	Epochs = 10000
)

/*
================================================================
                         LIAISON
================================================================

Une liaison possède deux directions indépendantes :

    Forward
        ↓
    A → B

    Backward
        ↑
    A ← B

Les channels ne sont PAS bufferisés.

La communication est donc synchrone.
*/
type Liaison struct {
	Forward  chan Neuronne
	Backward chan Neuronne
}

/*
Crée une liaison bidirectionnelle.
*/
func nouvelleLiaison() Liaison {
	return Liaison{
		Forward:  make(chan Neuronne),
		Backward: make(chan Neuronne),
	}
}

/*
================================================================
                         MESSAGE
================================================================

Neuronne représente l'état transporté par l'anneau.

Le message évolue pendant son parcours.

Forward :

    Entrees
      ↓
    HiddenZ
      ↓
    HiddenA
      ↓
    OutputZ
      ↓
    OutputA
      ↓
    Erreur

Backward :

    DeltaOutput
      ↓
    DeltaHidden
      ↓
    mise à jour des paramètres
*/
type Neuronne struct {

	/*
	 * Données d'entrée.
	 */
	Entrees []float64

	/*
	 * Valeur attendue.
	 */
	Cible []float64

	/*
	 * Couche cachée :
	 *
	 * HiddenZ = W*x
	 * HiddenA = sigmoid(HiddenZ)
	 */
	HiddenZ []float64
	HiddenA []float64

	/*
	 * Couche de sortie :
	 *
	 * OutputZ = W*HiddenA
	 * OutputA = sigmoid(OutputZ)
	 */
	OutputZ []float64
	OutputA []float64

	/*
	 * Gradients calculés pendant le Backward.
	 */
	DeltaHidden []float64
	DeltaOutput []float64

	/*
	 * Erreur du réseau.
	 */
	Erreur float64
}

/*
================================================================
                         MODÈLE
================================================================
*/

type Reseau struct {

	/*
	 * Couche cachée :
	 *
	 * 2 entrées
	 * 2 neurones
	 */
	PoidsHidden [][]float64
	BiaisHidden []float64

	/*
	 * Couche de sortie :
	 *
	 * 2 neurones cachés
	 * 1 sortie
	 */
	PoidsOutput [][]float64
	BiaisOutput []float64

	/*
	 * Taux d'apprentissage.
	 */
	LearningRate float64
}

/*
================================================================
                    CONVERSION CPU / CUDA
================================================================
*/

/*
CUDA utilise ici des float32.

Le modèle Go conserve ses paramètres en float64 pour
simplifier la manipulation côté application.
*/

func versFloat32(
	valeurs []float64,
) []float32 {

	resultat :=
		make([]float32, len(valeurs))

	for i, valeur := range valeurs {
		resultat[i] =
			float32(valeur)
	}

	return resultat
}

func versFloat64(
	valeurs []float32,
) []float64 {

	resultat :=
		make([]float64, len(valeurs))

	for i, valeur := range valeurs {
		resultat[i] =
			float64(valeur)
	}

	return resultat
}

/*
Aplati une matrice Go :

    [][]float64

en :

    []float32

avec un stockage row-major.
*/
func matriceFloat32(
	matrice [][]float64,
) []float32 {

	taille := 0

	for _, ligne := range matrice {
		taille += len(ligne)
	}

	resultat :=
		make([]float32, 0, taille)

	for _, ligne := range matrice {
		for _, valeur := range ligne {
			resultat =
				append(
					resultat,
					float32(valeur),
				)
		}
	}

	return resultat
}

/*
Reconstruit la matrice Go après une modification CUDA.
*/
func mettreAJourMatrice(
	matrice [][]float64,
	valeurs []float32,
) {

	index := 0

	for i := range matrice {

		for j := range matrice[i] {

			matrice[i][j] =
				float64(valeurs[index])

			index++
		}
	}
}

/*
================================================================
                         GPU LINEAR
================================================================
*/

func gpuLinear(
	input []float64,
	poids [][]float64,
) []float64 {

	input32 :=
		versFloat32(input)

	poids32 :=
		matriceFloat32(poids)

	output32 :=
		make(
			[]float32,
			len(poids),
		)

	status :=
		C.cuda_linear(
			(*C.float)(
				unsafe.Pointer(
					&input32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&poids32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&output32[0],
				),
			),

			C.int(len(input32)),
			C.int(len(poids)),
		)

	runtime.KeepAlive(input32)
	runtime.KeepAlive(poids32)

	if status != 0 {
		panic("CUDA: cuda_linear() a échoué")
	}

	return versFloat64(output32)
}

/*
================================================================
                         GPU BIAS
================================================================
*/

func gpuBias(
	valeurs []float64,
	biais []float64,
) []float64 {

	valeurs32 :=
		versFloat32(valeurs)

	biais32 :=
		versFloat32(biais)

	status :=
		C.cuda_add_bias(
			(*C.float)(
				unsafe.Pointer(
					&valeurs32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&biais32[0],
				),
			),

			C.int(len(valeurs32)),
		)

	runtime.KeepAlive(valeurs32)
	runtime.KeepAlive(biais32)

	if status != 0 {
		panic("CUDA: cuda_add_bias() a échoué")
	}

	return versFloat64(valeurs32)
}

/*
================================================================
                         GPU SIGMOID
================================================================
*/

func gpuSigmoid(
	input []float64,
) []float64 {

	input32 :=
		versFloat32(input)

	output32 :=
		make(
			[]float32,
			len(input32),
		)

	status :=
		C.cuda_sigmoid(
			(*C.float)(
				unsafe.Pointer(
					&input32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&output32[0],
				),
			),

			C.int(len(input32)),
		)

	runtime.KeepAlive(input32)

	if status != 0 {
		panic("CUDA: cuda_sigmoid() a échoué")
	}

	return versFloat64(output32)
}

/*
================================================================
                    GPU UPDATE WEIGHTS
================================================================
*/

func gpuUpdateWeights(
	poids [][]float64,
	gradient []float64,
	input []float64,
	learningRate float64,
) {

	poids32 :=
		matriceFloat32(poids)

	gradient32 :=
		versFloat32(gradient)

	input32 :=
		versFloat32(input)

	status :=
		C.cuda_update_weights(
			(*C.float)(
				unsafe.Pointer(
					&poids32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&gradient32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&input32[0],
				),
			),

			C.float(learningRate),

			C.int(len(poids)),

			C.int(len(input)),
		)

	runtime.KeepAlive(poids32)
	runtime.KeepAlive(gradient32)
	runtime.KeepAlive(input32)

	if status != 0 {
		panic(
			"CUDA: cuda_update_weights() a échoué",
		)
	}

	mettreAJourMatrice(
		poids,
		poids32,
	)
}

/*
================================================================
                      GPU UPDATE BIAS
================================================================
*/

func gpuUpdateBias(
	biais []float64,
	gradient []float64,
	learningRate float64,
) {

	biais32 :=
		versFloat32(biais)

	gradient32 :=
		versFloat32(gradient)

	status :=
		C.cuda_update_bias(
			(*C.float)(
				unsafe.Pointer(
					&biais32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&gradient32[0],
				),
			),

			C.float(learningRate),

			C.int(len(biais)),
		)

	runtime.KeepAlive(biais32)
	runtime.KeepAlive(gradient32)

	if status != 0 {
		panic(
			"CUDA: cuda_update_bias() a échoué",
		)
	}

	for i := range biais {
		biais[i] =
			float64(biais32[i])
	}
}

/*
================================================================
                   GPU HIDDEN GRADIENT
================================================================
*/

func gpuHiddenGradient(
	poidsOutput [][]float64,
	deltaOutput []float64,
	hiddenActivation []float64,
) []float64 {

	poids32 :=
		matriceFloat32(poidsOutput)

	deltaOutput32 :=
		versFloat32(deltaOutput)

	hidden32 :=
		versFloat32(hiddenActivation)

	deltaHidden32 :=
		make(
			[]float32,
			len(hiddenActivation),
		)

	status :=
		C.cuda_hidden_gradient(
			(*C.float)(
				unsafe.Pointer(
					&poids32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&deltaOutput32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&hidden32[0],
				),
			),

			(*C.float)(
				unsafe.Pointer(
					&deltaHidden32[0],
				),
			),

			C.int(len(hiddenActivation)),
			C.int(len(deltaOutput)),
		)

	runtime.KeepAlive(poids32)
	runtime.KeepAlive(deltaOutput32)
	runtime.KeepAlive(hidden32)

	if status != 0 {
		panic(
			"CUDA: cuda_hidden_gradient() a échoué",
		)
	}

	return versFloat64(deltaHidden32)
}

/*
================================================================
                    INITIALISATION DU RÉSEAU
================================================================
*/

func nouveauReseau() *Reseau {

	return &Reseau{

		/*
		 * 2 entrées → 2 neurones cachés.
		 */
		PoidsHidden: [][]float64{

			{
				rand.Float64()*2 - 1,
				rand.Float64()*2 - 1,
			},

			{
				rand.Float64()*2 - 1,
				rand.Float64()*2 - 1,
			},
		},

		BiaisHidden: []float64{
			rand.Float64()*2 - 1,
			rand.Float64()*2 - 1,
		},

		/*
		 * 2 neurones cachés → 1 sortie.
		 */
		PoidsOutput: [][]float64{

			{
				rand.Float64()*2 - 1,
				rand.Float64()*2 - 1,
			},
		},

		BiaisOutput: []float64{
			rand.Float64()*2 - 1,
		},

		/*
		 * Taux d'apprentissage.
		 */
		LearningRate: 0.8,
	}
}

/*
================================================================
                         NODE INPUT
================================================================

Responsabilité :

    extérieur
       ↓
    Node_Input
       ↓
    anneau

Puis, au retour :

    anneau
       ↓
    Node_Input
       ↓
    extérieur
*/
func Node_Input(
	entree chan Neuronne,
	sortie Liaison,
	termine chan Neuronne,
) {

	for {

		select {

		/*
		 * Nouveau cas d'apprentissage.
		 */
		case n := <-entree:

			sortie.Forward <- n

		/*
		 * Le message a effectué son Backward.
		 */
		case n := <-sortie.Backward:

			termine <- n
		}
	}
}

/*
================================================================
                       NODE NEURONNE
================================================================

IMPORTANT :

Node_Neuronne ne calcule PAS le neurone.

Il prépare uniquement l'espace de travail du message.

Les calculs sont réalisés par les Nodes spécialisés :
    
    Sum
    Biais
    Sigmoid
*/
func Node_Neuronne(
	entree Liaison,
	sortie Liaison,
) {

	for {

		select {

		/*
		 * ------------------------------
		 * FORWARD
		 * ------------------------------
		 */
		case n := <-entree.Forward:

			n.HiddenZ =
				make(
					[]float64,
					NombreCaches,
				)

			n.HiddenA =
				make(
					[]float64,
					NombreCaches,
				)

			n.OutputZ =
				make(
					[]float64,
					NombreSorties,
				)

			n.OutputA =
				make(
					[]float64,
					NombreSorties,
				)

			n.DeltaHidden =
				make(
					[]float64,
					NombreCaches,
				)

			n.DeltaOutput =
				make(
					[]float64,
					NombreSorties,
				)

			sortie.Forward <- n

		/*
		 * ------------------------------
		 * BACKWARD
		 * ------------------------------
		 */
		case n := <-sortie.Backward:

			entree.Backward <- n
		}
	}
}

/*
================================================================
                         NODE SUM
================================================================

Un Node_Sum peut être instancié pour :

    - la couche cachée
    - la couche de sortie

Le Node ne s'occupe pas du biais.
*/
type CoucheSum int

const (
	SumHidden CoucheSum = iota
	SumOutput
)

func Node_Sum(
	reseau *Reseau,
	couche CoucheSum,
	entree Liaison,
	sortie Liaison,
) {

	for {

		select {

		/*
		 * ======================================================
		 * FORWARD
		 * ======================================================
		 */
		case n := <-entree.Forward:

			switch couche {

			case SumHidden:

				n.HiddenZ =
					gpuLinear(
						n.Entrees,
						reseau.PoidsHidden,
					)

			case SumOutput:

				n.OutputZ =
					gpuLinear(
						n.HiddenA,
						reseau.PoidsOutput,
					)
			}

			sortie.Forward <- n

		/*
		 * ======================================================
		 * BACKWARD
		 * ======================================================
		 */
		case n := <-sortie.Backward:

			switch couche {

			case SumOutput:

				/*
				 * Avant de modifier les poids de sortie,
				 * on calcule le gradient à transmettre
				 * vers la couche cachée.
				 *
				 * Il faut utiliser les poids actuels.
				 */
				n.DeltaHidden =
					gpuHiddenGradient(
						reseau.PoidsOutput,
						n.DeltaOutput,
						n.HiddenA,
					)

				/*
				 * Mise à jour des poids :
				 *
				 * W -= η * delta * entrée
				 */
				gpuUpdateWeights(
					reseau.PoidsOutput,
					n.DeltaOutput,
					n.HiddenA,
					reseau.LearningRate,
				)

			case SumHidden:

				/*
				 * Mise à jour des poids de la couche cachée.
				 */
				gpuUpdateWeights(
					reseau.PoidsHidden,
					n.DeltaHidden,
					n.Entrees,
					reseau.LearningRate,
				)
			}

			entree.Backward <- n
		}
	}
}

/*
================================================================
                        NODE BIAIS
================================================================

Responsabilité :

    z + b

Il existe une instance pour chaque couche.
*/
type CoucheBiais int

const (
	BiaisHidden CoucheBiais = iota
	BiaisOutput
)

func Node_Biais(
	reseau *Reseau,
	couche CoucheBiais,
	entree Liaison,
	sortie Liaison,
) {

	for {

		select {

		/*
		 * ======================================================
		 * FORWARD
		 * ======================================================
		 */
		case n := <-entree.Forward:

			switch couche {

			case BiaisHidden:

				n.HiddenZ =
					gpuBias(
						n.HiddenZ,
						reseau.BiaisHidden,
					)

			case BiaisOutput:

				n.OutputZ =
					gpuBias(
						n.OutputZ,
						reseau.BiaisOutput,
					)
			}

			sortie.Forward <- n

		/*
		 * ======================================================
		 * BACKWARD
		 * ======================================================
		 */
		case n := <-sortie.Backward:

			switch couche {

			case BiaisHidden:

				gpuUpdateBias(
					reseau.BiaisHidden,
					n.DeltaHidden,
					reseau.LearningRate,
				)

			case BiaisOutput:

				gpuUpdateBias(
					reseau.BiaisOutput,
					n.DeltaOutput,
					reseau.LearningRate,
				)
			}

			entree.Backward <- n
		}
	}
}

/*
================================================================
                       NODE SIGMOID
================================================================

Même principe :

    Node_Sigmoid Hidden
    Node_Sigmoid Output
*/
type CoucheSigmoid int

const (
	SigmoidHidden CoucheSigmoid = iota
	SigmoidOutput
)

func Node_Sigmoid(
	couche CoucheSigmoid,
	entree Liaison,
	sortie Liaison,
) {

	for {

		select {

		/*
		 * ======================================================
		 * FORWARD
		 * ======================================================
		 */
		case n := <-entree.Forward:

			switch couche {

			case SigmoidHidden:

				n.HiddenA =
					gpuSigmoid(
						n.HiddenZ,
					)

			case SigmoidOutput:

				n.OutputA =
					gpuSigmoid(
						n.OutputZ,
					)
			}

			sortie.Forward <- n

		/*
		 * ======================================================
		 * BACKWARD
		 * ======================================================
		 */
		case n := <-sortie.Backward:

			switch couche {

			case SigmoidOutput:

				/*
				 * MSE :

					   E = 1/2 (y - t)^2

				 * Donc :

					   dE/dy = y - t

				 * Puis :

					   delta =
					       (y - t)
					       *
					       sigmoid'(y)
				 */
				for i := 0;
					i < NombreSorties;
					i++ {

					erreur :=
						n.OutputA[i] -
							n.Cible[i]

					n.DeltaOutput[i] =
						erreur *
							n.OutputA[i] *
							(1.0 - n.OutputA[i])
				}

			case SigmoidHidden:

				/*
				 * Le gradient a déjà été propagé
				 * depuis la couche de sortie par
				 * Node_Sum(SumOutput).
				 *
				 * Il n'est donc pas nécessaire de
				 * recalculer DeltaHidden ici.
				 */
			}

			entree.Backward <- n
		}
	}
}

/*
================================================================
                         NODE OUTPUT
================================================================

Dernière étape du Forward.

Le Node_Output :

    1. reçoit la prédiction ;
    2. compare avec la cible ;
    3. calcule la perte ;
    4. renvoie le message dans l'autre sens.
*/
func Node_Output(
	entree Liaison,
	sortie Liaison,
) {

	for {

		select {

		/*
		 * ======================================================
		 * FORWARD
		 * ======================================================
		 */
		case n := <-entree.Forward:

			/*
			 * Erreur MSE :

				   E =
				       1/2 *
				       (prediction - cible)^2
			 */
			n.Erreur = 0

			for i := 0;
				i < NombreSorties;
				i++ {

				difference :=
					n.OutputA[i] -
						n.Cible[i]

				n.Erreur +=
					0.5 *
						difference *
						difference
			}

			/*
			 * Le Forward est terminé.

			 * Le message repart dans le sens
			 * inverse.
			 */
			sortie.Backward <- n

		/*
		 * ======================================================
		 * BACKWARD
		 * ======================================================

		 * Normalement le message ne revient pas ici :
		 * le Node_Input ferme la boucle.
		 */
		case n := <-entree.Backward:

			sortie.Backward <- n
		}
	}
}

/*
================================================================
                        DONNÉES XOR
================================================================
*/

func donneesXOR() []Neuronne {

	return []Neuronne{

		{
			Entrees: []float64{0, 0},
			Cible:   []float64{0},
		},

		{
			Entrees: []float64{0, 1},
			Cible:   []float64{1},
		},

		{
			Entrees: []float64{1, 0},
			Cible:   []float64{1},
		},

		{
			Entrees: []float64{1, 1},
			Cible:   []float64{0},
		},
	}
}

/*
================================================================
                        ENTRAÎNEMENT
================================================================
*/

func entrainer(
	input chan Neuronne,
	termine chan Neuronne,
	reseau *Reseau,
) {

	donnees :=
		donneesXOR()

	for epoch := 0;
		epoch < Epochs;
		epoch++ {

		erreurTotale := 0.0

		for _, exemple := range donnees {

			/*
			 * Injection dans l'anneau.
			 */
			input <- exemple

			/*
			 * Attente du retour après
			 * le Forward ET le Backward.
			 */
			resultat :=
				<-termine

			erreurTotale +=
				resultat.Erreur
		}

		/*
		 * Affichage périodique.
		 */
		if epoch%1000 == 0 {

			fmt.Printf(
				"epoch %5d | erreur = %.8f\n",
				epoch,
				erreurTotale,
			)
		}
	}

	fmt.Println()
	fmt.Println(
		"Apprentissage terminé.",
	)
}

/*
================================================================
                         PREDICTION
================================================================
*/

func predire(
	reseau *Reseau,
	entrees []float64,
) float64 {

	/*
	 * -----------------------------
	 * Couche cachée
	 * -----------------------------
	 */

	hiddenZ :=
		gpuLinear(
			entrees,
			reseau.PoidsHidden,
		)

	hiddenZ =
		gpuBias(
			hiddenZ,
			reseau.BiaisHidden,
		)

	hiddenA :=
		gpuSigmoid(hiddenZ)


	/*
	 * -----------------------------
	 * Couche sortie
	 * -----------------------------
	 */

	outputZ :=
		gpuLinear(
			hiddenA,
			reseau.PoidsOutput,
		)

	outputZ =
		gpuBias(
			outputZ,
			reseau.BiaisOutput,
		)

	outputA :=
		gpuSigmoid(outputZ)

	return outputA[0]
}

/*
================================================================
                         TEST XOR
================================================================
*/

func tester(
	reseau *Reseau,
) {

	fmt.Println()
	fmt.Println("==============================")
	fmt.Println("          TEST XOR")
	fmt.Println("==============================")

	tests :=
		[][]float64{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		}

	for _, entree := range tests {

		resultat :=
			predire(
				reseau,
				entree,
			)

		fmt.Printf(
			"%.0f XOR %.0f = %.6f\n",
			entree[0],
			entree[1],
			resultat,
		)
	}
}

/*
================================================================
                     AFFICHAGE PARAMÈTRES
================================================================
*/

func afficherParametres(
	reseau *Reseau,
) {

	fmt.Println()

	fmt.Println(
		"==============================",
	)

	fmt.Println(
		"       PARAMÈTRES FINAUX",
	)

	fmt.Println(
		"==============================",
	)

	fmt.Println()

	fmt.Println(
		"Poids couche cachée :",
	)

	for _, ligne :=
		range reseau.PoidsHidden {

		fmt.Println(
			ligne,
		)
	}

	fmt.Println()

	fmt.Println(
		"Biais couche cachée :",
		reseau.BiaisHidden,
	)

	fmt.Println()

	fmt.Println(
		"Poids couche sortie :",
		reseau.PoidsOutput,
	)

	fmt.Println()

	fmt.Println(
		"Biais couche sortie :",
		reseau.BiaisOutput,
	)
}

/*
================================================================
                            MAIN
================================================================
*/

func main() {

	/*
	 * Initialisation aléatoire.
	 */
	rand.Seed(
		time.Now().UnixNano(),
	)


	/*
	 * ==========================================================
	 * CUDA
	 * ==========================================================
	 */

	if C.cuda_nn_init() != 0 {

		panic(
			"Impossible d'initialiser CUDA.",
		)
	}

	defer C.cuda_nn_shutdown()


	/*
	 * ==========================================================
	 * MODÈLE
	 * ==========================================================
	 */

	reseau :=
		nouveauReseau()


	/*
	 * ==========================================================
	 * ANNEAU
	 * ==========================================================
	 *
	 * Forward :
	 *
	 * Input
	 *   ↓
	 * Neuronne
	 *   ↓
	 * SumHidden
	 *   ↓
	 * BiaisHidden
	 *   ↓
	 * SigmoidHidden
	 *   ↓
	 * SumOutput
	 *   ↓
	 * BiaisOutput
	 *   ↓
	 * SigmoidOutput
	 *   ↓
	 * Output
	 *
	 * Backward :
	 *
	 * Output
	 *   ↓
	 * SigmoidOutput
	 *   ↓
	 * BiaisOutput
	 *   ↓
	 * SumOutput
	 *   ↓
	 * SigmoidHidden
	 *   ↓
	 * BiaisHidden
	 *   ↓
	 * SumHidden
	 *   ↓
	 * Neuronne
	 *   ↓
	 * Input
	 *
	 * C'est ici que l'anneau est fermé.
	 */


	inputNeuronne :=
		nouvelleLiaison()

	neuronneSumHidden :=
		nouvelleLiaison()

	sumHiddenBiais :=
		nouvelleLiaison()

	biaisHiddenSigmoid :=
		nouvelleLiaison()

	sigmoidHiddenSumOutput :=
		nouvelleLiaison()

	sumOutputBiais :=
		nouvelleLiaison()

	biaisOutputSigmoid :=
		nouvelleLiaison()

	sigmoidOutputOutput :=
		nouvelleLiaison()


	/*
	 * Channel entre le programme et Node_Input.
	 */
	input :=
		make(chan Neuronne)

	/*
	 * Channel de retour vers le programme.
	 */
	termine :=
		make(chan Neuronne)


	/*
	 * ==========================================================
	 * LANCEMENT DES NODES
	 * ==========================================================
	 */

	go Node_Input(
		input,
		inputNeuronne,
		termine,
	)

	go Node_Neuronne(
		inputNeuronne,
		neuronneSumHidden,
	)

	go Node_Sum(
		reseau,
		SumHidden,
		neuronneSumHidden,
		sumHiddenBiais,
	)

	go Node_Biais(
		reseau,
		BiaisHidden,
		sumHiddenBiais,
		biaisHiddenSigmoid,
	)

	go Node_Sigmoid(
		SigmoidHidden,
		biaisHiddenSigmoid,
		sigmoidHiddenSumOutput,
	)

	go Node_Sum(
		reseau,
		SumOutput,
		sigmoidHiddenSumOutput,
		sumOutputBiais,
	)

	go Node_Biais(
		reseau,
		BiaisOutput,
		sumOutputBiais,
		biaisOutputSigmoid,
	)

	go Node_Sigmoid(
		SigmoidOutput,
		biaisOutputSigmoid,
		sigmoidOutputOutput,
	)

	/*
	 * Node_Output ferme l'anneau :
	 *
	 *     Output
	 *        │
	 *        ▼
	 *     Backward
	 *        │
	 *        ▼
	 *     SigmoidOutput
	 */
	go Node_Output(
		sigmoidOutputOutput,
		sigmoidOutputOutput,
	)


	/*
	 * ==========================================================
	 * APPRENTISSAGE
	 * ==========================================================
	 */

	entrainer(
		input,
		termine,
		reseau,
  )


	/*
	 * ==========================================================
	 * TEST
	 * ==========================================================
	 */

	tester(
		reseau,
	)


	/*
	 * ==========================================================
	 * PARAMÈTRES
	 * ==========================================================
	 */

	afficherParametres( reseau, )

	fmt.Println()
	fmt.Println(
		"Réseau neuronal terminé.",
	)
}
