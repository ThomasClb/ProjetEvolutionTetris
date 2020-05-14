# Projet Algorithmes évolutionnaires Tetris

### Fichiers
__run.py :__ : NN implémenté avec en entrée les 4 paramètres de l'actuelle map + l'id de la pièce qui tombe et en sortie BOARD_WIDTH + 4 neurones qui indiquent la position et la rotation de la pièce qui chute.  

__main_evaluate_state.py :__  NN implémenté avec en entrée les 4 paramètres de la map suivante (après que la pièce soit tombée) et en sortie un neurone qui évalue si cette map suivant est bien ou pas.  

<br>

### Ce qui fonctionne
Lancer `python main_evaluate_state solve` pour lancer un CMA-ES  


#### Pour le voir jouer

1. Modifier la variable FILE_TEST de main_evaluate_state.py pour qu'il prenne la valeur du meilleur fichier dans best_score_fitness ou best_fitness_score (s'il le dossier n'est pas vide...)  
2. Lancer `python main_evaluate_state test` pour voir l'agent jouer  
