import numpy as np
import pandas as pd
import random
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

# --- 1. CONFIGURATION ---
POPULATION_SIZE = 10
GENERATIONS = 5
MUTATION_RATE = 0.1
ELITISM_COUNT = 2

GENE_BOUNDS = [
    (10, 200),  # n_estimators
    (1, 30),    # max_depth
    (2, 10)     # min_samples_split
]

# --- 2. DATA LOADING (FIXED) ---
def load_data():
    print("Loading Forest Fire data...")
    
    # Get the directory where THIS script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "forest_fire_features.csv")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find file at: {csv_path}")
        print("Please make sure 'forest_fire_features.csv' is in the same folder as this script.")
        exit() # Stop the program if file is missing
        
    df = pd.read_csv(csv_path)
    
    # Separate Features (X) and Labels (y)
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. THE GENETIC ALGORITHM CLASS ---
class GeneticOptimizer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def create_individual(self):
        return [random.randint(low, high) for low, high in GENE_BOUNDS]

    def create_population(self):
        return [self.create_individual() for _ in range(POPULATION_SIZE)]

    def fitness(self, individual):
        n_est, depth, split = individual
        if split < 2: split = 2
        
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_split=split,
            n_jobs=-1,
            random_state=42
        )
        # We use 'recall' to force the model to prioritize finding the positive class (Fire)
        scores = cross_val_score(clf, self.X_train, self.y_train, cv=3, scoring='recall')
        return scores.mean()

    def selection(self, population, fitness_scores):
        tournament_size = 3
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        if random.random() < 0.7:
            point = random.randint(1, len(GENE_BOUNDS) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < MUTATION_RATE:
                low, high = GENE_BOUNDS[i]
                individual[i] = random.randint(low, high)
        return individual

    def run(self):
        population = self.create_population()
        best_ind = None
        
        for gen in range(GENERATIONS):
            print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")
            fitness_scores = [self.fitness(ind) for ind in population]
            
            best_score = max(fitness_scores)
            best_ind = population[fitness_scores.index(best_score)]
            print(f"Best Accuracy: {best_score:.4f} | Params: {best_ind}")
            
            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            new_population = [ind for ind, score in sorted_pop[:ELITISM_COUNT]]
            
            parents = self.selection(population, fitness_scores)
            
            while len(new_population) < POPULATION_SIZE:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                c1, c2 = self.crossover(p1, p2)
                new_population.append(self.mutate(c1))
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(self.mutate(c2))
            
            population = new_population

        return best_ind

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # B. Run Genetic Algorithm
    print("Starting Genetic Optimization...")
    ga = GeneticOptimizer(X_train, y_train)
    best_params = ga.run()
    
    print("\n--------------------------------------------")
    print(f"OPTIMIZATION COMPLETE.")
    print(f"Best Parameters: n_estimators={best_params[0]}, max_depth={best_params[1]}, min_samples_split={best_params[2]}")
    print("--------------------------------------------\n")
    
    # C. Train Final Model & Save
    print("Training final model...")
    final_model = RandomForestClassifier(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        min_samples_split=best_params[2],
        random_state=42
    )
    final_model.fit(X_train, y_train)
    
    # SAVE THE MODEL
    joblib.dump(final_model, 'fire_detection_model.pkl')
    print("Model saved as 'fire_detection_model.pkl'")
    
    # D. Evaluate
    predictions = final_model.predict(X_test)
    print("\nFinal Performance on Test Data:")
    print(classification_report(y_test, predictions))
