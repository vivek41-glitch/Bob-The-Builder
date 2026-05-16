import subprocess
import csv
import os
import copy
import random
from datetime import datetime
from train import run_training
from objective import calculate_score
from config import BASELINE_CONFIG, EXPERIMENT_LOG_FILE, BEST_MODEL_FILE

class BobOptimizer:
    def __init__(self):
        self.baseline_config = copy.deepcopy(BASELINE_CONFIG)
        self.current_config = copy.deepcopy(BASELINE_CONFIG)
        self.best_config = copy.deepcopy(BASELINE_CONFIG)
        self.best_score = 0
        self.best_accuracy = 0
        self.experiments = []
        self.iteration = 0
        
        # Possible changes Bob can make
        self.learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        self.batch_sizes = [32, 64, 128]
        self.optimizers = ['adam', 'sgd', 'adamw']
        self.dropout_rates = [0.0, 0.2, 0.3, 0.5]
        self.use_batch_norm_options = [True, False]
        self.weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
        self.augmentation_options = [True, False]
        
    def git_commit(self, message):
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            print(f" Committed: {message}")
            return True
        except subprocess.CalledProcessError:
            print(f" Commit failed")
            return False
    
    def git_rollback(self):
        try:
            subprocess.run(["git", "reset", "--hard", "HEAD"], check=True)
            print(f" Rolled back to previous state")
            return True
        except subprocess.CalledProcessError:
            print(f" Rollback failed")
            return False
    
    def apply_change(self):
        """Apply a random change to the current config"""
        change_type = random.choice(['lr', 'batch', 'optimizer', 'dropout', 'batchnorm', 'weightdecay', 'augmentation'])
        
        if change_type == 'lr':
            new_value = random.choice(self.learning_rates)
            old_value = self.current_config['learning_rate']
            self.current_config['learning_rate'] = new_value
            return f"learning_rate: {old_value} -> {new_value}"
        
        elif change_type == 'batch':
            new_value = random.choice(self.batch_sizes)
            old_value = self.current_config['batch_size']
            self.current_config['batch_size'] = new_value
            return f"batch_size: {old_value} -> {new_value}"
        
        elif change_type == 'optimizer':
            new_value = random.choice(self.optimizers)
            old_value = self.current_config['optimizer']
            self.current_config['optimizer'] = new_value
            return f"optimizer: {old_value} -> {new_value}"
        
        elif change_type == 'dropout':
            new_value = random.choice(self.dropout_rates)
            old_value = self.current_config['dropout_rate']
            self.current_config['dropout_rate'] = new_value
            return f"dropout_rate: {old_value} -> {new_value}"
        
        elif change_type == 'batchnorm':
            new_value = random.choice(self.use_batch_norm_options)
            old_value = self.current_config['use_batch_norm']
            self.current_config['use_batch_norm'] = new_value
            return f"use_batch_norm: {old_value} -> {new_value}"
        
        elif change_type == 'weightdecay':
            new_value = random.choice(self.weight_decays)
            old_value = self.current_config['weight_decay']
            self.current_config['weight_decay'] = new_value
            return f"weight_decay: {old_value} -> {new_value}"
        
        elif change_type == 'augmentation':
            new_value = random.choice(self.augmentation_options)
            old_value = self.current_config['augmentation']
            self.current_config['augmentation'] = new_value
            return f"augmentation: {old_value} -> {new_value}"
    
    def run_baseline(self):
        print("\n" + "="*50)
        print("RUNNING BASELINE EXPERIMENT")
        print("="*50)
        
        accuracy, loss, training_time = run_training(self.baseline_config)
        score = calculate_score(accuracy, loss, training_time)
        
        self.baseline_accuracy = accuracy
        self.baseline_score = score
        self.best_accuracy = accuracy
        self.best_score = score
        
        self.log_experiment("baseline", self.baseline_config, accuracy, loss, training_time, score, None)
        
        print(f"\n BASELINE RESULTS:")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Loss: {loss:.4f}")
        print(f"   Time: {training_time:.2f}s")
        print(f"   Score: {score:.2f}")
        
        return accuracy, score
    
    def log_experiment(self, iteration, config, accuracy, loss, training_time, score, change_desc):
        log_file = EXPERIMENT_LOG_FILE
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['iteration', 'timestamp', 'change', 'lr', 'batch_size', 'epochs', 'optimizer', 
                               'dropout_rate', 'use_batch_norm', 'weight_decay', 'augmentation',
                               'accuracy', 'loss', 'time', 'score'])
            
            writer.writerow([
                iteration, datetime.now().isoformat(), change_desc,
                config['learning_rate'], config['batch_size'], config['epochs'],
                config['optimizer'], config['dropout_rate'], config['use_batch_norm'],
                config['weight_decay'], config['augmentation'],
                accuracy, loss, training_time, score
            ])
    
    def run_optimization(self, num_iterations=30):
        print("\n" + "="*50)
        print("BOB THE BUILDER - AUTO OPTIMIZATION STARTING")
        print("="*50)
        
        self.run_baseline()
        
        for i in range(1, num_iterations + 1):
            self.iteration = i
            print(f"\n{'='*50}")
            print(f"ITERATION {i}/{num_iterations}")
            print(f"{'='*50}")
            
            change_desc = self.apply_change()
            print(f" Bob's change: {change_desc}")
            
            # Run training with new config
            accuracy, loss, training_time = run_training(self.current_config)
            score = calculate_score(accuracy, loss, training_time)
            
            print(f" Results - Acc: {accuracy:.2f}%, Loss: {loss:.4f}, Time: {training_time:.2f}s, Score: {score:.2f}")
            
            if score > self.best_score:
                # Improvement - commit the change
                print(f" IMPROVEMENT! Score improved from {self.best_score:.2f} to {score:.2f}")
                self.best_score = score
                self.best_accuracy = accuracy
                self.best_config = copy.deepcopy(self.current_config)
                self.git_commit(f"Iteration {i}: {change_desc} | Acc: {accuracy:.2f}% | Score: {score:.2f}")
            else:
                # No improvement - rollback
                print(f" NO IMPROVEMENT. Rolling back...")
                self.current_config = copy.deepcopy(self.best_config) if hasattr(self, 'best_config') else copy.deepcopy(self.baseline_config)
                self.git_rollback()
            
            self.log_experiment(i, self.current_config, accuracy, loss, training_time, score, change_desc)
            
            # Print best so far
            print(f" Best so far: Accuracy {self.best_accuracy:.2f}%, Score {self.best_score:.2f}")
        
        self.print_final_results()
    
    def print_final_results(self):
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETE - FINAL RESULTS")
        print("="*50)
        
        print(f"\n IMPROVEMENT:")
        print(f"   Baseline Accuracy: {self.baseline_accuracy:.2f}%")
        print(f"   Best Accuracy: {self.best_accuracy:.2f}%")
        print(f"   Improvement: {self.best_accuracy - self.baseline_accuracy:+.2f}%")
        
        print(f"\n BEST CONFIGURATION FOUND:")
        for key, value in self.best_config.items():
            print(f"   {key}: {value}")
        
        print(f"\n Results saved to: {EXPERIMENT_LOG_FILE}")
        print(f" Best model saved to: {BEST_MODEL_FILE}")

if __name__ == "__main__":
    optimizer = BobOptimizer()
    optimizer.run_optimization(num_iterations=30)
    