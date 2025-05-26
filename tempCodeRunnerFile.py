import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkAbsen1:
    def __init__(self):
        # Fixed weights untuk konsistensi dengan perhitungan manual
        # Arsitektur: 2 Input -> 3 Hidden -> 1 Output (sesuai absen 1)
        
        # W1: Input ke Hidden (2x3)
        self.W1 = np.array([[0.2, 0.4, 0.1],    # X1 -> H1,H2,H3
                           [0.3, 0.1, 0.5]])    # X2 -> H1,H2,H3
        
        # W2: Hidden ke Output (3x1)  
        self.W2 = np.array([[0.6],              # H1 -> Y1
                           [0.3],              # H2 -> Y1
                           [0.4]])             # H3 -> Y1
        
        # Bias
        self.b1 = np.array([[0.1, 0.2, 0.1]])   # Hidden bias
        self.b2 = np.array([[0.2]])             # Output bias
        
        # Learning rate untuk absen 1
        self.learning_rate = 0.1
        
        # Storage untuk tracking
        self.errors = []
        self.epoch_results = []
        
    def sigmoid(self, x):
        # Prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward propagation dengan detail untuk manual checking
        
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y):
        # Backward propagation dengan detail untuk manual checking
        m = X.shape[0]
        
        # Output layer gradients
        self.dZ2 = self.a2 - y
        self.dW2 = (1/m) * np.dot(self.a1.T, self.dZ2)
        self.db2 = (1/m) * np.sum(self.dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        self.dA1 = np.dot(self.dZ2, self.W2.T)
        self.dZ1 = self.dA1 * self.sigmoid_derivative(self.a1)
        self.dW1 = (1/m) * np.dot(X.T, self.dZ1)
        self.db1 = (1/m) * np.sum(self.dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
    
    def train_one_epoch(self, X, y, epoch_num):
        epoch_data = {
            'epoch': epoch_num,
            'patterns': [],
            'total_mse': 0,
            'weights_before': {
                'W1': self.W1.copy(),
                'W2': self.W2.copy(),
                'b1': self.b1.copy(),
                'b2': self.b2.copy()
            }
        }
        
        total_error = 0
        
        # Process each pattern
        for i in range(len(X)):
            # Single pattern
            x_single = X[i:i+1]
            y_single = y[i:i+1]
            
            # Forward pass
            output = self.forward(x_single)
            
            # Calculate error
            error = y_single - output
            mse = 0.5 * np.sum(error ** 2)
            total_error += mse
            
            # Store pattern result
            pattern_data = {
                'pattern_num': i+1,
                'input': X[i],
                'target': y[i][0],
                'output': output[0][0],
                'error': error[0][0],
                'mse': mse,
                'hidden_outputs': self.a1[0].copy(),
                'hidden_nets': self.z1[0].copy()
            }
            epoch_data['patterns'].append(pattern_data)
            
            # Backward pass (update after each pattern)
            self.backward(x_single, y_single)
        
        epoch_data['total_mse'] = total_error
        epoch_data['weights_after'] = {
            'W1': self.W1.copy(),
            'W2': self.W2.copy(),
            'b1': self.b1.copy(),
            'b2': self.b2.copy()
        }
        
        self.epoch_results.append(epoch_data)
        self.errors.append(total_error)
        
        return total_error
    
    def train(self, X, y, epochs=10, target_error=0.01):
        print("=== TRAINING NEURAL NETWORK XOR - ABSEN 1 ===")
        print("Arsitektur: 2 Input -> 3 Hidden -> 1 Output")
        print(f"Learning Rate: {self.learning_rate}")
        print("\nBobot Awal:")
        print("W1 (Input ke Hidden):")
        print(self.W1)
        print("W2 (Hidden ke Output):")
        print(self.W2)
        print("Bias Hidden:", self.b1)
        print("Bias Output:", self.b2)
        print("\n" + "="*50)
        
        for epoch in range(epochs):
            error = self.train_one_epoch(X, y, epoch + 1)
            
            if epoch < 3:  # Detail untuk 3 epoch pertama
                self.print_epoch_details(epoch)
            else:
                print(f"Epoch {epoch+1}: Total MSE = {error:.6f}")
            
            if error < target_error:
                print(f"\nTarget error tercapai di epoch {epoch+1}!")
                break
        
        print("\n" + "="*50)
        print("HASIL AKHIR:")
        self.test_network(X, y)
    
    def print_epoch_details(self, epoch_idx):
        epoch_data = self.epoch_results[epoch_idx]
        print(f"\n=== EPOCH {epoch_data['epoch']} - DETAIL ===")
        
        for pattern in epoch_data['patterns']:
            print(f"\nPattern {pattern['pattern_num']}: ({pattern['input'][0]}, {pattern['input'][1]}) -> {pattern['target']}")
            print(f"  Hidden nets: [{pattern['hidden_nets'][0]:.3f}, {pattern['hidden_nets'][1]:.3f}, {pattern['hidden_nets'][2]:.3f}]")
            print(f"  Hidden outputs: [{pattern['hidden_outputs'][0]:.3f}, {pattern['hidden_outputs'][1]:.3f}, {pattern['hidden_outputs'][2]:.3f}]")
            print(f"  Output: {pattern['output']:.3f}")
            print(f"  Error: {pattern['error']:.3f}")
            print(f"  MSE: {pattern['mse']:.3f}")
        
        print(f"\nTotal MSE Epoch {epoch_data['epoch']}: {epoch_data['total_mse']:.6f}")
        print("Bobot setelah update:")
        print("W1:", epoch_data['weights_after']['W1'])
        print("W2:", epoch_data['weights_after']['W2'].flatten())
    
    def test_network(self, X, y):
        print("\nTesting Network:")
        print("Input -> Target | Predicted | Error")
        print("-" * 40)
        
        total_error = 0
        for i in range(len(X)):
            x_test = X[i:i+1]
            prediction = self.forward(x_test)[0][0]
            target = y[i][0]
            error = abs(target - prediction)
            total_error += error
            
            print(f"({X[i][0]}, {X[i][1]}) -> {target} | {prediction:.4f} | {error:.4f}")
        
        accuracy = (4 - total_error) / 4 * 100
        print(f"\nAkurasi rata-rata: {accuracy:.2f}%")
        
        print("\nBobot Akhir:")
        print("W1 (Input ke Hidden):")
        print(self.W1)
        print("W2 (Hidden ke Output):")
        print(self.W2.flatten())
    
    def plot_results(self):
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Error vs Epoch
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.errors)+1), self.errors, 'b-o', markersize=4)
        plt.title('Error vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Total MSE')
        plt.grid(True)
        plt.yscale('log')
        
        # Plot 2: Prediction vs Target
        plt.subplot(1, 2, 2)
        targets = [0, 1, 1, 0]
        X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        predictions = []
        
        for i in range(len(X_test)):
            pred = self.forward(X_test[i:i+1])[0][0]
            predictions.append(pred)
        
        plt.scatter(targets, predictions, c=['red', 'blue', 'blue', 'red'], s=100)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('Target')
        plt.ylabel('Predicted')
        plt.title('Prediction vs Target')
        plt.grid(True)
        
        # Add labels
        labels = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
        for i, label in enumerate(labels):
            plt.annotate(label, (targets[i], predictions[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
    
    def export_manual_calculation_data(self):
        """Export data untuk perhitungan manual"""
        print("\n" + "="*60)
        print("DATA UNTUK PERHITUNGAN MANUAL DI KERTAS")
        print("="*60)
        
        print("\n1. BOBOT AWAL:")
        print("W1 (Input ke Hidden):")
        print("      H1    H2    H3")
        print(f"X1  {self.epoch_results[0]['weights_before']['W1'][0][0]:.1f}  {self.epoch_results[0]['weights_before']['W1'][0][1]:.1f}  {self.epoch_results[0]['weights_before']['W1'][0][2]:.1f}")
        print(f"X2  {self.epoch_results[0]['weights_before']['W1'][1][0]:.1f}  {self.epoch_results[0]['weights_before']['W1'][1][1]:.1f}  {self.epoch_results[0]['weights_before']['W1'][1][2]:.1f}")
        
        print("\nW2 (Hidden ke Output):")
        print("      Y1")
        print(f"H1  {self.epoch_results[0]['weights_before']['W2'][0][0]:.1f}")
        print(f"H2  {self.epoch_results[0]['weights_before']['W2'][1][0]:.1f}")
        print(f"H3  {self.epoch_results[0]['weights_before']['W2'][2][0]:.1f}")
        
        print(f"\nBias Hidden: {self.epoch_results[0]['weights_before']['b1'][0]}")
        print(f"Bias Output: {self.epoch_results[0]['weights_before']['b2'][0]}")
        
        print("\n2. TABEL UNTUK 3 EPOCH PERTAMA:")
        print("| Epoch | Pattern | X1 | X2 | Target | Output | Error  | MSE    |")
        print("|-------|---------|----|----|--------|--------|--------|--------|")
        
        for epoch_idx in range(min(3, len(self.epoch_results))):
            epoch_data = self.epoch_results[epoch_idx]
            for pattern in epoch_data['patterns']:
                print(f"|   {epoch_data['epoch']}   |    {pattern['pattern_num']}    | {pattern['input'][0]}  | {pattern['input'][1]}  |   {pattern['target']}    | {pattern['output']:.3f}  | {pattern['error']:.3f}  | {pattern['mse']:.3f}  |")

# Main execution
if __name__ == "__main__":
    # XOR Dataset
    X = np.array([[0, 0],
                  [0, 1], 
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1], 
                  [0]])
    
    print("DATASET XOR:")
    print("Input (X1, X2) -> Output (Y)")
    for i in range(len(X)):
        print(f"({X[i][0]}, {X[i][1]}) -> {y[i][0]}")
    
    # Create and train network
    nn = NeuralNetworkAbsen1()
    nn.train(X, y, epochs=15, target_error=0.01)
    
    # Plot results
    nn.plot_results()
    
    # Export data for manual calculation
    nn.export_manual_calculation_data()
    
    print("\n" + "="*60)
    print("INSTRUKSI PENGERJAAN DI KERTAS:")
    print("="*60)
    print("1. Gunakan bobot awal di atas")
    print("2. Hitung 3 epoch pertama secara manual")
    print("3. Bandingkan dengan hasil kode untuk verifikasi")
    print("4. Learning rate = 0.1")
    print("5. Arsitektur: 2 Input -> 3 Hidden -> 1 Output")
    print("6. Aktivasi: Sigmoid")
    print("7. Update bobot setelah setiap pattern (online learning)")
    
    # Fungsi sigmoid reference
    print("\nREFERENSI FUNGSI SIGMOID:")
    print("σ(x) = 1/(1+e^(-x))")
    test_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for val in test_values:
        sigmoid_val = 1/(1+np.exp(-val))
        print(f"σ({val}) = {sigmoid_val:.3f}")