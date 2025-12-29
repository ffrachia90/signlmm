"""
SignLMM POC - Entrenamiento del Modelo
======================================
Entrena un clasificador de señas usando LSTM bidireccional.

Uso:
    python scripts/train_model.py --data data/processed --output models/
"""

import os
import json
import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SignClassifier(nn.Module):
    """
    Clasificador de señas usando LSTM bidireccional.
    
    Arquitectura:
    - Input: (batch, seq_len, features)
    - LSTM bidireccional de 2 capas
    - Fully connected para clasificación
    """
    
    def __init__(
        self,
        input_size: int = 150,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 50,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer simple
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * 2)
        
        # Attention
        attention_weights = self.attention(lstm_out)
        # attention_weights: (batch, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)
        # context: (batch, hidden_size * 2)
        
        # Clasificación
        output = self.classifier(context)
        
        return output


class EarlyStopping:
    """Early stopping para evitar overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def load_dataset(data_dir: str, normalize: bool = True):
    """Carga el dataset procesado."""
    data_path = Path(data_dir)
    
    X_train = np.load(data_path / 'X_train.npy')
    X_test = np.load(data_path / 'X_test.npy')
    y_train = np.load(data_path / 'y_train.npy')
    y_test = np.load(data_path / 'y_test.npy')
    
    # Normalizar
    if normalize:
        mean = np.load(data_path / 'mean.npy')
        std = np.load(data_path / 'std.npy')
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
    
    # Cargar metadata
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Cargar label encoder
    with open(data_path / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return X_train, X_test, y_train, y_test, metadata, label_encoder


def train_model(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 10,
    device: str = None
):
    """
    Entrena el modelo de clasificación de señas.
    """
    # Detectar device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Usando device: {device}")
    
    # Cargar datos
    print("Cargando dataset...")
    X_train, X_test, y_train, y_test, metadata, label_encoder = load_dataset(data_dir)
    
    num_classes = metadata['num_classes']
    input_size = metadata['feature_dim']
    
    print(f"Train: {len(X_train)} muestras")
    print(f"Test: {len(X_test)} muestras")
    print(f"Clases: {num_classes}")
    print(f"Features por frame: {input_size}")
    
    # Crear DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    model = SignClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    print(f"\nModelo creado:")
    print(f"  - Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss y optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    early_stopping = EarlyStopping(patience=patience)
    
    # Tracking
    best_accuracy = 0
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    
    # Entrenamiento
    print("\nIniciando entrenamiento...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Eval
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = correct / total
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(accuracy)
        
        # Learning rate scheduler
        scheduler.step(test_loss)
        
        # Mostrar progreso
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:.2%}")
        
        # Guardar mejor modelo
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'config': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'num_classes': num_classes,
                    'dropout': dropout
                }
            }, output_path / 'best_model.pt')
        
        # Early stopping
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping en epoch {epoch+1}")
            break
    
    # Guardar historial
    output_path = Path(output_dir)
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Copiar label encoder y metadata
    import shutil
    shutil.copy(Path(data_dir) / 'label_encoder.pkl', output_path / 'label_encoder.pkl')
    shutil.copy(Path(data_dir) / 'mean.npy', output_path / 'mean.npy')
    shutil.copy(Path(data_dir) / 'std.npy', output_path / 'std.npy')
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"   Mejor accuracy: {best_accuracy:.2%}")
    print(f"   Modelo guardado en: {output_path / 'best_model.pt'}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Entrena modelo de clasificación de señas')
    parser.add_argument('--data', '-d', required=True, help='Directorio con dataset procesado')
    parser.add_argument('--output', '-o', required=True, help='Directorio para guardar modelo')
    parser.add_argument('--epochs', type=int, default=100, help='Número de epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM hidden size (default: 128)')
    parser.add_argument('--num-layers', type=int, default=2, help='LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout (default: 0.3)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device
    )


if __name__ == '__main__':
    main()


