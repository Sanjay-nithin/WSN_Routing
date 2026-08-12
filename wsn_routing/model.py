import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNClassifier(nn.Module):
    """
    1-D Convolutional Neural Network (CNN) for WSN node health classification.
    Takes tabular features and treats them as a 1-D sequence (1 channel, sequence_length = num_features).
    """
    def __init__(self, in_features, num_classes=3):
        super().__init__()
        self.in_features = in_features
        
        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(0.3)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, in_features)
        # Reshape to (batch_size, 1, in_features) for 1-D Conv
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Global Average Pooling (GAP) across features: (batch_size, 128)
        x = x.mean(dim=2)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @torch.no_grad()
    def predict_proba(self, x):
        """Returns softmax probabilities."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, x):
        """Returns class predictions (0, 1, or 2)."""
        self.eval()
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def train_model(model, train_loader, test_loader, num_epochs=30, lr=0.001, device='cpu', auto_balance=True):
    """
    Trains the CNN model, logs performance metrics, and returns the trained model and history.
    """
    model = model.to(device)
    
    # Calculate class weights for cross entropy loss if auto_balance is True
    loss_fn = nn.CrossEntropyLoss()
    if auto_balance:
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        class_counts = np.bincount(all_labels)
        total_samples = len(all_labels)
        num_classes = len(class_counts)
        # Standard balanced weights: total_samples / (num_classes * class_count)
        weights = total_samples / (num_classes * class_counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"[Trainer] Automatically computed class weights: {weights}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Evaluation Phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                
                test_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == y_batch).sum().item()
                test_total += y_batch.size(0)
                
        epoch_test_loss = test_loss / test_total
        epoch_test_acc = test_correct / test_total
        
        # Step LR Scheduler
        scheduler.step(epoch_test_acc)
        
        # Log metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)
        
        # Save best model
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            best_model_state = model.state_dict().copy()
            
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f} - Test Acc: {epoch_test_acc*100:.2f}%")
              
    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[Trainer] Restored best model weights with Test Acc: {best_test_acc*100:.2f}%")
        
    return model, history


def get_predictions(model, loader, device='cpu'):
    """Helper to collect predictions and ground truths from a DataLoader."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=-1)
            preds = outputs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)
