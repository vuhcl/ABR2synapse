import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.ABRA_35.models import CNN


class SimplerImprovedCNN(nn.Module):
    def __init__(self, use_batch_norm=False):
        super(SimplerImprovedCNN, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1)
        )

    def forward(self, x):
        if self.use_batch_norm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout(x)
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.dropout(x)
            x = F.relu(self.conv3(x))

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class MinimalWaveICNN(nn.Module):
    def __init__(self):
        super(MinimalWaveICNN, self).__init__()

        # Extremely simple for 30 points
        self.conv1 = nn.Conv1d(
            1, 4, kernel_size=3, padding=1
        )  # 30 -> 30, only 4 filters
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # Minimal FC
        self.fc = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))  # No batch norm, no dropout
        x = self.pool(x).squeeze(-1)  # (batch_size, 4)
        x = self.fc(x)
        return x


class WaveICNN(nn.Module):
    def __init__(self, dropout_rate=0.3, use_batch_norm=False):
        super(WaveICNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)  # 30 -> 30
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)  # 30 -> 30

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(32)

        self.pool = nn.MaxPool1d(2)  # 30 -> 15 -> 7 (with padding)

        self.dropout = nn.Dropout(dropout_rate)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(16, 1)
        )

    def forward(self, x):
        if self.use_batch_norm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)  # (batch_size, 16, 15)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)  # (batch_size, 32, 7)
        else:
            x = F.relu(self.conv1(x))
            x = self.pool(x)  # (batch_size, 16, 15)

            x = F.relu(self.conv2(x))
            x = self.pool(x)  # (batch_size, 32, 7)

        x = self.dropout(x)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch_size, 32)

        # Fully connected layers
        x = self.fc(x)
        return x


def multi_task_loss(
    synapse_pred,
    amplitude_pred,
    noise_pred,
    synapse_true,
    amplitude_true,
    noise_true,
    synapse_weight=1.0,
    amplitude_weight=1.0,
    noise_weight=1.0,
):
    """
    Multi-task loss combining synapse prediction and amplitude prediction
    """
    synapse_loss = F.mse_loss(synapse_pred.squeeze(), synapse_true)
    amplitude_loss = F.mse_loss(amplitude_pred.squeeze(), amplitude_true)
    noise_loss = F.mse_loss(noise_pred.squeeze(), noise_true)

    total_loss = (
        synapse_weight * synapse_loss
        + amplitude_weight * amplitude_loss
        + noise_weight * noise_loss
    )
    return total_loss, synapse_loss, amplitude_loss, noise_loss


class FeaturePredictorCNN(nn.Module):
    def __init__(self, use_batch_norm=False):
        super(FeaturePredictorCNN, self).__init__()

        # Modified CNN backbone to expose intermediate features
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5)
        )

        # Prediction heads
        self.amplitude_head = nn.Linear(64, 1)  # Predict amplitude from waveform
        self.noise_head = nn.Linear(64, 1)
        self.synapse_head = nn.Linear(66, 1)  # 64 CNN features + 1 amplitude prediction

    def extract_features(self, x):
        if self.use_batch_norm:
            """Extract CNN features from waveform"""
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout(x)
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            x = self.dropout(x)
            x = F.relu(self.conv3(x))

        x = self.global_pool(x).squeeze(-1)
        x = self.feature_layer(x)
        return x

    def forward(self, waveform):
        # Extract features from waveform
        cnn_features = self.extract_features(waveform)

        # Predict amplitude from CNN features
        pred_amplitude = self.amplitude_head(cnn_features)

        pred_noise = self.noise_head(cnn_features)

        # Combine CNN features + predicted amplitude for synapse prediction
        combined = torch.cat([cnn_features, pred_amplitude, pred_noise], dim=1)
        synapses = self.synapse_head(combined)

        return synapses, pred_amplitude, pred_noise


class FeatureGuidedCNN(nn.Module):
    def __init__(self):
        super(FeatureGuidedCNN, self).__init__()

        # CNN path for learning representations
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)

        # Global pooling to extract summary statistics
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Combine CNN features + hand-crafted features
        # 32 (CNN max) + 5 (manual features) = 37 total features
        self.feature_combiner = nn.Sequential(
            nn.Linear(37, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Prediction heads
        self.amplitude_head = nn.Linear(32, 1)
        self.noise_head = nn.Linear(32, 1)
        self.synapse_head = nn.Linear(34, 1)  # 32 + 2 (predicted amplitude)

    def extract_manual_features(self, x):
        """Extract the features we know correlate well with amplitude"""
        # x shape: (batch_size, 1, seq_len)
        x_squeezed = x.squeeze(1)  # (batch_size, seq_len)

        peak = torch.max(x_squeezed, dim=1)[0]
        trough = torch.min(x_squeezed, dim=1)[0]
        peak_to_peak = peak - trough
        std = torch.std(x_squeezed, dim=1)
        mean_abs = torch.mean(torch.abs(x_squeezed), dim=1)

        return torch.stack([peak, trough, peak_to_peak, std, mean_abs], dim=1)

    def forward(self, x):
        # CNN features
        conv_out = F.relu(self.conv1(x))
        conv_out = F.relu(self.conv2(conv_out))

        # Extract pooled representations
        max_features = self.global_max_pool(conv_out).squeeze(-1)  # (batch_size, 32)

        # Manual features that we know work
        manual_features = self.extract_manual_features(x)  # (batch_size, 5)

        # Combine all features
        combined_features = torch.cat(
            [max_features, manual_features], dim=1
        )  # (batch_size, 37)

        # Process combined features
        processed_features = self.feature_combiner(
            combined_features
        )  # (batch_size, 32)

        # Predict amplitude
        amplitude_pred = self.amplitude_head(processed_features)

        noise_pred = self.noise_head(processed_features)

        # Predict synapses using processed features + predicted amplitude
        synapse_input = torch.cat(
            [processed_features, amplitude_pred, noise_pred], dim=1
        )
        synapse_pred = self.synapse_head(synapse_input)

        return synapse_pred, amplitude_pred, noise_pred


def train_cnn_model(
    train_loader, val_loader, num_epochs=100, early_stop=True, model_type="default"
):
    """
    Train the CNN model
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if model_type == "default":
        model = CNN(128, 32, 0.5, 0.3, 0.1)
        model.load_state_dict(torch.load("./utils/ABRA_35/models/waveI_cnn.pth"))
    elif model_type == "improved":
        model = SimplerImprovedCNN()
    elif model_type == "wavei":
        model = WaveICNN(dropout_rate=0.3)
    elif model_type == "minimal_wavei":
        model = MinimalWaveICNN()
    elif model_type == "feature_pred":
        model = FeaturePredictorCNN()
    elif model_type == "feature_guided":
        model = FeatureGuidedCNN()
    model.train()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    if model_type == "minimal_wavei":
        optimizer = optim.Adam(
            model.parameters(), lr=0.001, weight_decay=1e-6
        )  # Less regularization
    elif model_type == "wavei":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    # early stopping
    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    # Training loop
    train_losses = []
    val_losses = []
    train_synapse_losses = []
    train_amplitude_losses = []
    train_noise_losses = []

    print_every = num_epochs // 10 if num_epochs >= 10 else 1
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_syn_loss, train_amp_loss, train_noise_loss = 0.0, 0.0, 0.0

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            if model_type == "feature_pred" or model_type == "feature_guided":
                waveform, synapse_target, amplitude_target, noise_target = data
                synapse_pred, amplitude_pred, noise_pred = model(waveform)
                loss, syn_loss, amp_loss, noise_loss = multi_task_loss(
                    synapse_pred,
                    amplitude_pred,
                    noise_pred,
                    synapse_target,
                    amplitude_target,
                    noise_target,
                )
            else:
                data, target = data
                output = model(data)
                loss = criterion(output.squeeze(), target)

            loss.backward()

            if model == "wavei":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            if model_type == "feature_pred" or model_type == "feature_guided":
                train_syn_loss += syn_loss.item()
                train_amp_loss += amp_loss.item()
                train_noise_loss += noise_loss.item()

        # Validation phase
        model.eval()
        val_loss, val_syn_loss, val_amp_loss, val_noise_loss = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for data in val_loader:
                if model_type == "feature_pred" or model_type == "feature_guided":
                    waveform, synapse_target, amplitude_target, noise_target = data
                    synapse_pred, amplitude_pred, noise_pred = model(waveform)
                    loss, syn_loss, amp_loss, noise_loss = multi_task_loss(
                        synapse_pred,
                        amplitude_pred,
                        noise_pred,
                        synapse_target,
                        amplitude_target,
                        noise_target,
                    )
                    val_syn_loss += syn_loss.item()
                    val_amp_loss += amp_loss.item()
                    val_noise_loss += noise_loss.item()
                else:
                    data, target = data
                    output = model(data)
                    loss = criterion(output.squeeze(), target)
                val_loss += loss.item()

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_syn_loss = (
            train_syn_loss / len(train_loader)
            if model_type == "feature_pred" or model_type == "feature_guided"
            else 0
        )
        avg_train_amp_loss = (
            train_amp_loss / len(train_loader)
            if model_type == "feature_pred" or model_type == "feature_guided"
            else 0
        )
        avg_train_noise_loss = (
            train_noise_loss / len(train_loader)
            if model_type == "feature_pred" or model_type == "feature_guided"
            else 0
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_synapse_losses.append(avg_train_syn_loss)
        train_amplitude_losses.append(avg_train_amp_loss)
        train_noise_losses.append(avg_train_noise_loss)

        # Learning rate scheduling
        if model_type != "minimal_wavei":
            scheduler.step(avg_val_loss)

        # print progress
        if epoch % print_every == 0:
            if model_type == "feature_pred" or model_type == "feature_guided":
                print(
                    f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f} (Syn: {avg_train_syn_loss:.4f}, Amp: {avg_train_amp_loss:.4f}, Noise: {avg_train_noise_loss:.4f}), Val Loss: {avg_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}"
                )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience and early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if model_type == "feature_pred" or model_type == "feature_guided":
        return (
            model,
            train_losses,
            val_losses,
            train_synapse_losses,
            train_amplitude_losses,
            train_noise_losses,
        )
    else:
        return model, train_losses, val_losses
