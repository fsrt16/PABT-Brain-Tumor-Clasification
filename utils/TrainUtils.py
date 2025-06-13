import os
import time
import torch
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard


class TrainingTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_train_end(self, logs=None):
        self.train_end_time = time.time()
        duration = self.train_end_time - self.train_start_time
        print(f"Training completed in {duration // 60:.0f} min {duration % 60:.2f} sec")


def compile_model(model, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    return model


def get_tf_callbacks(model_name='pabt_result'):
    weight_path = f"{model_name}.weights.h5"

    checkpoint = ModelCheckpoint(
        weight_path,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.8,
        patience=10,
        verbose=1,
        mode='auto',
        min_lr=0.0001,
        cooldown=5
    )

    early_stop = EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=20,
        restore_best_weights=True
    )

    log_dir = os.path.join("logs", model_name, time.strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    return [checkpoint, reduce_lr, early_stop, tensorboard, TrainingTimer()]


def train_tf_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    return history


class pytorchIMPL:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, train_loader, val_loader, epochs=25):
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_loader):.4f}")
            self.evaluate(val_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        print(f"Validation Loss: {val_loss/len(data_loader):.4f}, Accuracy: {accuracy*100:.2f}%")


class tfIMPL:
    def __init__(self, model, model_name='pabt_result'):
        self.model = model
        self.model_name = model_name
        self.callbacks = get_tf_callbacks(model_name)

    def compile_and_train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        compile_model(self.model)
        history = train_tf_model(self.model, X_train, y_train, X_val, y_val, epochs, batch_size, self.callbacks)
        return history
