import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import timm
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


from OGYEI_dataset import OGYEIDataset, get_transforms


class OGYEI_EfficientNetClassifier(nn.Module):
    """
    Модель на базе EfficientNet для классификации таблеток
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Количество классов для классификации
            model_name: Название EfficientNet модели
            pretrained: Использовать предобученные веса ImageNet
            dropout_rate: Dropout в классификаторе
            freeze_backbone: Заморозить backbone при инициализации
        """
        super().__init__()
        
        # Загружаем предобученную EfficientNet
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        
        # Получаем размерность features
        self.feature_dim = self.model.num_features
        
        # Создаем новый классификатор для наших классов
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Замораживаем backbone если нужно
        if freeze_backbone:
            self._freeze_backbone()
        
        print(f"Модель создана: {model_name}")
        print(f"Количество классов: {num_classes}")
        print(f"Размерность features: {self.feature_dim}")
        print(f"Backbone заморожен: {freeze_backbone}")
    
    def _freeze_backbone(self):
        """Замораживает все слои backbone"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Размораживает все слои backbone"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: Входные изображения [batch, channels, height, width]
            
        Returns:
            Логиты для каждого класса [batch, num_classes]
        """
        features = self.model(x)
        output = self.classifier(features)
        return output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Возвращает вероятности классов
        
        Args:
            x: Входные изображения
            
        Returns:
            Вероятности классов [batch, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Возвращает предсказанные классы
        
        Args:
            x: Входные изображения
            
        Returns:
            Предсказанные классы [batch]
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int=100,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    patience: int = 10,
    checkpoint_path: str = "models/meds_classifier.pt"
) -> Dict:
    """
    Простая функция обучения модели
    
    Args:
        model: Модель для обучения
        train_loader: DataLoader для тренировочных данных
        val_loader: DataLoader для валидационных данных
        num_classes: Количество классов
        num_epochs: Количество эпох
        learning_rate: Learning rate
        weight_decay: Weight decay для регуляризации
        device: Устройство для обучения
        patience: Количество эпох для ранней остановки
        checkpoint_path: Путь для сохранения лучшей модели
        
    Returns:
        Словарь с историей обучения
    """
    
    # Перемещаем модель на устройство
    model = model.to(device)
    
    # Loss функция с label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Оптимизатор
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler для learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # Для mixed precision training
    scaler = GradScaler(device)
    
    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Для ранней остановки
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    print(f"Начинаем обучение на {device}...")
    print(f"Размер тренировочного набора: {len(train_loader.dataset)}")
    print(f"Размер валидационного набора: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Тренировка
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for _, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Mixed precision forward pass
            with autocast(device_type=device):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Статистика
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct_step = predicted.eq(labels).sum().item()
            train_correct += train_correct_step
            
            # Обновляем progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * train_correct_step / labels.size(0)
            })
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct_step = predicted.eq(labels).sum().item()
                val_correct += val_correct_step
                
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * val_correct_step / labels.size(0)
                })
        
        # Вычисляем средние метрики
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Обновляем scheduler
        scheduler.step()
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            epochs_no_improve = 0
            print(f"✓ Модель сохранена! Val Acc: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
        
        # Печатаем статистику эпохи
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        print(f"  Epochs no improve: {epochs_no_improve}/{patience}")
        
        # Проверка ранней остановки
        if epochs_no_improve >= patience:
            print(f"\nРанняя остановка на эпохе {epoch+1}")
            break
    
    print(f"\nОбучение завершено!")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
    
    # Загружаем лучшую модель
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return history, model


def evaluate_model(
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda"
    ) -> Dict:
    """
    Оценка модели на тестовом наборе
    
    Args:
        model: Обученная модель
        test_loader: DataLoader для тестовых данных
        device: Устройство для вычислений
        
    Returns:
        Словарь с метриками
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Evaluation')
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct_step = predicted.eq(labels).sum().item()
            test_correct += test_correct_step
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            test_bar.set_postfix({
                'acc': 100. * test_correct_step / labels.size(0)
            })
    
    test_acc = 100. * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\nТестирование завершено:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Correct/Total: {test_correct}/{test_total}")
    
    return {
        'test_loss': avg_test_loss,
        'test_acc': test_acc,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'confusion_matrix': cm,
        'class_accuracy': class_acc
    }


def plot_training_history(history: Dict, save_path: str = None):
    """
    Визуализация истории обучения
    
    Args:
        history: История обучения из train_model
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(history['learning_rates'], label='LR', linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_model_and_train(
    checkpoint_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
    num_classes: int = 100,
    model_name: str = "efficientnet_b3",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    device: str = None,
) -> Tuple[nn.Module, Dict]:
    """
    Полный пайплайн: создание модели и обучение
    
    Args:
        train_loader: DataLoader для тренировки
        val_loader: DataLoader для валидации
        test_loader: DataLoader для тестирования (опционально)
        num_classes: Количество классов
        model_name: Название EfficientNet модели
        pretrained: Использовать предобученные веса
        freeze_backbone: Начинать с замороженного backbone
        num_epochs: Количество эпох обучения
        batch_size: Размер батча
        learning_rate: Learning rate
        device: Устройство для обучения
        
    Returns:
        model: Обученная модель
        history: История обучения
    """
    
    # Определяем устройство
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Используемое устройство: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Создаем модель
    print("\n" + "="*60)
    print("Создание модели...")
    model = OGYEI_EfficientNetClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    # Считаем параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,} ({trainable_params/total_params:.1%})")
    
    # Обучаем модель
    print("\n" + "="*60)
    print("Начало обучения...")
    
    history, trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        checkpoint_path=checkpoint_path
    )
    
    # Визуализируем историю обучения
    print("\n" + "="*60)
    print("Визуализация истории обучения...")
    plot_training_history(history, save_path="training_history.png")
    
    # Оценка на тестовом наборе если есть
    if test_loader is not None:
        print("\n" + "="*60)
        print("Оценка на тестовом наборе...")
        test_results = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            device=device
        )
        
        # Визуализация confusion matrix (для первых 20 классов)
        if test_results['confusion_matrix'].shape[0] > 20:
            cm = test_results['confusion_matrix'][:20, :20]
        else:
            cm = test_results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.title('Confusion Matrix (первые 20 классов)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return trained_model, history


# Простой пример использования
if __name__ == "__main__":
    # Импортируем ваши датасеты
    data_dir = 'tablets-classifier/data'

    train_dataset = OGYEIDataset(
        root_dir=data_dir,
        subset='train',
        number_split=1,  # Берем первую часть имени файла как метку
        transform=get_transforms(augment=True)  # Трансформации с аугментацией
    )
        
    val_dataset = OGYEIDataset(
        root_dir=data_dir,
        subset='valid',
        number_split=1,
        transform=get_transforms(augment=False)  # Трансформации без аугментации
    )

    test_dataset = OGYEIDataset(
        root_dir=data_dir,
        subset='test',
        number_split=1,
        transform=get_transforms(augment=False)  # Трансформации без аугментации
    )
    
    # Создаем DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Запускаем полный пайплайн
    model, history = create_model_and_train(
        checkpoint_path = './models/meds_classifier.pt',
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=len(train_dataset.classes),  # Автоматически определяем количество классов
        model_name="efficientnet_b3",
        pretrained=True,
        freeze_backbone=True,  # Начинаем с замороженного backbone
        num_epochs=30,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Пример предсказания на одном изображении
    print("\n" + "="*60)
    print("Пример предсказания...")
    
    # Берем один батч
    images, labels = next(iter(test_loader))
    single_image = images[0:1]  # Берем одно изображение
    
    # Получаем предсказание
    model.eval()
    with torch.no_grad():
        logits = model(single_image.to(next(model.parameters()).device))
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    
    print(f"Предсказанный класс: {predicted_class}")
    print(f"Вероятность: {probs[0, predicted_class]:.4f}")
    print(f"Топ-3 класса: {torch.topk(probs[0], 3).indices.tolist()}")
    print(f"Топ-3 вероятности: {torch.topk(probs[0], 3).values.tolist()}")