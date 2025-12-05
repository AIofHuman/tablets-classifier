import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter

class OGYEIDataset(Dataset):
    def __init__(self, root_dir, subset, number_split=1, transform=None):
        """
        Args:
            root_dir (str): Директория с изображениями
            number_split (int): Количество частей имени файла для формирования метки
            transform (callable, optional): Трансформации для изображений
        """
        self.root_dir = root_dir
        self.subset = subset
        self.number_split = number_split
        self.transform = transform
        
        # Собираем все файлы и соответствующие метки
        self.dataset = pd.DataFrame(columns=['image_path', 'class_name', 'class_idx'])        
        
        # Список поддерживаемых расширений изображений
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
        
        
        path_to_images = os.path.join(root_dir, subset, 'images')
        if not os.path.exists(path_to_images):
            raise ValueError(f"Folder {path_to_images} not found!")
        
        for filename in os.listdir(path_to_images):
            filepath = os.path.join(path_to_images, filename)
            
            # Проверяем, что это файл и имеет правильное расширение
            if (os.path.isfile(filepath) and 
                any(filename.lower().endswith(ext) for ext in img_extensions)):
                
                # Извлекаем метку из имени файла
                parts = os.path.splitext(filename)[0].split('_')
                label = '_'.join(parts[:self.number_split]) if len(parts) >= self.number_split else parts[0]
                
                self.dataset.loc[len(self.dataset),:] = [filepath, label, -1]

        # Создаем маппинг классов в индексы
        self.classes = sorted(self.dataset['class_name'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Обновляем индексы классов в DataFrame
        self.dataset['class_idx'] = self.dataset['class_name'].map(self.class_to_idx)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        row = self.dataset.iloc[idx]
        filepath = row['image_path']
        label = row['class_idx']
        
        # Загружаем изображение
        try:
            image = Image.open(filepath).convert('RGB')
        except Exception as e:
            print(f"Cann't load image {filepath}: {e}")
            # Возвращаем черное изображение того же размера
            image = Image.new('RGB', (224, 224), color='black')
        
        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Возвращает распределение классов в датасете"""
        return Counter(self.dataset['class_name'])
    
    def get_class_weights(self, method='balanced'):
        """
        Вычисляет веса классов для балансировки
        
        Args:
            method (str): Метод расчета весов
                - 'balanced': обратно пропорционально количеству образцов
                - 'sqrt': обратно пропорционально квадратному корню
                - 'uniform': одинаковые веса для всех классов
        
        Returns:
            torch.Tensor: Веса классов
        """
        import torch
        from torch import tensor
        
        class_counts = self.dataset['class_idx'].value_counts().sort_index().values
        n_classes = len(class_counts)
        
        if method == 'balanced':
            weights = 1.0 / class_counts
        elif method == 'sqrt':
            weights = 1.0 / np.sqrt(class_counts)
        elif method == 'uniform':
            weights = np.ones(n_classes)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        # Нормализуем веса
        weights = weights / weights.sum() * n_classes
        
        return tensor(weights, dtype=torch.float32)
    



# Пример использования с трансформациями
def get_transforms(augment=False, img_size=224):
    """
    Создает трансформации для изображений
    
    Args:
        augment (bool): Если True, добавляет аугментации
        img_size (int): Размер изображения после ресайза
    """
    if augment:
        # Трансформации для тренировочных данных (с аугментацией)
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return train_transform
    else:
        # Трансформации для валидационных/тестовых данных (без аугментации)
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return val_transform


# Пример использования
if __name__ == "__main__":
    # Создаем датасет с трансформациями
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # data находится на одном уровне с src, поэтому поднимаемся на одну директорию вверх
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
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
    
    # Выводим информацию о датасете
    print(f"Количество изображений в тренировочном наборе: {len(train_dataset)}")
    print(f"Количество классов: {len(train_dataset.classes)}")
    print(f"Классы: {train_dataset.classes}")

    # Выводим информацию о датасете
    print(f"Количество изображений в validation наборе: {len(val_dataset)}")
    print(f"Количество классов: {len(val_dataset.classes)}")
    print(f"Классы: {val_dataset.classes}")

    # Выводим информацию о датасете
    print(f"Количество изображений в тренировочном наборе: {len(test_dataset)}")
    print(f"Количество классов: {len(test_dataset.classes)}")
    print(f"Классы: {test_dataset.classes}")
    
    
