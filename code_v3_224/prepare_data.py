import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# إعدادات
SAVE_DIR = "data/cifar100_224"
BATCH_SIZE = 128
NUM_WORKERS = 6  # يمكنك زيادتها هنا لأننا سنشغله كسكربت منفصل

def pre_process_and_save():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print("Preparing 224x224 dataset...")
    
    # التحويل: تكبير فقط
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # تحميل البيانات الأصلية
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=resize_transform)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=resize_transform)

    # دالة للحفظ
    def save_dataset(dataset, name):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        all_images = []
        all_labels = []
        
        print(f"Processing {name} set...")
        for imgs, labels in tqdm(loader):
            all_images.append(imgs)
            all_labels.append(labels)
        
        # تجميع وحفظ
        images_tensor = torch.cat(all_images)
        labels_tensor = torch.cat(all_labels)
        
        torch.save((images_tensor, labels_tensor), os.path.join(SAVE_DIR, f"{name}.pt"))
        print(f"Saved {name}.pt: {images_tensor.shape}")

    save_dataset(train_set, "train")
    save_dataset(test_set, "test")

if __name__ == '__main__':
    pre_process_and_save()