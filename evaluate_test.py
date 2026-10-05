# evaluate_test.py
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# import the train to test
from trainexportfull import EmotionCNN, EmotionDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data and select the held-out split
df = pd.read_csv('emotion_dataset.csv')
test_df = df[df['Usage'] == 'PrivateTest']
print("Test samples:", len(test_df))

# Same preprocessing as validation (no augmentation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
loader = DataLoader(EmotionDataset(test_df, transform=transform),
                    batch_size=64, shuffle=False)

# Load epoch 22 weights 
model = EmotionCNN(num_classes=2).to(device)
model.load_state_dict(torch.load('exports/emotion_model_epoch_22.pth',
                                 map_location=device))
model.eval()

preds, labels = [], []
with torch.no_grad():
    for x, y in loader:
        out = model(x.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())

cm = confusion_matrix(labels, preds)
print("Confusion matrix:\n", cm)

# Happy (class 0) is the positive class
tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
accuracy = (tp + tn) / cm.sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = 2 * precision * recall / (precision + recall)

print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score:    {f1:.4f}")