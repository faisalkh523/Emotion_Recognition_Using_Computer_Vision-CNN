# train_and_export_2emotions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
import sys
import warnings
import traceback
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# MODIFIED: Emotion classes from dataset (2 emotions)
EMOTIONS = ['happy', 'sad']  # Changed from 7 to 2 emotions

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=2):  # MODIFIED: Changed from 7 to 2
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)  # MODIFIED: Will output 2 classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.global_pool(x)
        x = x.view(-1, 256 * 6 * 6)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ExportFriendlyEmotionCNN(nn.Module):
    """A version of EmotionCNN that's more export-friendly for ONNX/TorchScript"""
    def __init__(self, original_model):
        super(ExportFriendlyEmotionCNN, self).__init__()
        # Copy all layers except adaptive pool
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.conv2 = original_model.conv2
        self.bn2 = original_model.bn2
        self.conv3 = original_model.conv3
        self.bn3 = original_model.bn3
        self.conv4 = original_model.conv4
        self.bn4 = original_model.bn4
        self.pool = original_model.pool
        self.dropout = original_model.dropout

        # fixed pooling for export
        self.upsample = nn.Upsample(size=(6, 6), mode='bilinear', align_corners=False)

        self.fc1 = original_model.fc1
        self.fc2 = original_model.fc2
        self.fc3 = original_model.fc3

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Use upsample instead of adaptive pool for export compatibility
        x = self.upsample(x)

        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# MODIFIED: Renamed dataset class to be more generic
class EmotionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pixels = self.dataframe.iloc[idx]['pixels']
        emotion = int(self.dataframe.iloc[idx]['emotion'])

        # Convert string of pixels to numpy array
        image = np.array([int(pixel) for pixel in pixels.split()], dtype=np.uint8)
        image = image.reshape(48, 48)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, emotion

def check_and_prepare_data():
    """Check if data is available and prepare it if needed"""
    print("\n" + "="*70)
    print("CHECKING AND PREPARING DATA")
    print("="*70)

    # MODIFIED: Look for emotion_dataset.csv first
    if os.path.exists('emotion_dataset.csv'):
        csv_file = 'emotion_dataset.csv'
        print(f"Found your custom dataset: {csv_file}")
    else:
        # Fallback to FER2013 CSV if exists (for backward compatibility)
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'fer2013' in f.lower()]

        if not csv_files:
            print("Error: No emotion_dataset.csv file found!")
            print("\nPlease make sure you have created emotion_dataset.csv using the data preparation script.")
            print("The file should be in the current directory.")
            return None

        csv_file = csv_files[0]
        print(f"Found data file: {csv_file}")

    try:
        # Try to load the data
        print("Loading data...")
        dataframe = pd.read_csv(csv_file)
        print(f"Data shape: {dataframe.shape}")
        print(f"Columns: {list(dataframe.columns)}")

        # Check required columns
        required_columns = ['emotion', 'pixels']

        # MODIFIED: Check for Usage column (might not exist in  CSV)
        if 'Usage' not in dataframe.columns:
            print("Warning: 'Usage' column not found. Will split data automatically.")
            
            # Add Usage column with default values
            dataframe['Usage'] = 'Training'
            if len(dataframe) > 100:

                # Auto split: 70% train, 15% validation, 15% test
                from sklearn.model_selection import train_test_split
                train_idx, temp_idx = train_test_split(
                    dataframe.index, test_size=0.3, random_state=42,
                    stratify=dataframe['emotion'] if 'emotion' in dataframe.columns else None
                )
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=0.5, random_state=42,
                    stratify=dataframe.loc[temp_idx, 'emotion'] if 'emotion' in dataframe.columns else None
                )

                dataframe.loc[train_idx, 'Usage'] = 'Training'
                dataframe.loc[val_idx, 'Usage'] = 'PublicTest'
                dataframe.loc[test_idx, 'Usage'] = 'PrivateTest'

        required_columns.append('Usage')
        missing_columns = [col for col in required_columns if col not in dataframe.columns]

        if missing_columns:
            print(f"Error: CSV file missing required columns: {missing_columns}")
            print("Please check your CSV file format.")
            return None

        # Check unique emotion values and adjust EMOTIONS list
        unique_emotions = sorted(dataframe['emotion'].unique())
        print(f"\nUnique emotion values found: {unique_emotions}")

        # MODIFIED: Dynamically adjust EMOTIONS based on data
        global EMOTIONS
        if len(unique_emotions) == 2:
            # Use your emotion names if we have exactly 2
            EMOTIONS = ['happy', 'sad']
            print(f"Using 2 emotions: {EMOTIONS}")
        else:
            # If more than 2 emotions, use generic names
            EMOTIONS = [f'class_{i}' for i in range(len(unique_emotions))]
            print(f"Warning: Found {len(unique_emotions)} emotion classes. Using generic names.")

        # Check data distribution
        print("\nData distribution:")
        print(f"Total samples: {len(dataframe)}")
        print(f"Training samples: {len(dataframe[dataframe['Usage'] == 'Training'])}")
        print(f"PublicTest samples: {len(dataframe[dataframe['Usage'] == 'PublicTest'])}")
        print(f"PrivateTest samples: {len(dataframe[dataframe['Usage'] == 'PrivateTest'])}")

        # Check emotion distribution
        print("\nEmotion distribution:")
        for i, emotion in enumerate(EMOTIONS):
            count = len(dataframe[dataframe['emotion'] == i])
            print(f"  {emotion}: {count} samples")

        # Verify a few samples
        print("\nVerifying sample data...")
        sample = dataframe.iloc[0]
        emotion_idx = int(sample['emotion'])
        emotion_name = EMOTIONS[emotion_idx] if emotion_idx < len(EMOTIONS) else f'class_{emotion_idx}'
        print(f"Sample emotion: {sample['emotion']} ({emotion_name})")
        print(f"Pixel string length: {len(sample['pixels'])}")

        return dataframe

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        print(traceback.format_exc())
        return None


def export_key_formats(model, device, num_classes=2):  # MODIFIED: Added num_classes parameter
    """Export model to key formats only"""
    model.eval()

    # Create exports directory if it doesn't exist
    os.makedirs("exports", exist_ok=True)

    # 1. Save original PyTorch weights
    torch.save(model.state_dict(), 'exports/emotion_model_final.pth')
    print("✓ PyTorch weights saved: exports/emotion_model_final.pth")

    # 2. Create export-friendly model
    export_model = ExportFriendlyEmotionCNN(model)
    export_model.eval()

    # Copy weights from original model
    export_model.load_state_dict(model.state_dict())

    # 3. Export TorchScript
    try:
        print("\nExporting TorchScript model...")
        example_input = torch.randn(1, 1, 48, 48).to(device)

        # Trace the model
        traced_model = torch.jit.trace(export_model, example_input)
        traced_model.save('exports/emotion_model_torchscript.pt')

        # Verify the traced model
        loaded_model = torch.jit.load('exports/emotion_model_torchscript.pt')
        with torch.no_grad():
            test_output = loaded_model(example_input)
            if test_output.shape == (1, num_classes):  # MODIFIED: Check for num_classes
                print("✓ TorchScript model saved and verified: exports/emotion_model_torchscript.pt")
            else:
                print(f"✗ TorchScript model verification failed. Expected shape (1, {num_classes}), got {test_output.shape}")

    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")

        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': 'EmotionCNN',
            'input_shape': (1, 48, 48),
            'emotions': EMOTIONS,
            'num_classes': num_classes,
            'training_info': {
                'accuracy': 'Not recorded',  # Will be updated after training
                'best_epoch': 'Not recorded'
            }
        }, 'exports/emotion_model_complete.pth')
        print("✓ Complete PyTorch model saved: exports/emotion_model_complete.pth")
    except Exception as e:
        print(f"✗ Complete model save failed: {e}")

    # 7. Save model info
    model_info = {
        "emotions": EMOTIONS,
        "num_classes": num_classes,
        "input_size": [1, 48, 48],
        "normalization": {
            "mean": [0.5],
            "std": [0.5]
        },
        "model_architecture": "EmotionCNN",
        "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "available_formats": []
    }

    # List created files
    created_files = []
    for file in os.listdir('exports'):
        if file.endswith(('.pth', '.pt', '.json')):
            created_files.append(file)
            model_info["available_formats"].append(file.split('.')[-1])

    with open("exports/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print("✓ Model info saved: exports/model_info.json")

    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    print("Files created in 'exports' directory:")
    for file in created_files:
        print(f"  - {file}")

    # Create test scripts
    create_test_scripts(num_classes)

def create_test_scripts(num_classes=2):  # MODIFIED: Added num_classes parameter
    """Create test scripts for the exported model"""

    # Update the EmotionCNN class definition in test script
    emotion_cnn_class = f'''class EmotionCNN(torch.nn.Module):
    def __init__(self, num_classes={num_classes}):
        super(EmotionCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.global_pool = torch.nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = torch.nn.Linear(256 * 6 * 6, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.global_pool(x)
        x = x.view(-1, 256 * 6 * 6)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x'''

    # Main test script with testing
    test_script = f'''# test_model.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import sys
import os

print("="*70)
print("EMOTION RECOGNITION MODEL TEST (2 Emotions)")
print("="*70)

# Load model info
try:
    with open('exports/model_info.json', 'r') as f:
        model_info = json.load(f)
    EMOTIONS = model_info['emotions']
    NUM_CLASSES = model_info.get('num_classes', {num_classes})
    print(f"Model supports {{NUM_CLASSES}} emotions: {{', '.join(EMOTIONS)}}")
    print(f"Input shape: 1x48x48 (grayscale)")
except:
    print("Error: Could not load model info. Make sure exports/model_info.json exists")
    sys.exit(1)

# Simple test model class
{emotion_cnn_class}

def test_pytorch_model():
    """Test the PyTorch model"""
    print("\\n1. Testing PyTorch model...")

    try:
        # Load model
        model = EmotionCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load('exports/emotion_model_final.pth', map_location='cpu'))
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 1, 48, 48)

        # Run inference
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = F.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities).item()

        print(f"  ✓ Input shape: {{dummy_input.shape}}")
        print(f"  ✓ Output shape: {{output.shape}}")
        print(f"  ✓ Predicted emotion: {{EMOTIONS[predicted_idx]}}")
        print(f"  ✓ All probabilities:")
        for i, prob in enumerate(probabilities[0]):
            emotion_name = EMOTIONS[i] if i < len(EMOTIONS) else f'class_{{i}}'
            print(f"      {{emotion_name:10}}: {{prob:.2%}}")

        return True
    except Exception as e:
        print(f"  ✗ Error: {{e}}")
        return False

def test_torchscript_model():
    """Test the TorchScript model"""
    print("\\n2. Testing TorchScript model...")

    try:
        # Load TorchScript model
        model = torch.jit.load('exports/emotion_model_torchscript.pt', map_location='cpu')
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 1, 48, 48)

        # Run inference
        output = model(dummy_input)
        probabilities = F.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities).item()

        print(f"  ✓ Input shape: {{dummy_input.shape}}")
        print(f"  ✓ Output shape: {{output.shape}}")
        print(f"  ✓ Predicted emotion: {{EMOTIONS[predicted_idx]}}")

        return True
    except Exception as e:
        print(f"  ✗ Error: {{e}}")
        return False


def test_complete_model():
    """Test the complete saved model"""
    print("\\n4. Testing complete PyTorch model...")

    try:
        # Load complete model
        checkpoint = torch.load('exports/emotion_model_complete.pth', map_location='cpu')

        model = EmotionCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Architecture: {{checkpoint.get('model_architecture', 'Unknown')}}")
        print(f"  ✓ Emotions: {{checkpoint.get('emotions', 'Unknown')}}")

        return True
    except Exception as e:
        print(f"  ✗ Error: {{e}}")
        return False

if __name__ == "__main__":
    # Run all tests
    pytorch_success = test_pytorch_model()
    torchscript_success = test_torchscript_model()
    complete_success = test_complete_model()

    print("\\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    tests = [
        ("PyTorch Model", pytorch_success),
        ("TorchScript Model", torchscript_success),
        ("Complete Model", complete_success)
    ]

    all_passed = True
    for test_name, success in tests:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{{test_name:20}} {{status}}")
        all_passed = all_passed and success

    print("\\nTo test with your own image:")
    print("  python test_image.py <image_path>")
    print("  or")
    print("  python test_image.py (will use test_face.jpg if available)")

    if all_passed:
        print("\\n✓ All tests passed! Your model is ready to use.")
    else:
        print("\\nSome tests failed. Check the error messages above.")
'''

    with open('test_model.py', 'w') as f:
        f.write(test_script)

    # Image test script with TFLite support
    image_test_script = f'''# test_image.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import sys
import os

def load_model():
    """Load the best available model"""
    # Try TorchScript first (fastest)
    if os.path.exists('exports/emotion_model_torchscript.pt'):
        try:
            model = torch.jit.load('exports/emotion_model_torchscript.pt', map_location='cpu')
            print("Loaded TorchScript model (fast inference)")
            return model, 'torchscript'
        except:
            pass

    # Try complete model
    if os.path.exists('exports/emotion_model_complete.pth'):
        try:
            # Define model class
            {emotion_cnn_class}

            checkpoint = torch.load('exports/emotion_model_complete.pth', map_location='cpu')
            num_classes = checkpoint.get('num_classes', {num_classes})
            model = EmotionCNN(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded complete PyTorch model")
            return model, 'pytorch'
        except Exception as e:
            print(f"Error loading complete model: {{e}}")

    print("Error: No model found in exports directory")
    return None, None

def preprocess_image(image_path):
    """Preprocess an image for the model"""
    try:
        # Load image
        img = Image.open(image_path)

        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to 48x48
        img = img.resize((48, 48))

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Apply normalization (mean=0.5, std=0.5)
        img_array = (img_array - 0.5) / 0.5

        return img_array, img
    except Exception as e:
        print(f"Error preprocessing image: {{e}}")
        return None, None

def run_inference(model, model_type, input_tensor):
    """Run inference based on model type"""
    # PyTorch/TorchScript models expect NCHW format
    input_tensor_torch = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor_torch)
        probabilities = F.softmax(output, dim=1)

    return probabilities

def print_results(probabilities, emotions, top_n=2):
    """Print prediction results in a nice format"""
    # Sort emotions by probability
    if isinstance(probabilities, torch.Tensor):
        probabilities_np = probabilities.numpy()[0]
    else:
        probabilities_np = probabilities[0]

    sorted_indices = np.argsort(probabilities_np)[::-1]

    print("\\n" + "="*50)
    print("EMOTION PREDICTION RESULTS")
    print("="*50)

    print(f"Top {{top_n}} predictions:")
    print("-" * 40)

    for i, idx in enumerate(sorted_indices[:top_n]):
        emotion = emotions[idx] if idx < len(emotions) else f'class_{{idx}}'
        prob = probabilities_np[idx]
        bar = "█" * int(prob * 30)
        print(f"{{i+1}}. {{emotion:12}} {{prob:6.2%}} {{bar}}")

    print("\\nAll emotions:")
    print("-" * 40)
    for idx in sorted_indices:
        emotion = emotions[idx] if idx < len(emotions) else f'class_{{idx}}'
        prob = probabilities_np[idx]
        print(f"  {{emotion:12}} {{prob:6.2%}}")

def main():
    # Load model info
    try:
        with open('exports/model_info.json', 'r') as f:
            model_info = json.load(f)
        EMOTIONS = model_info['emotions']
    except:
        print("Error: Could not load model info")
        return

    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try default test images
        test_images = ['test_face.jpg', 'face.jpg', 'test.jpg', 'happy_sample.jpg', 'sad_sample.jpg']
        for img in test_images:
            if os.path.exists(img):
                image_path = img
                break
        else:
            print("Error: No image specified and no test image found")
            print("Usage: python test_image.py <image_path>")
            print("Or create a test image named 'test_face.jpg'")
            return

    print(f"Testing image: {{image_path}}")

    # Preprocess image
    input_array, original_img = preprocess_image(image_path)
    if input_array is None:
        print(f"Could not process image: {{image_path}}")
        return

    # Load model
    model, model_type = load_model()
    if model is None:
        return

    # Run inference
    probabilities = run_inference(model, model_type, input_array)

    if probabilities is None:
        print("Inference failed")
        return

    # Get prediction
    if isinstance(probabilities, torch.Tensor):
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_idx].item()
    else:
        predicted_idx = np.argmax(probabilities[0])
        confidence = probabilities[0][predicted_idx]

    # Display results
    predicted_emotion = EMOTIONS[predicted_idx] if predicted_idx < len(EMOTIONS) else f'class_{{predicted_idx}}'

    print(f"\\nPredicted emotion: {{predicted_emotion.upper()}} ({{(confidence*100):.1f}}% confidence)")
    print(f"Model type: {{model_type}}")
    print_results(probabilities, EMOTIONS, top_n=len(EMOTIONS))

    # Show image info
    if original_img:
        print(f"\\nImage info: {{original_img.size[0]}}x{{original_img.size[1]}}, mode: {{original_img.mode}}")

    print("\\n" + "="*50)
    print("To test another image: python test_image.py <image_path>")

if __name__ == "__main__":
    main()
'''

    with open('test_image.py', 'w') as f:
        f.write(image_test_script)

    print("Test scripts created: test_model.py, test_image.py")
    print("\nTo test your model:")
    print("1. Run: python test_model.py")
    print("2. Test with an image: python test_image.py <your_image.jpg>")
    print("3. Or use the test image: python test_image.py")

# storing for plot    
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def train_and_export_model():
    """Train the model and export it in multiple formats"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check and prepare data
    dataframe = check_and_prepare_data()
    if dataframe is None:
        print("\nData preparation failed. Cannot proceed with training.")
        return

    # Create exports directory
    os.makedirs("exports", exist_ok=True)

    # Prepare data
    train_data = dataframe[dataframe['Usage'] == 'Training']
    val_data = dataframe[dataframe['Usage'] == 'PublicTest']

    if len(val_data) == 0:
        # If no PublicTest, use some of Training for validation
        print("No PublicTest data found, splitting training data...")
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # MODIFIED: Calculate number of classes from data
    num_classes = len(dataframe['emotion'].unique())
    print(f"Number of emotion classes: {num_classes}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # MODIFIED: Use EmotionDataset instead of FER2013Dataset
    train_dataset = EmotionDataset(train_data, transform=train_transform)
    val_dataset = EmotionDataset(val_data, transform=val_transform)

    # Batch size - adjust based on dataset size
    batch_size = min(64, len(train_dataset))
    if batch_size < 8:
        batch_size = 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize model with correct number of classes
    model = EmotionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    best_val_accuracy = 0.0

    # early stopping
    patience = 5
    counter = 0
    

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    # MODIFIED: Adjust epochs based on dataset size
    num_epochs = 50
    if len(train_dataset) < 500:
        num_epochs = 100
    elif len(train_dataset) < 1000:
        num_epochs = 40

    start_time = time.time()
    best_epoch = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            if batch_idx % 50 == 0 and batch_idx > 0 and len(train_loader) > 5:
                batch_accuracy = 100 * (predicted == target).sum().item() / target.size(0)
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}, Accuracy = {batch_accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
 
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total


    
        # storing for visualization
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        
        # Update learning rate
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start_time

        print(f'\nEpoch {epoch+1}/{num_epochs} (Time: {epoch_time:.1f}s):')
        print(f'  Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'  Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            counter = 0  

            torch.save(model.state_dict(), 'emotion_model.pth')
            torch.save(model.state_dict(), f'exports/emotion_model_epoch_{epoch+1}.pth')
            print(f' Saved best model (epoch {epoch+1})')
        else:
            counter += 1
            print(f" Early stopping counter: {counter}/{patience}")

            if counter >= patience:
                print(f"\n Early stopping triggered at epoch {epoch+1}")
                break

    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time:.1f} seconds!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}")

    # Load best model
    if os.path.exists('emotion_model.pth'):
        model.load_state_dict(torch.load('emotion_model.pth'))
        print(f"Loaded best model from emotion_model.pth (epoch {best_epoch})")

    # val and train loss and accuracies
    print("\nGenerating training plots...")

    plt.figure(figsize=(12, 5))

    # LOSS GRAPH
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")              
    plt.ylabel("Loss")               
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    # ACCURACY GRAPH
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")              
    plt.ylabel("Accuracy (%)")       
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
    plt.close()


        
    model.eval()

    # CONFUSION MATRIX
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in val_loader:   # using validation set
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Test inference
    print("\n" + "="*70)
    print("TESTING MODEL INFERENCE")
    print("="*70)

    
    test_input = torch.randn(1, 1, 48, 48).to(device)
    with torch.no_grad():
        output = model(test_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()


    print(f"Test inference successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of classes: {num_classes}")

    emotion_name = EMOTIONS[predicted_class] if predicted_class < len(EMOTIONS) else f'class_{predicted_class}'
    print(f"Predicted emotion: {emotion_name} (class {predicted_class})")

    print(f"Probabilities:")
    for i, prob in enumerate(probabilities[0].cpu().numpy()):
        emotion_name = EMOTIONS[i] if i < len(EMOTIONS) else f'class_{i}'
        print(f"  {emotion_name}: {prob:.2%}")

    # Export model in multiple formats
    print("\n" + "="*70)
    print("EXPORTING MODEL")
    print("="*70)

    # Update training info before export
    export_key_formats(model, device, num_classes)


    print("\n" + "="*70)
    print("TRAINING AND EXPORT COMPLETE!")
    print("="*70)
    

    # Final summary
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Test your model: python test_model.py")
    print("2. Test with images: python test_image.py <your_image.jpg>")
    print("3. Check the exports directory for your models")
    print("4. Use the models in your applications")

    
    # List all created files
    print("\n" + "="*70)
    print("CREATED FILES")
    print("="*70)
    print("In current directory:")
    for file in os.listdir('.'):
        if file.endswith(('.py', '.pth', '.csv')):
            print(f"  - {file}")

    print("\nIn exports directory:")
    if os.path.exists('exports'):
        for file in os.listdir('exports'):
            print(f"  - {file}")

def main():
    """Main function with error handling"""
    print("\n" + "="*70)
    print("EMOTION RECOGNITION MODEL TRAINING AND EXPORT (2 Emotions)")
    print("="*70)

    print("\nSupported export formats:")
    print("1. PyTorch (.pth)")
    print("2. TorchScript (.pt)")
    print("3. Complete PyTorch model (.pth)")

    try:
        train_and_export_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        print(traceback.format_exc())
        print("\nTroubleshooting tips:")
        print("1. Make sure you have emotion_dataset.csv file")
        print("2. Check if PyTorch is properly installed")
        print("3. Try reducing batch size if you have memory issues")
        print("4. Check available disk space")
              
if __name__ == "__main__":
    main()
