import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix

# Define the emotion mapping
emotion_mapping = {
    "happy": 1,
    "sad": 2,
    "neutral": 3,
    "angry": 4,
    "surprise": 5,
    "disgust": 6,
    "fear": 7
}

# Read the CSV file
csv_path = "/home/qixuan/Documents/R1-Omni/DFEW_all_instruction/set_1_train_test.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# Map predicted and true labels to numeric values
df['label_numeric'] = df['label']
df['predicted_label_numeric'] = df['predicted_label'].str.lower().map(emotion_mapping)
print(df['label'])
# Drop rows where mapping failed (if any)
df = df.dropna(subset=['label_numeric', 'predicted_label_numeric'])

# Extract true and predicted labels
true_labels = df['label_numeric'].astype(int)
predicted_labels = df['predicted_label_numeric'].astype(int)
print(29, true_labels)
# Calculate Unweighted Average Recall (UAR)
uar = recall_score(true_labels, predicted_labels, average='macro')
print(f"Unweighted Average Recall (UAR): {uar:.4f}")

# Calculate Weighted Average Recall (WAR)
war = recall_score(true_labels, predicted_labels, average='weighted')
print(f"Weighted Average Recall (WAR): {war:.4f}")

# Optional: Print confusion matrix for detailed analysis
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)