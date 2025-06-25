import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load features and labels
X = np.load("clip_features.npy")
y = np.load("clip_labels.npy")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize SVM (RBF kernel)
svm = SVC(kernel='rbf', C=1.0, probability=True)  # Set probability=True if you want confidence scores
svm.fit(X_train, y_train)

# Evaluate on test set
y_pred = svm.predict(X_test)
print("\n🧾 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["front", "back", "side"]))

print("\n🔍 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(svm, "svm_phone_view_model.joblib")
print("\n✅ SVM model saved to 'svm_phone_view_model.joblib'")
