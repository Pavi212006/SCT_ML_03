import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#SETUP
np.random.seed(88) 
samples = 400
pixel_data = np.random.rand(samples, 20) 
# Labels: 0 for CAT, 1 for DOG
labels = np.array([0]*200 + [1]*200) 

#DATA PARTITIONING
X_train, X_test, y_train, y_test = train_test_split(pixel_data, labels, test_size=0.2, random_state=42)

#SVM ENGINE
ImagePro_SVM = svm.SVC(kernel='linear', C=1.0)
ImagePro_SVM.fit(X_train, y_train)

#USER-FRIENDLY INTERFACE
print("\n" + "=" * 45)
print("IMAGE RECOGNITION SYSTEM")
print("=" * 45)

def run_identification():
    test_img = np.random.rand(1, 20)
    prediction = ImagePro_SVM.predict(test_img)
    result = "IDENTIFIED: DOG" if prediction == 1 else "IDENTIFIED: CAT"
    print(f"RESULT: {result}")

run_identification()

#PERFORMANCE VISUALIZATION
final_acc = accuracy_score(y_test, ImagePro_SVM.predict(X_test))
plt.figure(figsize=(7, 5))
plt.bar(['SVM Prediction Accuracy'], [final_acc], color='indigo')
plt.title(f'SVM Image Classifier Accuracy: {final_acc*100:.1f}%')
plt.ylabel('Confidence Level (0-1)')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
