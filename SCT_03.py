"""Synthetic binary-classification demo using a linear Support Vector Machine.

This is not an image classifier: the input rows are generated numeric features,
not pixels or features extracted from cat/dog images.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_SEED = 88
N_PER_CLASS = 200
N_FEATURES = 20

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Create two synthetic classes with slightly different feature patterns.
    # These are illustrative numeric vectors, not real image data.
    class_a = rng.normal(loc=-0.6, scale=1.0, size=(N_PER_CLASS, N_FEATURES))
    class_b = rng.normal(loc=0.6, scale=1.0, size=(N_PER_CLASS, N_FEATURES))
    X = np.vstack([class_a, class_b])
    y = np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Scale features and train a linear SVM. Pipeline ensures scaling is fit
    # only on training data and applied consistently to the test/example data.
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0, random_state=RANDOM_SEED),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Synthetic binary classification with a linear SVM")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features per sample: {N_FEATURES}")
    print(f"Test accuracy: {accuracy:.3f}")
    print("\nClassification report (0 = Class A, 1 = Class B):")
    print(classification_report(y_test, y_pred, target_names=["Class A", "Class B"], zero_division=0))

    # Predict one new synthetic example from Class A's feature distribution.
    example = rng.normal(loc=-0.6, scale=1.0, size=(1, N_FEATURES))
    predicted_class = int(model.predict(example)[0])
    print(f"Example prediction: Class {'B' if predicted_class == 1 else 'A'}")
    print("This prediction is only a demonstration on generated data.")

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Class A", "Class B"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Linear SVM — synthetic test data")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
