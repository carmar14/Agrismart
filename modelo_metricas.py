import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import joblib

# Cargar los datos de validación
df = pd.read_csv('frijol.csv')

# Definir las variables y la variable objetivo
X = df[['temperatura', 'humedad_suelo', 'humedad_ambiente', 'luz', 'ph_suelo', 'co2', 'nitrogeno', 'fosforo', 'potasio']]

y_classifier = df['vive']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test,y_train_classifier, y_test_classifier = train_test_split(
    X, y_classifier, test_size=0.3, random_state=42)

# Separar características y etiquetas
#X_test = df.drop(columns=['label'])  # Suponiendo que la columna de etiquetas se llama 'label'
#y_test = df['label']

# Entrenar el modelo RandomForestClassifier para clasificación multiclase
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train_classifier)

# Evaluar los modelos
y_pred_classifier = classifier.predict(X_test)

# Predecir las clases y las probabilidades
y_pred = classifier.predict(X_test)
y_score = classifier.predict_proba(X_test)

# Generar el reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test_classifier, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test_classifier, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_classifier), yticklabels=np.unique(y_test_classifier))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Convertir etiquetas en formato binario para ROC y Precision-Recall
classes = np.unique(y_test_classifier)
y_test_bin = label_binarize(y_test_classifier, classes=classes)

# Graficar las curvas ROC y Precision-Recall para cada clase
plt.figure(figsize=(12, 5))

# Curva ROC
plt.subplot(1, 2, 1)
for i, class_label in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Clase {class_label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC')
plt.legend(loc='lower right')

# Curva Precision-Recall
plt.subplot(1, 2, 2)
for i, class_label in enumerate(classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=f'Clase {class_label}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curvas Precision-Recall')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()
