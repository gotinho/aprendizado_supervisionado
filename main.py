import pandas as pd
import zipfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# (KNN, Árvore, MLP, SVM, RandomForest, AdaBoost, Naive Bayes)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# (Acurácia, F1-Score, Precisão, Revocação)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    with zipfile.ZipFile("data/dataset.zip", "r") as zip_file:
        with zip_file.open("dataset.csv") as file:
            df = pd.read_csv(file)
            # Separando atributos e rotulos
            X = df.iloc[:, 1:-1]
            Y = df.iloc[:, -1]
    return (X, Y)


def main():
    X, Y = load_data()
    # Normalização dos dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Separando dados para teste e treino
    test_size = 0.2
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size)

    # knn = KNeighborsClassifier()
    # knn.fit(X,Y)
    # y_pred = knn.predict(X_test)
    
    # accuracy = accuracy_score(Y_test, y_pred)
    # f1 = f1_score(Y_test, y_pred)
    # precision = precision_score(Y_test, y_pred)
    # recall = recall_score(Y_test, y_pred)
    
    # print(f"Accuracy:  {accuracy:.2f}")
    # print(f"F1 Score:  {f1:.2f}")
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall:    {recall:.2f}")
  
    print("NB")

    nb = GaussianNB(var_smoothing=0.5)
    nb.fit(X,Y)
    y_pred = nb.predict(X_test)
    
    accuracy = accuracy_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")


if __name__ == "__main__":
    main()
