import pandas as pd
import zipfile
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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


def run_model(model, X, Y, x_test, y_test):
    inicio_trieino = time.time()
    model.fit(X, Y)
    fim_trieino = time.time()
    y_pred = model.predict(x_test)
    fim_teste = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    treino = fim_trieino - inicio_trieino
    teste = fim_teste - fim_trieino

    return (accuracy, f1, precision, recall, treino, teste)


def main():
    X, Y = load_data()
    # Normalização dos dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Separando dados para teste e treino
    test_size = 0.2
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size)

    models = {
        "KNN": KNeighborsClassifier(),
        "Árvore": DecisionTreeClassifier(),
        "MLP": MLPClassifier(max_iter=100),
        "SVM": SVC(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(var_smoothing=0.5),
    }

    diretorio = "output"
    arquivo_resultado = "resultado.csv"
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

    with open(os.path.join(diretorio, arquivo_resultado), "w") as saida:
        saida.write("Modelo;Accuracy;F1 Score;Precision;Recall;Treino;Teste\n")
        for name, model in models.items():
            accuracy, f1, precision, recall, treino, teste = run_model(
                model, X, Y, X_test, Y_test
            )
            print(f"Modelo:    {name}")
            print(f"Accuracy:  {accuracy:.2f}")
            print(f"F1 Score:  {f1:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall:    {recall:.2f}")
            print(f"Treino:    {treino:.2f}")
            print(f"Teste:     {teste:.2f}\n")
            saida.write(
                f"{name};{accuracy:.2f};{f1:.2f};{precision:.2f};{recall:.2f};{treino:.2f};{teste:.2f}\n"
            )

        modelo_escolhido = RandomForestClassifier(
            n_estimators=10, max_depth=10, max_features="log2", n_jobs=-1
        )
        accuracy, f1, precision, recall, treino, teste = run_model(
            modelo_escolhido, X, Y, X_test, Y_test
        )
        print("======================================")
        print(f"Modelo:    {modelo_escolhido}")
        print(f"Accuracy:  {accuracy:.2f}")
        print(f"F1 Score:  {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"Treino:    {treino:.2f}")
        print(f"Teste:     {teste:.2f}")
        saida.write(
            f"*RandomForest;{accuracy:.2f};{f1:.2f};{precision:.2f};{recall:.2f};{treino:.2f};{teste:.2f}\n"
        )


if __name__ == "__main__":
    main()
