import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title='Classifier boundaries'):
    # Визначаємо мінімальні та максимальні значення для кожної осі
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Крок для сітки
    mesh_step_size = 0.01

    # Створюємо сітку координат
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))

    # Прогнозуємо вихід на сітці
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    # Створюємо графік
    plt.figure()

    # Наносимо на контурну карту межі класифікації
    plt.contourf(x_vals, y_vals, output, cmap=plt.cm.coolwarm, alpha=0.3)

    # Наносимо точки даних на графік
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.coolwarm)

    # Додаємо назву графіку та підписи до осей
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Показуємо графік
    plt.show()