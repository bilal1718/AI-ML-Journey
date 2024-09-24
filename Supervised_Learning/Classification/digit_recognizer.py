from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
mnist=fetch_openml('mnist_784', as_frame=False)
X,y=mnist.data, mnist.target
print(X.shape)
print(y.shape)
def plot_digit(image_data):
    image=image_data.reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit=X[0]
plot_digit(some_digit)
plt.show()

X_train, X_test, y_train, y_test=X[:60000], X[60000:], y[:60000], y[60000:]


y_train_5=(y_train=='5')
y_test_5=(y_test=='5')

