import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

confusion_mat = confusion_matrix(true_labels, pred_labels)
print("Confusion matrix:\n", confusion_mat)

plt.figure(figsize=(5.5, 4.5))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
for i in range(5):
    for j in range(5):
        v = confusion_mat[i, j]
        plt.text(j, i, v, ha='center', va='center',
                 color='black' if v >= confusion_mat.max() / 2 else 'white')
plt.tight_layout()
plt.savefig('output/confusion_matrix.png', dpi=150)
plt.close()

targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(true_labels, pred_labels, target_names=targets))
