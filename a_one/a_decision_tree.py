from sklearn import tree
import csv
import os

# ----------------------------------------------------------------

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = ['apple', 'apple', 'orange', 'orange']

# Decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]]))

# -------------------------------------------------------------------

print(os.getcwd())

features = []
labels = []
reader = None
rows = 0
with open('df.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        rows+=1
        feature = [len(row['Million Plus Cities']),
                   len(row['Cause category']),
                   len(row['Cause Subcategory']),
                   len(row['Outcome of Incident'])
                   ]
        label = row['Count']

        features.append(feature)
        labels.append(label)


print('rows in dataset', rows)
x = features.pop(1013)
y = labels.pop(1013)
print('test',x,y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features[:-2], labels[:-2])
print('prediction ',clf.predict([x]))
