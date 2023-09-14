def main ():
    from sklearn.datasets import load_iris
    from sklearn import tree
    from numpy import delete

    iris = load_iris()
    print(iris.feature_names)
    print(iris.target_names)
    print(iris.data[0])
    print(iris.target[0])

    test_idx =  [0, 50, 100]

    train_target = delete(iris.target, test_idx)
    train_data = delete(iris.data, test_idx, axis=0)

    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data,train_target)
    print(test_target)
    print(clf.predict(test_data))

if '__main__' == __name__:
    main()

