from sklearn import cross_validation


def split_data(X, y, t_size):
    if len(X) != len(y):
        return None

    if hasattr(cross_validation, 'train_test_split'):
        return cross_validation.train_test_split(X, y, test_size=t_size)

    return None
