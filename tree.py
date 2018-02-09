import operator
from audioop import reverse
from math import log


class Tree:
    leaf = False
    prediction = None
    feature = None
    threshold = None
    left = None
    right = None


def predict(tree, point):
    if tree.leaf:
        return tree.prediction
    i = tree.feature
    # print("tree threshold : " , tree.threshold)
    # print("point.values[i] : ", point.values[i])
    # if (point.values[i] < int(0 if tree.threshold is None else tree.threshold)):
    if (point.values[i] < (tree.threshold)):
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)


def most_likely_class(prediction):
    labels = list(prediction.keys())
    probs = list(prediction.values())
    return labels[probs.index(max(probs))]


def accuracy(data, predictions):
    total = 0
    correct = 0
    for i in range(len(data)):
        point = data[i]
        pred = predictions[i]
        total += 1
        guess = most_likely_class(pred)
        if guess == point.label:
            correct += 1
    return float(correct) / total

# Done
def split_data(data, feature, threshold):
    left = []
    right = []
    # TODO: split data into left and right by given feature.
    # left should contain points whose values are less than threshold
    # right should contain points with values greater than or equal to threshold
    for d in data:
        if d.values[feature]<threshold:
            left.append(d)
        else:
            right.append(d)
    return (left, right)

# Done
def count_labels(data):
    counts = {}
    # TODO: counts should count the labels in data
    for d in data:
        if not d.label in counts:
            counts[d.label] = 1
        else:
            counts[d.label] +=1
    # e.g. counts = {'spam': 10, 'ham': 4}
    return counts

# Done
def counts_to_entropy(counts):
    entropy = 0.0
    # TODO: should convert a dictionary of counts into entropy
    labelcount = 0
    total = 0
    for k,v in counts.items():
        labelcount += 1
        total += v
    for k, v in counts.items():
        try:
            entropy += (-(v / total)) * (log((v / total), 2))
        except:
            entropy += 0

    return entropy


def get_entropy(data):
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy

# This is an inefficient way to find the best threshold to maximize
# information gain.
def find_best_threshold(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    # TODO: Write a method to find the best threshold.
    for d in data:
        left,right = split_data(data, feature, d.values[feature])
        leftentropy = get_entropy(left)
        rightentropy = get_entropy(right)
        gain = entropy - ((len(left)/len(data))*leftentropy) - ((len(right)/len(data))*rightentropy)
        if(gain>best_gain):
            best_gain = gain
            best_threshold = d.values[feature]
    return (best_gain, best_threshold)


def find_best_threshold_fast(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None

    right_list = []
    left_list = []
    right_counts_dict = {}
    left_counts_dict = {}
    for d in data:
        right_sub_list = []
        right_sub_list.append(d.label)
        right_sub_list.append(d.values[feature])
        right_list.append(right_sub_list)
    # print("right counts : " , right_list)
    right_list = sorted(right_list, key=operator.itemgetter(1), reverse=True)
    # print("sorted right counts : ", right_list)

    # # TODO: counts should count the labels in data
    for d in data:
        if not d.label in right_counts_dict:
            right_counts_dict[d.label] = 1
        else:
            right_counts_dict[d.label] +=1

    for k,v in right_counts_dict.items():
        left_counts_dict[k] = 0

    # print("left ",left_counts_dict)
    # print("right", right_counts_dict)
    #
    left_dict_len = 0
    right_dict_len = len(data)
    for d in reversed(right_list):
        left_entropy = counts_to_entropy(left_counts_dict)
        right_entropy = counts_to_entropy(right_counts_dict)
        gain = entropy - (((left_dict_len) / len(data)) * left_entropy) - (((right_dict_len) / len(data)) * right_entropy)
        if(gain>best_gain):
            best_gain = gain
            best_threshold = d[1]
        popped = right_list.pop()
        # print("popped : ", popped)
        left_counts_dict[popped[0]] += 1
        right_counts_dict[popped[0]] -=1
        left_dict_len += 1
        right_dict_len -=1


    # print("best gain fast : ", best_gain)
    # print("best threshold fast : ", best_threshold)
    # TODO: Write a more efficient method to find the best threshold.
    return (best_gain, best_threshold)


def find_best_split(data):
    if len(data) < 2:
        return None, None
    best_feature = None
    best_threshold = None
    best_gain = 0
    # TODO: find the feature and threshold that maximize information gain.

    #checking with all the features for finding the best feature and threshold
    for f in range(0,len(data[0].values)):
        temp_best_gain, temp_best_threshold = find_best_threshold_fast(data,f)
        if(temp_best_gain>best_gain):
            best_gain = temp_best_gain
            best_threshold = temp_best_threshold
            best_feature = f
    # print("best feature : ", best_feature)
    # print("best threshold : ", best_threshold)

    return (best_feature, best_threshold)


def make_leaf(data):
    tree = Tree()
    tree.leaf = True
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label]) / len(data)
    tree.prediction = prediction
    return tree


def c45(data, max_levels):
    tree = Tree()
    counts = count_labels(data)
    # print("counts : ", counts)

    prediction = {}
    if max_levels <= 0:
        return make_leaf(data)
    # print("list(counts.values())[0] : ", list(counts.values())[0])
    # print("list(counts.values())[1] : ", list(counts.values())[1])
    if(len(counts)<2 or list(counts.values())[0] == list(counts.values())[1]):
        return make_leaf(data)


    # TODO: Construct a decision tree with the data and return it.
    # Your algorithm should return a leaf if the maximum level depth is reached
    # or if there is no split that gains information, otherwise it should greedily
    # choose an feature and threshold to split on and recurse on both partitions
    # of the data.

    #TODO: if there is no split that gains information return leaf
    feature, threshold = find_best_split(data)
    if (feature==None and threshold==None):
        return make_leaf(data)

    #TODO: Recursively do partition and call the method
    left_data, right_data = split_data(data, feature, threshold)
    tree.feature = feature
    tree.threshold = threshold
    tree.left = c45(left_data, max_levels-1)
    tree.right = c45(right_data, max_levels - 1)
    print("left_data len : ", len(left_data))
    print("right_data len : ", len(right_data))
    for label in counts:
            prediction[label] = float(counts[label]) / len(data)

    tree.prediction = prediction
    # tree.leaf = False

    return tree


def submission(train, test):
    # TODO: Once your tests pass, make your submission as good as you can!
    tree = c45(train, 7)
    predictions = []
    for point in test:
        predictions.append(predict(tree, point))
    return predictions

# This might be useful for debugging.


def print_tree(tree):
    if tree.leaf:
        print("Leaf", tree.prediction)
    else:
        print("Branch", tree.feature, tree.threshold)
        print_tree(tree.left)
        print_tree(tree.right)
