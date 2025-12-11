POSITIVE = 1
NEGATIVE = 0

def score_to_label(scores:list, threshold:float):
    return [NEGATIVE if score < threshold else POSITIVE for score in scores]

def confusion_matrix(y_true:list, y_pred:list):
    assert(len(y_true) == len(y_pred))
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]: # correct classification
            if y_pred[i] == POSITIVE: # True Positive
                TP += 1
            if y_pred[i] == NEGATIVE: # True Negative
                TN += 1
        if y_true[i] != y_pred[i]: # wrong classification
            if y_pred[i] == POSITIVE: # False Positive
                FP += 1
            if y_pred[i] == NEGATIVE: # False Negative
                FN += 1
    return TP, TN, FP, FN

# ROC, Receiver Operating Characteristic Curve
def roc(y_true:list, scores:list)->list:
    assert(len(y_true) == len(scores))
    thresholds = list(set(scores))
    thresholds.sort(reverse=False) # ascend order
    min_threshold, max_threshold = thresholds[0] - 0.1, thresholds[-1] + 0.1
    thresholds.insert(0, min_threshold)
    thresholds.append(max_threshold)
    c = {}
    for threshold in thresholds:
        print(f'threshold: {threshold}')
        y_pred = score_to_label(scores, threshold)
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        print(f'TP={TP}\tFN={FN}\nFP={FP}\tTN={TN}')
        FPR = FP / (FP + TN) # FP rate
        TPR = TP / (TP + FN) # TP rate
        c[FPR] = max(c[FPR], TPR) if FPR in c else TPR 
    x = list(set(c))
    x.sort()
    curve = [(xx, c[xx]) for xx in x]
    return curve

def auc(roc_curve:list):
    a = roc_curve[0]
    areas = []
    for i in range(1, len(roc_curve)):
        b = roc_curve[i]
        area = (b[0]-a[0]) * abs(b[1]-a[1]) / 2
        areas.append(area)
    return sum(areas)

if __name__ == '__main__':
    y_true = [POSITIVE, NEGATIVE, NEGATIVE, POSITIVE, POSITIVE, POSITIVE, NEGATIVE, NEGATIVE]
    scores = [0.77, 0.43, 0.58, 0.82, 0.9, 0.49, 0.44, 0.17] # each in range (0.0, 1.0)
    threshold = 0.5
    y_pred = score_to_label(scores, threshold)

    print(f'y_true: {y_true}')
    print(f'y_pred: {y_pred}')

    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)

    print(f'TP={TP}\tFN={FN}\nFP={FP}\tTN={TN}')
    FPR = FP / (FP + TN) # FP rate
    TPR = TP / (TP + FN) # TP rate

    print(f'FPR: {FPR}\tTPR: {TPR}')
    print(f'TPR: {TPR}')
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TPR
    Specificity = TN / (TN + FP)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    print(f'Accuracy: {Accuracy}')
    print(f'Precision: {Precision}')
    print(f'Recall(=TPR): {Recall}')
    print(f'Specificity: {Specificity}')
    print(f'F1_score: {F1_score}')

    TPR = TP / (FP + FN)

    roc_curve = roc(y_true, scores)
    print(f'roc curve: {roc_curve}')

    auc_value = auc(roc_curve)
    print(f'auc: {auc_value}')

    # AUC, Area Under the Curve
