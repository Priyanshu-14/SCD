from sklearn.metrics import f1_score

def f1_score_scene(preds, labels):
    """
    preds: list/np.array of predicted scene cuts (0/1)
    labels: list/np.array of true scene cuts (0/1)
    """
    return f1_score(labels, preds, average="binary")
