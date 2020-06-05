import numpy as np


def metrics_at_k(ranked_indices, k, img_to_positive_catalog, ignore_first_rec=True):
    # Get the top k recommendations for every query image
    if ignore_first_rec:
        top_k = ranked_indices[:, 1:k + 1]
    else:
        top_k = ranked_indices[:, :k]

    # Initialize metric vectors
    adj_recall_at_k = np.full(top_k.shape[0], -999999)
    recall_at_k = np.full(top_k.shape[0], -999999)
    catch_at_k = np.full(top_k.shape[0], -999999)

    for idx in range(top_k.shape[0]):
        isin_recommendations = np.isin(img_to_positive_catalog[idx], top_k[idx])
        # Calculate metrics for a single example
        adj_recall = isin_recommendations.sum() / min(len(isin_recommendations), k)
        recall = isin_recommendations.mean()
        catch = isin_recommendations.max()
        # Cache the metrics
        adj_recall_at_k[idx] = adj_recall
        recall_at_k[idx] = recall
        catch_at_k[idx] = catch

    assert adj_recall_at_k.min() >= 0
    assert recall_at_k.min() >= 0
    assert catch_at_k.min() >= 0

    # Calculate and return average metric over test set
    return adj_recall_at_k.mean(), recall_at_k.mean(), catch_at_k.mean()