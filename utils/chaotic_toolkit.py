import math
import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils.logger import info

"""
Code reference https://www.kaggle.com/tigurius/introduction-to-taken-s-embedding
"""

def takens_embedding(series: np.ndarray, delay, dimension) -> np.ndarray:
    """
    This function returns the Takens embedding of data with delay into dimension,
    delay*dimension must be < len(data)
    """
    if delay * dimension > len(series):
        info(f'Not enough data for the given delay ({delay}) and dimension ({dimension}).'
             f'\ndelay * dimension > len(data): {delay * dimension} > {len(series)}')
        return series
    delay_embedding = np.array([series[0:len(series) - delay * dimension]])
    for i in range(1, dimension):
        delay_embedding = np.append(delay_embedding,
                                    [series[i * delay:len(series) - delay * (dimension - i)]], axis=0)
    return delay_embedding


def compute_mutual_information(series, delay, num_bins):
    """This function calculates the mutual information given the delay.
    First one calculates the minimium $x_{min}$  and maximum $x_{max}$ of the time-series. Then the interval $[x_{min},
    x_{max}]$ is divided into a large number of bins. Denote by $P_k$ the probability that an element of the time-series
     is in the $k$th bin and by $P_{h,k}(\tau)$ the probability that $x_i$ is in the $h$th bin while $x_{i+\tau}$ is in
     the $k$th bin. Then the mutual information is
     $$ I(\tau) = - \sum_{h=1}^{nBins} \sum_{k=1}^{nBins} P_{h,k}(\tau) \log \frac{P_{h,k}(\tau)}{P_h P_k}.$$
     The first minimum of $I(\tau)$ as a function of $\tau$ gives the optimal delay, since there we get largest
     information by adding $x_{i+\tau}$. All probabilities here are calculated as empirical probabilities. """
    mutual_information = 0
    max_val = max(series)
    min_val = min(series)
    delayed_series = series[delay:len(series)]
    shortened_series = series[0:len(series) - delay]
    bin_size = abs(max_val - min_val) / num_bins
    prob_in_bin_dict = {}
    condition_to_be_in_bin = {}
    condition_delay_to_be_in_bin = {}

    for i in range(0, num_bins):
        memoize_prob(i, bin_size, condition_to_be_in_bin, min_val, prob_in_bin_dict, shortened_series)

        for j in range(0, num_bins):
            memoize_prob(j, bin_size, condition_to_be_in_bin, min_val, prob_in_bin_dict, shortened_series)

            if j not in condition_delay_to_be_in_bin:
                cond = compute_condition(j, bin_size, min_val, delayed_series)
                condition_delay_to_be_in_bin.update({j: cond})

            p_ij = calculate_joint_prob(condition_delay_to_be_in_bin, condition_to_be_in_bin, i, j, shortened_series)
            if p_ij != 0 and prob_in_bin_dict[i] != 0 and prob_in_bin_dict[j] != 0:
                mutual_information -= p_ij * math.log(p_ij / (prob_in_bin_dict[i] * prob_in_bin_dict[j]))

    return mutual_information


def calculate_joint_prob(condition_delay_to_be_in_bin, condition_to_be_in_bin, i, j, shortened_series):
    return len(shortened_series[condition_to_be_in_bin[i] & condition_delay_to_be_in_bin[j]]) / len(shortened_series)


def memoize_prob(bin_index, bin_size, condition_to_be_in_bin, min_val, prob_in_bin_dict, shortened_series):
    if bin_index not in prob_in_bin_dict:
        compute_and_update_probability(bin_index, bin_size, condition_to_be_in_bin,
                                       min_val, prob_in_bin_dict, shortened_series)


def compute_and_update_probability(bin_index, bin_size, condition_to_be_in_bin, min_val, prob_in_bin_dict,
                                   shortened_series):
    cond = compute_condition(bin_index, bin_size, min_val, shortened_series)
    condition_to_be_in_bin.update({bin_index: cond})
    num_of_vals_in_bin = calculate_num_of_elements_in_bin(bin_index, condition_to_be_in_bin, shortened_series)
    prob_in_bin_dict.update({bin_index: num_of_vals_in_bin / len(shortened_series)})


def calculate_num_of_elements_in_bin(bin_index, condition_to_be_in_bin, shortened_series):
    return len(shortened_series[condition_to_be_in_bin[bin_index]])


def compute_condition(bin_index, bin_size, min_val, series):
    return (series >= (min_val + bin_index * bin_size)) & (series < (min_val + (bin_index + 1) * bin_size))


def calculate_false_nearest_neighours(data, delay, dimension) -> int:
    "Calculates the number of false nearest neighbours of embedding dimension"
    embedded_data = takens_embedding(data, delay, dimension)
    # the first nearest neighbour is the data point itself, so we choose the second one
    nearest_neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto', n_jobs=-1).fit(embedded_data.transpose())
    distances, indices = nearest_neighbors.kneighbors(embedded_data.transpose())
    # two data points are nearest neighbours if their distance is smaller than the standard deviation
    epsilon = np.std(distances.flatten())
    false_neighbors = 0
    for i in range(0, len(data) - delay * (dimension + 1)):
        if (0 < distances[i, 1]) and (distances[i, 1] < epsilon) and ((abs(
                data[i + dimension * delay] - data[indices[i, 1] + dimension * delay]) / distances[
                                                                           i, 1]) > 10):
            false_neighbors += 1
    return false_neighbors
