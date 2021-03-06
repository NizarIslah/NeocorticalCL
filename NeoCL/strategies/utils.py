from types import SimpleNamespace
import os
from pathlib import Path
import inspect
from pandas import read_csv

def pandas_to_list(input_str):
    return [float(el) for el in input_str.strip('[] ').split(' ')]


def get_average_metric(metric_dict: dict, metric_name: str = 'Top1_Acc_Stream'):
    """
    Compute the average of a metric based on the provided metric name.
    The average is computed across the instance of the metrics containing the
    given metric name in the input dictionary.
    :param metric_dict: dictionary containing metric name as keys and metric value as value.
        This dictionary is usually returned by the `eval` method of Avalanche strategies.
    :param metric_name: the metric name (or a part of it), to be used as pattern to filter the dictionary
    :return: a number representing the average of all the metric containing `metric_name` in their name
    """

    avg_stream_acc = []
    for k, v in metric_dict.items():
        if k.startswith(metric_name):
            avg_stream_acc.append(v)
    return sum(avg_stream_acc) / float(len(avg_stream_acc))


def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    args.__dict__['check'] = True
    if additional_args is not None:
        for k, v in additional_args.items():
            args.__dict__[k] = v
    return args


__all__ = ['get_average_metric', 'create_default_args']
