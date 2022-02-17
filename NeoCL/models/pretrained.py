import torch
from avalanche.models import IncrementalClassifier


class PretrainedIncrementalClassifier(IncrementalClassifier):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.
    """

    def __init__(self, pretrained_model, in_features, initial_out_features=2):
        """
        :param in_features: number of input features, should be = to pretrained out size
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__(in_features)
        self.encoder = pretrained_model
        self.encoder.freeze()
        self.classifier = torch.nn.Linear(in_features, initial_out_features)

    @torch.no_grad()
    def adaptation(self, dataset):
        """ If `dataset` contains unseen classes the classifier is expanded.

        :param dataset: data from the current experience.
        :return:
        """
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        new_nclasses = max(self.classifier.out_features,
                           max(dataset.targets) + 1)

        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b

    def forward(self, x, **kwargs):
        """ compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        with torch.no_grad():
            z = self.encoder(x)
        return self.classifier(z)