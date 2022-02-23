import torch
from torch import nn
from avalanche.models import IncrementalClassifier
from torch.nn import Linear, ReLU, MultiheadAttention
import clip

class SSLIcarl(nn.Module):
    def __init__(self, pretrained_net, embedding_size, num_classes):
        super(SSLIcarl, self).__init__()
        self.feature_extractor = pretrained_net
        # self.feature_extractor.freeze()
        self.classifier = Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)  # Already flattened
        x = self.classifier(x)
        return x
    
class CLIP_Attention(nn.Module):
    def __init__(self, pretrained_net, preprocess, num_classes, num_heads, embed_dim=2048):
        super(CLIP_Attention, self).__init__()
        self.preprocess = preprocess
        self.model = pretrained_net
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.classifier = Linear(embed_dim, num_classes)
    
    def forward(self, image, text):
        image = self.preprocess(image)
        text = clip.tokenize(text)
        with torch.no_grad():
            img_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            print(text_features.shape)
        atn_out = self.attention(query=img_features, key=text_features, value=img_features)
        logits = self.classifier(atn_out)
        return logits

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
        self.feature_extractor = pretrained_model
        self.classifier = Linear(in_features, initial_out_features)

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
        z = self.encoder(x)
        z = self.pre_classifier(z)
        return self.classifier(z)