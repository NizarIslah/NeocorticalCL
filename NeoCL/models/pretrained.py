import torch
from torch import nn
from avalanche.models import IncrementalClassifier
from avalanche.training import ICaRL
from torch.nn import Linear, ReLU, MultiheadAttention, Sequential
import clip
from torchvision import datasets
from collections import defaultdict, deque
import itertools
    
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


class ImageNetWiki(datasets.CIFAR100):
    def __init__(self, root, class_text, transforms=None, text_transforms=None, context_len=77, train=True):
        split = 'train' if train else 'val'
        super().__init__(root, train)
        self.transforms = transforms
        self.n_classes = 100
        self.class_text = class_text
        self.context_len = context_len
        self.text_transforms = text_transforms

    def __getitem__(self, index):
        im, label = super().__getitem__(index)
        text = self.class_text[label]['articles'][0][:self.context_len]
        text = clip.tokenize(text).squeeze()
        if self.text_transforms:
            text = self.text_transforms(text)
        if self.transforms:
            im = self.transforms(im)
        text = tex.repeat(text.unsqueeze()[:,None,None])
        print(im.shape,text.shape)
        im_text=torch.cat([im,text,0])
        return im, label
    

class CLIP_Attention(nn.Module):
    def __init__(self, pretrained_net, preprocess, num_classes, num_heads, embed_dim=2048, text_len=77):
        super(CLIP_Attention, self).__init__()
        self.preprocess = preprocess
        self.text_len=text_len
        self.feature_extractor = pretrained_net
        self.classifier = Sequential(
            MultiheadAttention(embed_dim, num_heads),
            ReLU(),
            Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        image, text = x[:, :-self.text_len], x[:, -self.text_len:]
        print(image.shape, text.shape)
        image = self.preprocess(image)
        with torch.no_grad():
            img_features = self.feature_extractor.encode_image(image)
            text_features = self.feature_extractor.encode_text(token)
        layers = [module for module in self.classifier.modules() if not isinstance(module, nn.Sequential)]
        out = layers[0](query=img_features, key=text_features, value=img_features)
        for l, module in enumerate(layers[1:]):
            out = module(out)
        return out

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