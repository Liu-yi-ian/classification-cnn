"""Read ImageNet folders and the fixed CIFAR-10 plus ants/bees classification split."""
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
         'horse', 'ship', 'truck', 'ants', 'bees']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def letterbox(image, size=224, fill=(124, 116, 104)):
    """Fit the image inside a square without changing its aspect ratio."""
    image = image.convert('RGB')
    width, height = image.size
    scale = size / max(width, height)
    image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                         Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size, size), tuple(fill))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


class InputTransform:
    """Augment training images, then letterbox and normalize every input."""
    def __init__(self, training=False, config=None):
        config = config or {}
        self.size = config.get('size', 224)
        self.fill = config.get('fill', [124, 116, 104])
        self.mean = config.get('mean', MEAN)
        self.std = config.get('std', STD)
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(config.get('hflip', 0.5)),
            transforms.ColorJitter(*config.get('color_jitter', [0.1, 0.1, 0.1, 0])),
        ]) if training else None

    def __call__(self, image):
        image = image.convert('RGB')
        # Apply color changes before padding so the canvas color stays constant.
        if self.augment is not None:
            image = self.augment(image)
        return TF.normalize(TF.to_tensor(letterbox(image, self.size, self.fill)), self.mean, self.std)


class ClassificationData(Dataset):
    """Load fixed sample IDs; labels 0-9 are CIFAR-10, 10 is ants, and 11 is bees."""
    def __init__(self, split, root='data', manifest_path='data/splits/cifar10_insects_seed42.json', input_config=None):
        self.root = ROOT / root
        self.ids = json.loads((ROOT / manifest_path).read_text())[split]
        self.classes = NAMES
        self.transform = InputTransform(split == 'train', input_config)
        # Training and validation IDs index the CIFAR training array; test IDs
        # index the CIFAR test array. The JSON file determines the split itself.
        self.cifar = datasets.CIFAR10(str(self.root / '6d958be074577803'),
                                     train=split != 'test', download=False)
        self.labels = [self.cifar.targets[int(i.split('/')[1])] if i.startswith('cifar')
                       else NAMES.index(i.split('/')[2]) for i in self.ids]

    def __len__(self):
        return len(self.ids)

    def image(self, index):
        """Resolve an ID to either an in-memory CIFAR image or an insect image file."""
        sample_id = self.ids[index]
        if sample_id.startswith('cifar'):
            return Image.fromarray(self.cifar.data[int(sample_id.split('/')[1])])
        path = self.root / 'fbc41b31d544714d/hymenoptera_data' / sample_id.removeprefix('insects/')
        with Image.open(path) as image:
            return image.convert('RGB')

    def __getitem__(self, index):
        return self.transform(self.image(index)), self.labels[index]


def pretrain_transform(training, config):
    """Use random crops during pretraining and center crops during validation."""
    common = [transforms.ToTensor(), transforms.Normalize(config['mean'], config['std'])]
    spatial = ([transforms.RandomResizedCrop(config['size'], scale=tuple(config['random_resized_crop_scale'])),
                transforms.RandomHorizontalFlip(config['hflip'])] if training else
               [transforms.Resize(config['val_resize_short']), transforms.CenterCrop(config['val_center_crop'])])
    return transforms.Compose(spatial + common)


def imagenet_dataset(root, split, config):
    """Read class folders, select the configured classes, and apply image exclusions."""
    dataset = datasets.ImageFolder(root / split, pretrain_transform(split == 'train', config['input']))
    selected = (set(json.loads((ROOT / config['classes_file']).read_text()))
                if config.get('classes_file') else set(dataset.classes))
    classes = [name for name in dataset.classes if name in selected]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    remap = {i: class_to_idx[name] for i, name in enumerate(dataset.classes) if name in class_to_idx}
    excluded = set(json.loads((ROOT / config['exclusions_file']).read_text())[split])
    # Reindex selected class labels and remove image paths from the pretraining
    # exclusion list. Image pixels and downstream split IDs remain untouched.
    dataset.samples = [(path, remap[label]) for path, label in dataset.samples
                       if label in remap and Path(path).relative_to(root).as_posix() not in excluded]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    dataset.classes, dataset.class_to_idx = classes, class_to_idx
    return dataset


def training_datasets(config):
    """Return the training and validation datasets for the selected stage."""
    if config['stage'] == 'pretrain':
        root = ROOT / config['data_root']
        return tuple(imagenet_dataset(root, split, config) for split in ['train', 'val'])
    return tuple(ClassificationData(split, config['data_root'], config['split_manifest'], config['input'])
                 for split in ['train', 'val'])
