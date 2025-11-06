"""
Data generation utilities for creating synthetic scene descriptions.
Provides a simple dataset of scene descriptions for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter


class SceneDescriptionDataset(Dataset):
    """
    Dataset of synthetic scene descriptions.
    Generates simple but structured descriptions of visual scenes.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        max_length: int = 50,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.max_length = max_length
        random.seed(seed)

        # Scene components
        self.colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray']
        self.objects = ['ball', 'cube', 'sphere', 'box', 'cylinder', 'pyramid', 'star', 'triangle', 'circle', 'square']
        self.sizes = ['tiny', 'small', 'medium', 'large', 'huge']
        self.positions = ['left', 'right', 'center', 'top', 'bottom', 'corner', 'middle']
        self.backgrounds = ['plain', 'gradient', 'textured', 'striped', 'dotted']
        self.actions = ['sitting', 'floating', 'rotating', 'glowing', 'pulsing']

        # Generate descriptions
        self.descriptions = self._generate_descriptions()

        # Build vocabulary
        self.vocab, self.inv_vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)

        # Tokenize all descriptions
        self.tokenized_descriptions = [self._tokenize(desc) for desc in self.descriptions]

    def _generate_descriptions(self) -> List[str]:
        """Generate synthetic scene descriptions"""
        descriptions = []

        templates = [
            "a {size} {color} {object} in the {position}",
            "a {color} {object} {action} on a {background} background",
            "{size} {color} {object} and {size2} {color2} {object2}",
            "a scene with a {color} {object} {action}",
            "a {background} background with a {color} {object}",
            "{color} {object} on the {position} and {color2} {object2} on the {position2}",
            "multiple {color} {objects} arranged in a pattern",
            "a {size} {object} {action} near a {size2} {object2}",
            "abstract scene with {color} and {color2} shapes",
            "geometric {object} in {color} against {background} surface"
        ]

        for _ in range(self.num_samples):
            template = random.choice(templates)
            description = template.format(
                size=random.choice(self.sizes),
                size2=random.choice(self.sizes),
                color=random.choice(self.colors),
                color2=random.choice(self.colors),
                object=random.choice(self.objects),
                object2=random.choice(self.objects),
                objects=random.choice(self.objects) + 's',
                position=random.choice(self.positions),
                position2=random.choice(self.positions),
                background=random.choice(self.backgrounds),
                action=random.choice(self.actions)
            )
            descriptions.append(description.lower())

        return descriptions

    def _build_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from all descriptions"""
        # Special tokens
        vocab = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }

        # Collect all words
        all_words = []
        for desc in self.descriptions:
            words = desc.split()
            all_words.extend(words)

        # Count word frequencies
        word_counts = Counter(all_words)

        # Add words to vocabulary (keeping only frequent ones)
        idx = len(vocab)
        for word, count in word_counts.most_common():
            if count >= 2:  # Minimum frequency threshold
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1

        # Create inverse vocabulary
        inv_vocab = {v: k for k, v in vocab.items()}

        return vocab, inv_vocab

    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        words = text.split()
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        return tokens

    def _pad_sequence(self, tokens: List[int], max_len: int) -> Tuple[List[int], List[int]]:
        """Pad or truncate sequence to max_len"""
        # Add SOS and EOS tokens
        tokens = [self.vocab['<SOS>']] + tokens + [self.vocab['<EOS>']]

        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            tokens[-1] = self.vocab['<EOS>']

        # Create attention mask
        attention_mask = [1] * len(tokens)

        # Pad if necessary
        padding_length = max_len - len(tokens)
        if padding_length > 0:
            tokens.extend([self.vocab['<PAD>']] * padding_length)
            attention_mask.extend([0] * padding_length)

        return tokens, attention_mask

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        tokens = self.tokenized_descriptions[idx]
        padded_tokens, attention_mask = self._pad_sequence(tokens, self.max_length)

        return {
            'input_ids': torch.tensor(padded_tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(padded_tokens[1:], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:-1], dtype=torch.long),
            'original_text': self.descriptions[idx]
        }

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text"""
        tokens = token_ids.cpu().numpy().tolist()
        words = []

        for token in tokens:
            if token == self.vocab['<EOS>']:
                break
            if token in self.inv_vocab and token not in [self.vocab['<PAD>'], self.vocab['<SOS>']]:
                words.append(self.inv_vocab[token])

        return ' '.join(words)


class SceneCollator:
    """
    Custom collator for batching scene descriptions.
    Handles dynamic padding and batch preparation.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        # Extract fields
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])

        # Prepare labels for loss computation (mask padding)
        labels[labels == self.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'texts': [item['original_text'] for item in batch]
        }


def create_data_loaders(
    batch_size: int = 32,
    num_samples: int = 10000,
    train_split: float = 0.8,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, SceneDescriptionDataset]:
    """
    Create train and validation data loaders.

    Args:
        batch_size: Batch size for training
        num_samples: Total number of samples to generate
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, dataset)
    """
    # Create dataset
    dataset = SceneDescriptionDataset(num_samples=num_samples, seed=seed)

    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create collator
    collator = SceneCollator(pad_token_id=dataset.vocab['<PAD>'])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    return train_loader, val_loader, dataset


def extract_color_words(text: str, color_list: List[str]) -> List[str]:
    """
    Extract color words from text.
    Useful for color consistency tracking.
    """
    words = text.lower().split()
    return [word for word in words if word in color_list]


def generate_augmented_descriptions(
    base_descriptions: List[str],
    augmentation_factor: int = 2
) -> List[str]:
    """
    Generate augmented versions of descriptions.
    Adds variations while maintaining semantic meaning.
    """
    augmented = []

    synonyms = {
        'small': ['tiny', 'little', 'petite'],
        'large': ['big', 'huge', 'giant'],
        'center': ['middle', 'central'],
        'ball': ['sphere', 'orb'],
        'box': ['cube', 'block']
    }

    for desc in base_descriptions:
        augmented.append(desc)

        for _ in range(augmentation_factor - 1):
            aug_desc = desc
            for word, syns in synonyms.items():
                if word in aug_desc:
                    replacement = random.choice(syns)
                    aug_desc = aug_desc.replace(word, replacement)
            augmented.append(aug_desc)

    return augmented