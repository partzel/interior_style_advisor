from datasets import load_dataset, concatenate_datasets

def load_reddit_caption_dataset():
    dataset = load_dataset("SigLIP-forge/reddit-malelivingspace-captioned")
    all_batches = [ds for name, ds in dataset.items() if name.startswith("batch")]
    return concatenate_datasets(all_batches)