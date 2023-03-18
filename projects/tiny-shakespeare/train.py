from midigpt import TrainConfigure, Trainer
from midigpt.datasets import TextCharacterDataset, TextCharacterTokenizer

corpus_file_path = "tiny-shakespeare.txt"
tokenizer = TextCharacterTokenizer.from_file(corpus_file_path)

config = TrainConfigure(
    vocab_size=tokenizer.vocab_size,
    context_length=128,
    embedding_size=128,
    num_epochs=5,
    batch_size=120,
    num_heads=8,
    num_blocks=5,
)

dataset = TextCharacterDataset.from_file(corpus_file_path, tokenizer.vocabulary, config.context_length)
trainer = Trainer(config)

trainer.train(dataset)
