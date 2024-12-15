from clearml import Dataset

dataset = Dataset.create(
  dataset_name='FashionMNIST',
  dataset_project='dataset demo project', 
  dataset_version="1.0",
)
dataset.sync_folder("./data/")
dataset.upload()
dataset.finalize()

