Code Description and Usage
Data Preparation:

Download the LIDC-IDRI and Kaggle DSB2017 datasets, organize them into the format needed in data.py

Implement the _load_data method to return a list of dictionaries containing tio.ScalarImage and tio.LabelMap

Training startup:

bash
python main.py --train_paths /path/to/train --val_paths /path/to/val --epochs 175
Key implementation details:

Stacked adjacent 5-layer slices as input using a 2.5D processing strategy

Dynamic focus loss dynamically adjusts difficult sample weights via the β parameter

Depth-supervised strategy enhances gradient propagation via auxiliary segmentation heads

3D-CPM module fuses contextual information through multi-scale 3D convolution

Performance Optimization Recommendations:

Use mixed precision training (add scaler = torch.cuda.amp.GradScaler())

Use multi-GPU parallelism (model = nn.DataParallel(model))

Adjust batch size to fit GPU video memory

The full implementation requires tuning the data loading section to the actual data format and optimizing model performance with hyperparameter search.

