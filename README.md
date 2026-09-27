# Crop Disease Classifier

This project spots crop disease from a single leaf photo. It's built to run
on a phone, so a farmer or field worker can check a plant without an internet
connection.

It works in two steps. First, a **plant model** figures out which crop the leaf
comes from: soy, corn, rice, potato, or wheat. Then a smaller **disease model**
for that crop picks the disease. All six models are trained from scratch in
PyTorch and then shrunk down for Android.

Every model uses the same architecture: AlexNet with a
[CBAM](https://arxiv.org/abs/1807.06521) block added after each convolution
stage. CBAM is a light attention module. It helps the network focus on the
parts of the leaf that matter, like spots and color changes, instead of the
background.

## How it works

The pipeline has three scripts, and each one feeds the next:

```
data_prep.py           → *_balanced_dataset.pkl, key_dict.pkl
training.py            → outputs/<crop>/checkpoints/best_model.pth, plots, history
quantize_all_models.py → android_models/*.ptl, labels.json
```

### 1. Preparing the data

`models/data_prep.py` collects images for five crops. Three datasets come from
the Hugging Face Hub, and two come from local folders:

| Crop | Source | Classes |
| --- | --- | --- |
| Soy | `anandvermagmailcom/soybean-leaf-diseases` (HF) | bacterial blight, frogeye, healthy, soy rust |
| Corn | `Prachi1234/corn-leaf-disease` (HF) | leaf blight, common rust, gray leaf spot, healthy |
| Rice | `sharmin3/Rice-Leaf-Disease` (HF) | bacterial blight, blast, brown spot, healthy, tungro |
| Potato | `models/data/Potato Leaf Disease Dataset/` | Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, Virus |
| Wheat | `models/data/Wheat Dataset/` | BlackPoint, FusariumFootRot, HealthyLeaf, LeafBlight, WheatBlast |

Some diseases have far more photos than others. So the script cuts every class
down to the size of the smallest one, which keeps each disease equally
represented. After that, it joins all five crops into one set, labels each
image by crop, and balances that set the same way for the plant model. Each
dataset is saved as a pickle file, and the label maps go into `key_dict.pkl`.

### 2. Training

`models/training.py` trains the plant model first, then one disease model per
crop. They all share the same recipe:

- An 80/20 train/validation split, stratified so every class shows up in both
- 224×224 images with random crops, flips, small rotations, and light color
  changes during training
- Adam optimizer (learning rate 1e-3, weight decay 1e-4) with cross-entropy loss
- The learning rate halves when validation accuracy stops improving for 5 epochs
- 50 epochs with a batch size of 32

While it trains, the script saves a checkpoint every epoch and keeps the best
one as `best_model.pth`. It also logs to TensorBoard, saves a confusion matrix
every 10 epochs, and plots the loss and accuracy curves at the end.

On Windows, data loading runs in a single process (`num_workers=0`), because
multiprocessing with in-memory images isn't reliable there.

### 3. Exporting for Android

`models/quantize_all_models.py` gets the trained models ready for a phone. It
loads each `best_model.pth` and converts it to TorchScript. Next, it
quantizes the linear layers to 8-bit integers to make them smaller and faster,
and runs PyTorch's mobile optimizer. The result is one `.ptl` file per model,
plus a `labels.json` file the app uses to turn predictions back into names.

## Results

These are the best validation accuracies over 50 epochs, taken from
`models/outputs/<crop>/training_history.json`:

| Model | Classes | Best val acc | Notes |
| --- | --- | --- | --- |
| Plant model | 5 | 97.1% | Training and validation stay close |
| Corn | 4 | 92.2% | |
| Rice | 5 | 100% | Too perfect, needs a check |
| Wheat | 5 | 100% | Too perfect, needs a check |
| Potato | 7 | 40.6% | Only 68 images per class after balancing |
| Soy | 4 | 25.0% | Chance level, the model didn't learn |

The plant model and the corn model work well. The other results each tell a
different story:

- **Soy didn't train.** Accuracy stays at exactly 25% from the first epoch to
  the last, which is the same as guessing between four classes. This usually
  means training collapsed early, for example because the learning rate is too
  high for an AlexNet trained from scratch. It's not a sign that the data is
  too hard.
- **Potato doesn't have enough data.** The smallest class (Nematode) has only
  68 photos. Because every class gets cut down to that size, only 476 images
  are left for seven classes. Class weighting or oversampling would keep the
  other ~2,600 photos in play.
- **Rice and wheat at 100% need a closer look.** A perfect score often means
  the model found a shortcut, like near-duplicate photos in both splits or a
  background that gives the class away. It's worth checking for that before
  trusting these numbers.

The plots and confusion matrices for every model are in
`models/outputs/<crop>/`.

## Project structure

```
Crop Disease Classifier/
├── app/                        Android app (placeholder, not built yet)
└── models/
    ├── model.py                AlexNet + CBAM definition
    ├── data_prep.py            Download, balance, and save the datasets
    ├── training.py             Training loop for all six models
    ├── quantize_all_models.py  Export to quantized .ptl files for Android
    ├── data/                   Local potato and wheat images (not in git)
    ├── outputs/<crop>/         Plots and history (checkpoints and logs not in git)
    └── android_models/         labels.json (.ptl files not in git)
```

## Running it

All scripts use relative paths, so run them from inside `models/`:

```bash
cd models
```

```bash
pip install torch torchvision datasets pillow scikit-learn matplotlib seaborn tensorboard tqdm
```

```bash
python data_prep.py
```

```bash
python training.py
```

```bash
python quantize_all_models.py
```

After that, copy `android_models/*.ptl` into the app's
`app/src/main/assets/models/` folder and `labels.json` into
`app/src/main/assets/`.

**A note on memory:** `data_prep.py` keeps every decoded image in memory and
saves it all in pickle files. Together they come to about 44 GB, and wheat
alone is 22 GB. `training.py` then loads a whole pickle into RAM. Saving file
paths and loading images one at a time would bring that close to zero.

## Not in git

About 50 GB of files are left out by `.gitignore`: the pickled datasets, the
raw images in `models/data/`, the `.pth` checkpoints, the TensorBoard logs, and
the six `.ptl` exports (about 220 MB each). The pipeline above rebuilds all of
them.

## Known issues and next steps

- The soy model doesn't train yet (see Results).
- `app/` is still empty. The Android app isn't in this repository yet.
- Each `.ptl` file is about 220 MB. Quantization only shrinks the linear
  layers, and AlexNet's large 4096-wide classifier makes up most of the model.
  A mobile-first backbone like MobileNet or EfficientNet would be a better fit
  for phones.
- The label map is defined in three places (`data_prep.py`,
  `quantize_all_models.py`, and `key_dict.pkl`), so the copies can drift apart
  over time.
