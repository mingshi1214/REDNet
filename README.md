# REDNet
REDNet trained on thermal images from the [FLIR dataset](https://www.flir.ca/oem/adas/adas-dataset-form/ "FLIR dataset")

To automatically train and test all models with 32, 64, 128, 256 batch sizes: 

`python3 auto.py`

To train one model: 

```bash
python3 main.py --arch "REDNet30" \ # REDNet10, REDNet20 REDNet30
                --images_dir "" \ # ie. flir_data/images_thermal_train/data
                --outputs_dir "" \ # ie. results/REDNet10/32
                --jpeg_quality 10 \
                --patch_size 50 \
                --batch_size 32 \
                --num_epochs 2 \
                --threads 8 \
                --seed 123 \
                --val_images_dir "" \ # ie. flir_data/images_thermal_val/data
```

To test one model: 

```bash
python3 example.py --arch "REDNet30" \ # REDNet10, REDNet20, REDNet30
                   --weights_path "" \ # ie. results/REDNet10/32/REDNet10_weights.pth
                   --images_dir "" \ # ie. flir_data/images_thermal_val
                   --outputs_dir "" \ # ie. epoch_weights/REDNet10/32/test_images
                   --jpeg_quality 10
```

