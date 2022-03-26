import subprocess

train_path = "flir_data/images_thermal_train/data"
val_path = "flir_data/images_thermal_val/data"

models = ["REDNet10", 'REDNet20', "REDNet30"]

batches = ["32", "64", "128", "256"]

epochs = 30

for model in models:
    for batch in batches:
        results_path = "results/" + model + "/" + batch
        cmd = ["python3", "main.py", 
                                "--arch", "{}".format(model), 
                                "--images_dir", "{}".format(train_path), 
                                "--outputs_dir", "{}".format(results_path),
                                "--jpeg_quality", "10",
                                "--patch_size", "50",
                                "--batch_size" , "{}".format(batch), 
                                "--num_epochs", "{}".format(epochs),
                                "--threads", "8",
                                "--seed", "123", 
                                "--val_images_dir", "{}".format(val_path)]
        print("Starting "+ model+ " batch " + batch)
        list_files = subprocess.run(cmd)
        print("Finished {}, batch {}. ".format(model, batch) + "The exit code was: %d" % list_files.returncode)