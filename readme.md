# DeepSORT
**SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC**

1. Download MOT17 from the [official website](https://motchallenge.net/).

   ```
   path_to_dataset/MOTChallenge
   ├── MOT17
   	│   ├── test
   	│   └── train
   ```

## Requirements

- pytorch
- opencv
- scipy
- sklearn

For example, we have tested the following commands to create an environment for DeepSORT:

```shell
conda create -n deepsort python=3.8 -y
conda activate deepsort
pip3 install torch torchvision torchaudio
pip install opencv-python
pip install scipy
pip install scikit-learn==0.19.2
```

## Tracking

1. 首先将dinov2的ckpt_g.t7文件放入others文件夹, 其中ckpt_g.t7文件可以从others/model.py中训练得到，训练数据集为Market1501数据集，该数据集可以在（https://www.kaggle.com/datasets/pengcw1/market-1501/data）中下载得到
2. ```python others/generate_detections.py``` 从MOT17的边界框中获取.npy检测文件
3. ```python strong_sort.py MOT17 val --BoT``` 运行deepsort生成检测结果
4. 将检测结果output_deepsort放入TrackEval/data/trackers/mot_challenge文件夹 运行以下命令:
```
cd TrackEval/scripts
python run_mot_challenge.py \                                     
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKER_SUB_FOLDER '' \
    --METRICS HOTA CLEAR Identity VACE \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1 \
    --GT_LOC_FORMAT '{gt_folder}/{seq}/gt/gt_val_half_v2.txt' \
    --OUTPUT_SUMMARY False \
    --OUTPUT_EMPTY_CLASSES False \
    --OUTPUT_DETAILED False \
    --PLOT_CURVES False
```
以查看评估结果