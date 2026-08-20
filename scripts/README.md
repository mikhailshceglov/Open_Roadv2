# Scripts

One-off dataset preparation. Nothing here is imported by the package.

## `prepare_road_anomaly.py`

Flattens the RoadAnomaly release into the layout `configs/datasets/road_anomaly.yaml`
expects, and **remaps the anomaly label from 2 to 1**.

That remap is the point of the script. RoadAnomaly encodes anomalies as 2 while
almost everything else uses 1, and getting it wrong produces a run where every
metric is zero and nothing errors. The evaluator now stops on an empty ground
truth for the same reason, but converting once at preparation time is better than
catching it every run.

```bash
python scripts/prepare_road_anomaly.py --input /path/to/RoadAnomaly --output data/road_anomaly
```

## Datasets that need no script

TAD ships as folders of JPEG frames, so a clip is already a dataset — extract the
clips you want and point a config at each:

```bash
unzip -j tad.zip 'TAD/frames/abnormal/07_RoadSpills_028.mp4/*.jpg' \
  -d data/tad/07_RoadSpills_028
```

Note the directory inside the archive already ends in `.mp4`; it holds frames,
not a video.
