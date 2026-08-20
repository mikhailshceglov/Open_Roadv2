# Технический отчёт: RAAS, Objectomaly и global/airborne fusion

Дата отчёта: 19 августа 2026 года.

## 1. Краткое резюме

В проекте построена воспроизводимая система сегментации дорожных аномалий,
которая объединяет исходный RAAS/Maskomaly, Mask2Former, CLIP, официальный
Objectomaly и SAM. Реализованы и сопоставлены две группы методов:

1. исходные варианты RAAS: `maskomaly`, `maskomaly_id`, `maskomaly_ood`;
2. экспериментальные варианты RAAS + Objectomaly + global/airborne fusion:
   `objectomaly_global_maskomaly`, `objectomaly_global_maskomaly_id`,
   `objectomaly_global_maskomaly_ood`.

Итог эксперимента неоднозначен:

- на `ObstacleTrack-validation` global fusion особенно хорошо работает вместе
  с `maskomaly_id/ood`: mean F1 вырос с 58.65 до 73.00, PPV — с 53.61 до
  89.35, AUPR — с 91.21 до 92.37;
- на `AnomalyTrack-validation` все global-варианты ухудшились: mean F1 упал
  примерно на 24 процентных пункта, главным образом из-за большого количества
  ложных объектных компонент;
- текущий исследовательский pipeline не пригоден для online-инференса: на
  NVIDIA A100 80 GB он обрабатывает один кадр примерно за 58–64 секунды.

Следовательно, global-ветка подтверждает полезность SAM/CLIP refinement для
малых дорожных препятствий, но пока не является универсальной заменой RAAS и
нуждается одновременно в настройке precision на AnomalyTrack и в радикальной
оптимизации времени.

## 2. Происхождение компонентов и вклад интеграции

Важно разделять внешние архитектуры и код, реализованный в рамках проекта.

Внешняя основа:

- Detectron2 и Mask2Former используются как segmentation backbone;
- исходный RAAS/Maskomaly формирует coarse anomaly map;
- официальный `road-anomaly-benchmark` используется без изменения формул;
- Objectomaly, SAM, OASC и MBP подключены из официального Objectomaly.

Реализовано в данном проекте:

- единый официальный SMIYC evaluator для всех вариантов RAAS;
- корректный контракт RGB loader → BGR RAAS → float anomaly map → официальный
  float16 HDF5;
- cache-first bridge между несовместимыми окружениями RAAS и Objectomaly;
- экспорт semantic map из Mask2Former без повторного запуска backbone;
- global/airborne fusion без жёсткого road ROI;
- batched CLIP validation компактных SAM-регионов по normal/OOD prompt
  ensembles;
- поддержка аномалий в любой области кадра, включая небо;
- inference произвольной папки без ground truth с визуализациями;
- общий запуск шести model–dataset комбинаций;
- CUDA-синхронизированный покадровый profiler;
- повторное использование одного SAM mask cache всеми тремя RAAS-вариантами;
- unit/integration tests cache-контракта, карт, evaluator bridge и timing report.

Зафиксированные внешние версии:

| Компонент | Версия |
|---|---|
| `road-anomaly-benchmark` | `1c7804e0749686452eb02bbcfb81234e38795f8a` |
| Objectomaly | `66d2ad2a1b02d79389f4265d9d1d99ab6412324f` |
| OpenAI CLIP | `d50d76daa670286dd6cacf3bcd80b5e4823fc8e1` |
| SAM checkpoint | `sam_vit_h_4b8939.pth`, MD5 `4b8939a88964f0f4ff5f5b2642c598a6` |

Локальный HEAD на момент составления отчёта:
`b5a021c8eb09448ea19ababbad7cd9f7177d6ebb`.

Metric-файлы не содержат hash проекта и checksum RAAS checkpoint внутри
своего metadata. Поэтому связь результатов с этим HEAD предполагается по
workflow, но не доказана криптографически. Для публикационного пакета следует
добавить commit и SHA-256 весов.

## 3. Общая архитектура

```text
Официальный RGB-кадр
        |
        v
RGB -> BGR
        |
        v
Mask2Former / Detectron2
        |-------------------------------|
        |                               |
        v                               v
query logits + query masks       semantic Cityscapes map
        |
        v
RAAS coarse anomaly map [0, 1]
        |
        |             SAM ViT-H -> MaskBundle
        |                         (masks, bbox, area, quality)
        |                              |
        v                              v
Objectomaly OASC calibration <---------|
        |
        v
soft semantic attenuation
        |
        v
global SAM candidate selection
        |
        v
batched CLIP normal/OOD validation
        |
        v
candidate restoration / airborne protection
        |
        v
Objectomaly MBP boundary refinement
        |
        v
final float32 anomaly map [0, 1]
        |
        v
official Evaluation.save_output -> float16 HDF5
        |
        v
PixBinaryClass -> SegEval track-specific metrics
```

Mask2Former запускается один раз на кадр. Из того же forward pass извлекаются
query class logits, query masks и семантическая Cityscapes-карта.

## 4. Исходная архитектура RAAS

### 4.1 Общий backbone

Все три RAAS-варианта используют Mask2Former и модифицированный
`DefaultPredictor`, возвращающий:

```text
segmentation["sem_seg"]       C x H x W
mask_cls_result               N_queries x (N_classes + 1)
mask_pred_result              N_queries x H x W
```

Далее:

- к class logits применяется softmax;
- к mask logits применяется sigmoid;
- semantic map получается через `argmax` по `sem_seg`;
- результаты переносятся в NumPy.

### 4.2 `maskomaly`

Базовая anomaly map строится из двух сигналов.

Promotion map:

- используются query indices `[49, 31, 83, 32]`;
- для каждого query маска умножается на максимальную вероятность класса;
- по query берётся pixel-wise maximum.

Rejection map:

- начальное значение равно единице;
- для любого query, чей лучший класс не void `19` и confidence больше `0.7`,
  вычисляется `1 - mask * confidence`;
- по всем таким query берётся pixel-wise minimum;
- пересечения уверенных non-void masks дополнительно подавляются;
- query indices `[19, 24]` используются для коррекции ошибочной земли.

Итог:

```text
RAAS = 0.6 * rejection_map + 0.4 * promotion_map
```

Карта интерполируется до исходного размера изображения.

### 4.3 `maskomaly_id`

Сначала строится та же базовая RAAS-карта. Затем выполняется road-aware CLIP
ветка:

1. query mask `20` бинаризуется при threshold `0.5`;
2. внешние контуры заполняются и образуют road polygon;
3. разрывы между polygon и исходной road mask рассматриваются как candidates;
4. candidates разбиваются на connected components;
5. bbox каждого компонента классифицируется CLIP ViT-B/32 по 19 нормальным
   Cityscapes prompts;
6. если максимальная ID probability больше `0.85`, score компонента
   устанавливается в `0.05`; иначе — в `1.0`.

CLIP и tokenized prompts загружаются один раз при создании модели, а не для
каждого кадра.

### 4.4 `maskomaly_ood`

Геометрия road candidates совпадает с `maskomaly_id`, но к 19 ID prompts
добавлены три OOD prompts:

- `something unusual in a driving scene`;
- `an unexpected object in the road environment`;
- `a strange or unknown item on the street`.

Решение:

- если `max_ood_prob < 0.001`, компонент считается ID;
- если `max_id_prob > max_ood_prob`, `max_id_prob > 0.85` и разница больше
  `0.1`, компонент считается ID;
- иначе компонент считается OOD и получает score `1.0`.

В проведённых экспериментах `maskomaly_id` и `maskomaly_ood` дали полностью
одинаковые метрики как до, так и после global refinement. Это означает, что
различие prompt/decision ветвей на данных 40 validation-кадров не изменило
финальные anomaly maps в значимой для evaluator форме. Это следует проверить
прямым сравнением массивов или HDF5 checksum.

## 5. Cache-first Objectomaly bridge

RAAS и Objectomaly работают в отдельных окружениях:

- RAAS: Python 3.8, PyTorch 1.9, CUDA 11.1;
- Objectomaly bridge: Python 3.10, PyTorch 2.1, torchvision 0.16, CUDA 11.8.

Pipeline разделён на воспроизводимые стадии.

### Стадия A: RAAS export

Для каждого кадра сохраняются:

- lossless BGR `uint8` image;
- coarse RAAS anomaly map `float32` в `[0, 1]`;
- Mask2Former semantic map;
- dataset, frame ID, размер и timing;
- manifest с моделью, runtime и setup metadata.

### Стадия B: SAM masks

SAM ViT-H создаёт automatic masks. После фильтрации они сохраняются как
`MaskBundle`, содержащий masks, bbox, area и quality. Cache привязан к dataset,
frame ID, generator и точному SAM/postprocess config.

Один SAM cache используется всеми тремя RAAS-вариантами, поскольку изображения
и frame IDs одинаковы. Это исключает тройной фактический запуск SAM при полном
эксперименте.

### Стадия C: Objectomaly refinement

Последовательно выполняются:

1. OASC — region-aware calibration coarse map;
2. опциональная global fusion;
3. MBP — boundary refinement;
4. сохранение refined float32 map и diagnostics manifest.

### Стадия D: официальный импорт

Refined map загружается и сохраняется только через официальный
`Evaluation.save_output`, после чего запускаются официальные pixel- и
component-level метрики.

## 6. Реализованная global/airborne fusion

Цель global fusion — не ограничивать поиск жёсткой road ROI. Это позволяет
детектировать шины, детали автомобиля, мусор и строительные объекты как на
дороге, так и в воздухе/небе.

### 6.1 SAM

Экспериментальный config:

| Параметр | Значение |
|---|---:|
| Backbone | SAM ViT-H |
| `points_per_side` | 64 |
| `pred_iou_thresh` | 0.86 |
| `stability_score_thresh` | 0.92 |
| CPU NMS workaround | включён |

Сетка 64 x 64 выбрана для небольших объектов, но является главным источником
вычислительных затрат. Из-за ошибки mixed-device indexing в torchvision 0.16
только NMS принудительно выполняется на CPU; SAM encoder/decoder остаётся на
CUDA.

Postprocess:

```text
q_min = 0.86
area_frac_min = 0.00002
area_frac_max = 0.5
nms_iou = 0.7
partition = small_to_large
```

### 6.2 OASC

Используется вариант `quality_aware_residual_blending`:

```text
base_lambda = 0.5
alpha = 0.005
steepness = 20
aggregation = trimmed_mean, trim = 0.1
quality/area/anomaly-support/overlap factors = enabled
max_lambda = 0.6
```

OASC калибрует coarse RAAS score внутри SAM regions с учётом качества,
площади, anomaly support и overlap.

### 6.3 Soft semantic prior

Semantic map не используется как hard mask. Вместо этого RAAS/OASC score
умножается на class-specific factor:

| Класс | Factor |
|---|---:|
| road | 1.00 |
| sidewalk | 0.85 |
| building, wall | 0.55 |
| fence | 0.65 |
| pole | 0.80 |
| traffic light/sign | 0.90 |
| vegetation | 0.55 |
| terrain | 0.65 |
| sky | 0.35 |
| person/rider/vehicles | 0.90 |

Для road применяется дополнительный boost `1.05`. Ни одна область полностью
не обнуляется.

### 6.4 Global SAM candidates

Рассматриваются SAM masks с относительной площадью от `0.00002` до `0.03`.
Для региона вычисляются:

- quantile `0.9` исходного anomaly score;
- median score кольца вокруг маски с radius `9`;
- local contrast между region и ring;
- SAM quality;
- dominant semantic class.

Candidates сортируются по `(score + positive_contrast) * quality`, после чего
оставляются максимум 96.

Геометрический candidate считается подтверждённым при:

```text
score >= 0.45 and contrast >= 0.12
```

В текущем config `background_class_ids` содержит все 19 Cityscapes-классов.
Поэтому ветка `semantic_object` подтверждает только неизвестный semantic ID;
для распознанных классов основными основаниями остаются contrast и CLIP. Это
важный фактический нюанс текущей реализации.

### 6.5 Batched CLIP validation

Для candidates со score не меньше `0.05` запускается CLIP ViT-B/32.

Для каждого candidate используются два представления:

1. masked crop, где фон заменён серым;
2. context crop с padding 15%.

Features двух представлений усредняются и сравниваются с normal и OOD prompt
ensembles. Batch size равен 16.

Normal prompts описывают небо, облака, блики, туман, фонари, провода,
ограждения, разметку, растительность, обычные машины и грузовики. OOD prompts
описывают летящие детали, шины, мусор из автомобиля, падающий груз,
строительный мусор и неизвестные препятствия на дороге/в воздухе.

CLIP margin:

```text
margin = max(OOD similarity) - max(normal similarity)
```

При `margin >= 0.0` candidate считается CLIP-OOD. Его score восстанавливается
как минимум до `0.65`. Candidate также может быть защищён геометрическим
contrast-критерием. Итоговая карта ограничивается диапазоном `[0, 1]`.

### 6.6 MBP

После global fusion применяется `boundary_band_residual`:

```text
dilation_radius = 1
gaussian_sigma = 0.7
gaussian_kernel = 5 x 5
lambda = 0.25
```

Цель — уточнить границы, не разрушая внутренний anomaly score.

## 7. Официальный протокол SMIYC

Используются только публичные validation split:

| Каталог | Split | Кадров |
|---|---|---:|
| `dataset_AnomalyTrack` | `AnomalyTrack-validation` | 10 |
| `dataset_ObstacleTrack` | `ObstacleTrack-validation` | 30 |

Контракт prediction:

- одна карта `H x W` на кадр;
- `float32`, finite, диапазон `[0, 1]`;
- большее значение означает более аномальный пиксель;
- при несовпадении размера используется bilinear resize;
- официальный RGB loader преобразуется в BGR перед RAAS;
- `Evaluation.save_output` сохраняет карту в float16 HDF5;
- метрики считаются по сохранённой HDF5-карте.

GT после официального loader:

```text
0   normal
1   anomaly / obstacle
255 ignore / void
```

ROI и ignore обрабатываются только официальным loader/evaluator.

### 7.1 Pixel-level

`PixBinaryClass` агрегирует confusion matrices по всем валидным пикселям
split, а не усредняет изображения.

- AUPR: площадь под precision–recall в официальной AP-подобной реализации;
- FPR@95: false-positive rate в первой точке с TPR не меньше 95%;
- для каждой model–dataset пары выбирается собственный threshold максимального
  pixel F1.

### 7.2 Component-level

Карта бинаризуется как `prediction > best_pixel_f1_threshold`, после чего
строятся 8-связные компоненты.

Минимальные размеры:

| Track | Predicted component | GT component |
|---|---:|---:|
| AnomalyTrack | 500 px | 100 px |
| ObstacleTrack | 50 px | 10 px |

- `sIoU GT` — средний adjusted IoU по GT-компонентам;
- `PPV` — средняя доля корректных пикселей по predicted-компонентам;
- `mean F1` — среднее component F1 по 11 sIoU/PPV thresholds от 0.25 до 0.75.

`mean F1` не является pixel F1. Поэтому высокий AUPR может сочетаться с
низким mean F1 при фрагментации, неточных границах или ложных объектах.

Все метрики ниже выражены в процентах; разности — в процентных пунктах.

## 8. Результаты исходных RAAS-моделей

| Модель | Dataset | AUPR | FPR@95 | sIoU GT | PPV | mean F1 |
|---|---|---:|---:|---:|---:|---:|
| `maskomaly` | AnomalyTrack | 94.09 | 3.40 | 80.07 | 44.86 | 63.60 |
| `maskomaly` | ObstacleTrack | 88.62 | 2.81 | 49.19 | 54.06 | 53.21 |
| `maskomaly_id` | AnomalyTrack | 93.18 | 3.42 | 80.10 | 43.43 | 62.26 |
| `maskomaly_id` | ObstacleTrack | 91.21 | 0.42 | 55.16 | 53.61 | 58.65 |
| `maskomaly_ood` | AnomalyTrack | 93.18 | 3.42 | 80.10 | 43.43 | 62.26 |
| `maskomaly_ood` | ObstacleTrack | 91.21 | 0.42 | 55.16 | 53.61 | 58.65 |

## 9. Результаты global/airborne fusion

| Модель | Dataset | AUPR | FPR@95 | sIoU GT | PPV | mean F1 |
|---|---|---:|---:|---:|---:|---:|
| `global_maskomaly` | AnomalyTrack | 89.48 | 5.45 | 72.68 | 23.47 | 39.49 |
| `global_maskomaly` | ObstacleTrack | 75.12 | 0.34 | 65.44 | 60.66 | 65.46 |
| `global_maskomaly_id` | AnomalyTrack | 88.85 | 5.47 | 72.65 | 22.59 | 38.36 |
| `global_maskomaly_id` | ObstacleTrack | 92.37 | 0.30 | 51.64 | 89.35 | 73.00 |
| `global_maskomaly_ood` | AnomalyTrack | 88.85 | 5.47 | 72.65 | 22.59 | 38.36 |
| `global_maskomaly_ood` | ObstacleTrack | 92.37 | 0.30 | 51.64 | 89.35 | 73.00 |

### 9.1 Изменение относительно соответствующей RAAS-модели

#### AnomalyTrack

| Модель | ΔAUPR | ΔFPR@95 | ΔsIoU | ΔPPV | Δmean F1 |
|---|---:|---:|---:|---:|---:|
| `maskomaly` | -4.61 | +2.05 | -7.39 | -21.39 | -24.11 |
| `maskomaly_id` | -4.33 | +2.05 | -7.45 | -20.84 | -23.90 |
| `maskomaly_ood` | -4.33 | +2.05 | -7.45 | -20.84 | -23.90 |

Для FPR отрицательная разность является улучшением, положительная —
ухудшением.

На AnomalyTrack global fusion ухудшает все показатели. Главная проблема —
падение PPV примерно вдвое. Pipeline создаёт слишком много ложных predicted
components, вероятно из-за мягкого CLIP threshold `0.0`, низкого probe score
`0.05`, большого лимита 96 candidates и восстановления score до `0.65`.

#### ObstacleTrack

| Модель | ΔAUPR | ΔFPR@95 | ΔsIoU | ΔPPV | Δmean F1 |
|---|---:|---:|---:|---:|---:|
| `maskomaly` | -13.50 | -2.47 | +16.25 | +6.60 | +12.25 |
| `maskomaly_id` | +1.16 | -0.12 | -3.52 | +35.74 | +14.35 |
| `maskomaly_ood` | +1.16 | -0.12 | -3.52 | +35.74 | +14.35 |

Лучший сбалансированный результат дают `global_maskomaly_id/ood`: AUPR 92.37,
FPR@95 0.30, PPV 89.35 и mean F1 73.00. Небольшое падение sIoU означает,
что найденные компоненты очень чистые, но покрытие GT-объекта несколько хуже.

У `global_maskomaly` sIoU значительно растёт, но AUPR падает. Это означает,
что формы объектов при выбранном operating threshold улучшились, однако
глобальное ранжирование pixel scores по всем thresholds стало хуже.

## 10. Методика profiler

Каждая стадия измерялась через `time.perf_counter()` с
`torch.cuda.synchronize()` непосредственно до и после вызова. Это устраняет
занижение времени из-за асинхронного CUDA execution.

Измерены:

- `raas_inference`;
- `raas_postprocess`;
- `sam_generate`;
- `sam_postprocess`;
- `oasc`;
- `global_fusion`, включая CLIP;
- `mbp`;
- последовательная сумма `end_to_end_compute`.

Время чтения/записи изображений и cache, расчёт метрик и переключение conda
окружений не включены. Следовательно, `end_to_end_compute` является нижней
оценкой реальной wall-clock latency.

Первый кадр исключён из общей steady-state статистики из-за warm-up и ленивой
инициализации. Steady-state содержит 39 кадров. Per-dataset breakdown включает
все кадры соответствующего split, поэтому AnomalyTrack breakdown содержит
первый cold frame.

Objectomaly profiler выполнялся в среде:

| Параметр | Значение |
|---|---|
| GPU | NVIDIA A100 80GB PCIe |
| Python | 3.10.20 |
| PyTorch | 2.1.2 |
| torchvision | 0.16.2 |
| CUDA runtime | 11.8 |

RAAS inference измерялся в отдельном `raas` environment; поле `runtime` в
финальном timing JSON описывает именно refinement environment.

## 11. Сводный отчёт profiler

### 11.1 End-to-end steady-state

| Модель | Mean | p50 | p95 | Max | FPS по mean | Кадров/мин |
|---|---:|---:|---:|---:|---:|---:|
| `maskomaly` | 63.92 s | 57.38 s | 108.80 s | 139.85 s | 0.0156 | 0.94 |
| `maskomaly_id` | 60.47 s | 58.01 s | 100.56 s | 115.09 s | 0.0165 | 0.99 |
| `maskomaly_ood` | 58.44 s | 56.42 s | 94.58 s | 106.87 s | 0.0171 | 1.03 |

### 11.2 Mean latency по стадиям

| Стадия | `maskomaly` | `maskomaly_id` | `maskomaly_ood` |
|---|---:|---:|---:|
| RAAS inference | 3.36 s | 2.47 s | 2.49 s |
| RAAS postprocess | 0.02 s | 0.01 s | 0.01 s |
| SAM generation | 28.54 s | 28.54 s | 28.54 s |
| SAM postprocess | 12.80 s | 12.80 s | 12.80 s |
| OASC | 2.54 s | 1.68 s | 1.41 s |
| Global fusion + CLIP | 15.41 s | 14.72 s | 12.97 s |
| MBP | 1.26 s | 0.24 s | 0.21 s |

SAM generation и postprocess суммарно занимают около 41.34 секунды:

- 64.7% полного времени для `maskomaly`;
- 68.4% для `maskomaly_id`;
- 70.7% для `maskomaly_ood`.

Global fusion + CLIP занимает ещё 22–24%. RAAS — только 4–5% нового полного
pipeline. Относительно одной RAAS новая архитектура медленнее примерно в 19
раз для `maskomaly` и в 23–24 раза для ID/OOD вариантов.

### 11.3 По датасетам

| Модель | AnomalyTrack mean / p95 | ObstacleTrack mean / p95 |
|---|---:|---:|
| `maskomaly` | 81.71 / 134.78 s | 58.21 / 74.72 s |
| `maskomaly_id` | 75.32 / 112.30 s | 55.63 / 66.53 s |
| `maskomaly_ood` | 72.17 / 105.34 s | 53.88 / 65.22 s |

Global fusion на AnomalyTrack особенно дорога: 26–33 секунды против 8–10
секунд на ObstacleTrack. Это указывает на большее число/стоимость candidate
crops либо более сложные маски в этих кадрах.

### 11.4 Setup latency

| Модель | RAAS load | SAM setup | CLIP load | Итого в отчёте |
|---|---:|---:|---:|---:|
| `maskomaly` | 11.43 s | 10.39 s | 16.67 s | 38.49 s |
| `maskomaly_id` | 14.97 s | shared cache | 17.62 s | 32.59 s |
| `maskomaly_ood` | 14.86 s | shared cache | 16.13 s | 30.99 s |

Setup — разовая стоимость процесса и не включается в steady-state latency.
SAM setup записан только у первого варианта, который создавал общий cache.

## 12. Пригодность для online

Ориентиры latency:

| Частота | Максимальный бюджет кадра |
|---|---:|
| 30 FPS | 33.3 ms |
| 20 FPS | 50 ms |
| 10 FPS | 100 ms |
| 5 FPS | 200 ms |

Текущий лучший mean равен 58.44 секунды, p95 — 94.58 секунды. Следовательно,
реализация не пригодна для online даже на A100. На RTX 3050 Mobile ожидается
ещё большая задержка. Кроме того, cache-first pipeline с двумя conda
окружениями является evaluation architecture, а не production deployment.

Для реального online-варианта модели должны постоянно находиться в памяти
одного процесса либо нескольких resident services, а benchmark должен
измерять фактическую wall-clock latency с preprocessing, передачей данных и
памятью.

## 13. Рекомендованные следующие эксперименты

### Качество

1. Для снижения false positives на AnomalyTrack поднять
   `clip_margin_threshold`, `min_score` и `min_contrast`.
2. Уменьшить `max_candidates` и повысить `clip_probe_min_score`.
3. Не устанавливать всем CLIP-OOD candidates одинаковый floor `0.65`, а
   калибровать добавку непрерывно по CLIP margin и RAAS support.
4. Разделить конфигурации для AnomalyTrack и ObstacleTrack либо обучить общий
   gating на отдельном tuning split.
5. Сравнить HDF5 `maskomaly_id` и `maskomaly_ood`, поскольку метрики полностью
   совпадают.

### Скорость

1. Снизить SAM `points_per_side` с 64 до 32: число grid prompts уменьшится
   примерно в четыре раза.
2. Устранить CPU NMS workaround; SAM postprocess сейчас занимает 12.8 s.
3. Проверить SAM ViT-B, MobileSAM или более лёгкий region proposal model с
   обязательным полным пересчётом метрик.
4. Снизить `max_candidates` с 96 до 32–48.
5. На A100 увеличить CLIP batch size после проверки VRAM.
6. Запускать дорогую SAM/CLIP ветку условно, только для кадров или зон с
   достаточным RAAS uncertainty/anomaly support.
7. Исключить Python/NumPy loops из OASC, postprocess и candidate statistics,
   оставив операции на GPU.

Наиболее информативный следующий controlled experiment:

```text
points_per_side: 64 -> 32
max_candidates: 96 -> 32
clip_probe_min_score: 0.05 -> более строгий порог
```

После этого необходимо заново измерить одновременно метрики и profiler: нельзя
оценивать ускорение без проверки потери малых препятствий.

## 14. Воспроизводимый запуск

Из корня RAAS:

```bash
git submodule update --init --recursive

bash Maskomaly/scripts/run_global_smiyc_eval.sh \
  /absolute/path/to/datasets \
  results/smiyc_global_fusion \
  checkpoints/sam_vit_h_4b8939.pth
```

Результаты:

```text
results/smiyc_global_fusion/metrics/summary.csv
results/smiyc_global_fusion/metrics/summary.json
results/smiyc_global_fusion/refined/timings-<model>.csv
results/smiyc_global_fusion/refined/timings-summary-<model>.json
```

Для текущего переданного пакета они скопированы в каталог `metrics/` проекта.

## 15. Ограничения и статус

- Результаты относятся только к публичным validation split, не к закрытому
  SMIYC test leaderboard.
- Global config фактически настроен на validation данных; переносимость на
  другие домены не доказана.
- Всего оценено 40 кадров на модель, поэтому latency p95 имеет ограниченную
  статистическую устойчивость.
- Timing не включает I/O и межпроцессную передачу.
- Финальный runtime metadata отражает Objectomaly environment, а не обе среды
  одновременно.
- Objectomaly commit не содержит project-level LICENSE. Он подключён как
  неизменённый submodule; перед публикацией объединённого кода или контейнера
  требуется письменное разрешение либо официальная лицензия upstream.
- Последний локальный test run интеграции: 120 tests passed.

## 16. Итоговый вывод

Проект реализовал не просто новый postprocessing, а полный экспериментальный
контур: от Mask2Former/RAAS inference и глобальных SAM/CLIP candidates до
официального SMIYC evaluation и CUDA profiling.

`global_maskomaly_id/ood` является наиболее перспективной архитектурой для
малых дорожных препятствий: mean F1 73.00 и PPV 89.35 на ObstacleTrack. Однако
она пока не универсальна: AnomalyTrack mean F1 падает до 38.36, а latency
составляет около минуты на кадр. Следующий этап должен быть направлен на
строгий candidate gating и замену/оптимизацию SAM, после чего необходим новый
полный quality–latency прогон по тому же официальному протоколу.
