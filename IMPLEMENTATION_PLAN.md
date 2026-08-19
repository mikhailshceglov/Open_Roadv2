# План интеграции Objectomaly, road-aware proposals и CLIP validation

## Статус аудита

Аудит RAAS выполнен для ветки `SAM+road-aware+CLIP` на commit `b3e978d`.
Официальный Objectomaly дополнительно проверен на commit
`66d2ad2a1b02d79389f4265d9d1d99ab6412324f` от 2026-07-19. На этапе
аудита функциональный код RAAS не изменялся и Objectomaly в рабочее дерево
не копировался.

### Статус реализации

Cache-first milestone реализован в рабочем дереве:

- `third_party/Objectomaly` добавлен как submodule и закреплён на
  `66d2ad2a1b02d79389f4265d9d1d99ab6412324f`;
- `export_objectomaly_inputs.py` сохраняет lossless BGR и `float32` coarse
  maps с полным manifest;
- `run_objectomaly_refinement.py` вызывает официальный `SAMMaskGenerator`,
  `postprocess`, `apply_oasc` и `apply_mbp`, поддерживает повторный refinement
  по сохранённым SAM masks;
- `import_objectomaly_outputs.py` импортирует refined maps через официальный
  `Evaluation.save_output` и считает неизменённые SMIYC metrics;
- `run_objectomaly_infer.py` запускает тот же cache-first pipeline на
  произвольной папке без GT и сохраняет карты, heatmaps, overlays, binary
  masks и side-by-side comparisons;
- `global_fusion.py` применяет soft semantic priors вместо hard road ROI,
  сохраняет компактные SAM candidates во всех зонах кадра и валидирует их
  batched CLIP prompt ensemble для road/airborne OOD;
- добавлены strict cache-contract tests и инструкция по лицензированию.

Следующий milestone после серверного baseline — road-aware/global CLIP fusion
между `apply_oasc` и `apply_mbp`.

### Главный вывод

В текущем checkout реализован RAAS-owned cache bridge к официальному
Objectomaly submodule; upstream-код не скопирован и не изменён. Это важно,
поскольку upstream не содержит project-level `LICENSE`, а его окружение
(Python 3.10, PyTorch 2.1, torchvision 0.16) конфликтует с текущим RAAS
(Python 3.8, PyTorch 1.9, torchvision 0.10).

Рекомендуемый способ интеграции:

1. pinned git submodule `third_party/Objectomaly` на указанном commit;
2. сначала cache-first bridge в отдельном окружении Objectomaly;
3. RAAS сохраняет initial anomaly maps, Objectomaly сохраняет SAM masks и
   refined maps, официальный SMIYC evaluator RAAS считает итоговые метрики;
4. после фиксации baseline — тонкий RAAS adapter для end-to-end запуска без
   копирования и правки upstream-кода.

До публикации/распространения объединённого решения нужно получить у авторов
Objectomaly явную лицензию или разрешение. Submodule сохраняет происхождение
кода, но сам по себе не устраняет отсутствие лицензии.

## 1. Реальные inference/evaluation entry points

### Folder inference

Файл: `Maskomaly/scripts/infer.py`.

Поток выполнения:

```text
main()
  -> load_model(model_alias, args)
  -> run_inference(model, image, output_debug, stem, debug)
  -> model.get_soft_mask(...)
  -> soft_mask / heatmap / optional query masks
```

Особенности:

- вход читается `cv2.imread`, то есть BGR;
- model instance создаётся один раз до frame loop;
- доступны варианты `maskomaly`, `maskomaly_id`, `maskomaly_ood`;
- `--debug` передаёт `output_base_dir` и `filename` только моделям, в
  сигнатуре которых эти аргументы присутствуют;
- базовый `maskomaly` возвращает `(soft_mask, query_masks)`, а ID/OOD варианты
  возвращают только `soft_mask`;
- здесь логично подключать новый orchestrator только после появления
  Objectomaly, сохранив старые aliases без изменения поведения.

### Legacy dataset evaluation

Файл: `Maskomaly/scripts/run_eval.py`.

Поток аналогичен folder inference, но использует dataset classes из
`Maskomaly/maskomaly/datasets.py`. Этот evaluator рассчитывает собственные
pixel metrics через `Maskomaly/scripts/eval.py`; он не является официальным
SMIYC evaluator и не должен использоваться для итогового сравнения SMIYC.

Цикл намеренно продолжает работу после ошибки отдельного кадра. Для
официального сравнения такое поведение недопустимо.

### Official SMIYC evaluation

Файл: `Maskomaly/scripts/run_smiyc_eval.py`.

Поток выполнения:

```text
execute()
  -> load_model()
  -> run_inference()
  -> model.get_soft_mask(BGR)
  -> prepare_anomaly_map()
  -> official Evaluation.save_output()
  -> PixBinaryClass
  -> SegEval-AnomalyTrack / SegEval-ObstacleTrack
```

Это обязательная точка итогового сравнения нового pipeline. При добавлении
Objectomaly понадобится только зарегистрировать новый model alias и создать
его instance через существующий `load_model`; формулы evaluator менять нельзя.
Правила метрик зафиксированы в `docs/SMIYC_METRICS_PROTOCOL_RU.md`.

## 2. CAS и initial anomaly map

В RAAS initial Maskomaly map создаётся внутри `Maskomaly.get_soft_mask`:

- `Maskomaly/maskomaly/model_ori.py::Maskomaly.get_soft_mask`;
- `Maskomaly/maskomaly/model_id.py::Maskomaly.get_soft_mask`;
- `Maskomaly/maskomaly/model_ood.py::Maskomaly.get_soft_mask`.

Текущая схема:

```text
Mask2Former query class logits -> softmax
Mask2Former query mask logits  -> sigmoid
rejection map + acceptance map
0.6 * rejection + 0.4 * acceptance
resize to input size
```

Официальный адаптерный контракт Objectomaly:

- `objectomaly/backbones/base.py::AnomalyBackboneAdapter.anomaly_score`;
- вход: BGR `uint8` изображение `(H, W, 3)`;
- выход: `float32` карта `(H, W)` в `[0, 1]`, большее значение означает OOD.

`objectomaly/core/cas.py::CoarseAnomalyScoring.__call__` вызывает backbone,
опционально выполняет horizontal/vertical flip TTA и усредняет карты. Его
default включает horizontal flip. Поэтому для строгого сравнения с текущим
Maskomaly baseline `use_tta_hflip` необходимо выключить либо считать TTA
отдельным вариантом эксперимента.

Upstream `objectomaly/backbones/maskomaly.py::MaskomalyBackbone` читает только
предвычисленные карты и не умеет запускать RAAS. Для live-режима нужен
маленький RAAS-owned adapter вокруг уже созданного экземпляра
`model_ori.Maskomaly`, реализующий `anomaly_score`, без повторной загрузки
Mask2Former.

## 3. Доступные Mask2Former outputs

### Где они создаются

`Mask2Former/mask2former/mask2former_model.py::MaskFormer.forward` возвращает
в inference режиме три значения:

```text
processed_results
mask_cls_results = outputs["pred_logits"]
mask_pred_results = outputs["pred_masks"] после bilinear upsampling
```

В репозитории также есть копия
`Maskomaly/mask2former_replacements/mask2former/mask2former_model.py` с тем же
трёхзначным контрактом. Перед будущим refactor следует удалить неопределённость
import precedence тестом, но на этапе аудита файлы не меняются.

### Как outputs доходят до модели

`Maskomaly/detectron2_replacements/detectron2/engine/defaults.py::DefaultPredictor.__call__`
возвращает для одного изображения:

```text
predictions[0], mask_cls_results[0], mask_pred_results[0]
```

Затем `BaseSegmentationModel.get_probs_and_seg` в каждом `model_*.py`:

- применяет softmax к class logits;
- применяет sigmoid к mask logits;
- переводит оба массива на CPU/NumPy;
- возвращает их вместе с `segmentation`, где доступен `sem_seg`.

Таким образом, query masks и class scores уже доступны до построения initial
anomaly map. Это правильная точка данных для road-aware ветки. Желательно не
извлекать их повторным прогоном Mask2Former.

## 4. Существующая road-aware ветка

Road-aware логика уже реализована, но продублирована в:

- `Maskomaly/maskomaly/model_id.py::Maskomaly.get_soft_mask`;
- `Maskomaly/maskomaly/model_ood.py::Maskomaly.get_soft_mask`.

Текущая последовательность:

```text
query mask с hardcoded index 20
  -> threshold 0.5
  -> resize nearest-neighbor
  -> cv2.findContours(RETR_EXTERNAL)
  -> cv2.fillPoly(all external contours)
  -> road_gap = filled_polygon AND NOT road_mask
  -> cv2.connectedComponents
  -> bbox crop каждого gap
  -> CLIP validation
  -> overwrite soft_mask значением 0.05 или 1.0
```

Текущие ограничения, которые нужно устранять отдельными маленькими commit
после фиксации Objectomaly baseline:

- ветка всегда включена для ID/OOD моделей, config flag отсутствует;
- `road_index=20`, threshold `0.5`, CLIP thresholds и output scores hardcoded;
- нет minimum component area, только проверка bbox `w/h > 1`;
- нет configurable context padding;
- решения по компонентам не сохраняются в структурированный JSON/CSV;
- road polygon/gap и CLIP код продублированы между ID и OOD вариантами;
- road query выбирается по фиксированному номеру query, а не через явно
  проверенный контракт конкретного checkpoint;
- debug layout различается между `model_id.py` и `model_ood.py`.

Эту логику следует сохранить как отдельный источник candidates. Точное место
интеграции в RAAS-owned orchestrator — после
`objectomaly/refinement/oasc.py::apply_oasc` и до
`objectomaly/refinement/mbp.py::apply_mbp`; query outputs берутся из того же
единственного прогона Mask2Former.

## 5. Существующий CLIP

OpenAI CLIP импортируется напрямую как `clip` в `model_id.py` и
`model_ood.py`.

Что уже сделано правильно:

- `ViT-B/32` загружается один раз в `Maskomaly.__init__`;
- tokenized prompts создаются один раз в `__init__`;
- CLIP не загружается внутри frame/component loop;
- используется `torch.no_grad()`.

Что отсутствует:

- общий backend: ID и OOD модели загружают отдельную реализацию логики;
- batching region crops;
- явный config устройства;
- `torch.inference_mode()`;
- prompt config вне Python;
- masked crop + context crop;
- структурированный результат с top prompts/scores;
- SAM-region validation;
- global normal suppression;
- road protection при global suppression.

`Maskomaly/environment.yml` не содержит Segment Anything. OpenAI CLIP также
не закреплён в environment-файле; README предлагает отдельную установку
официального GitHub package. `PyYAML` и `OmegaConf` уже присутствуют.

## 6. Device и preprocessing

### Mask2Former

- device определяется Detectron2 config через `MODEL.DEVICE`;
- его можно переопределить через существующий `--opts`;
- `DefaultPredictor` принимает BGR и преобразует каналы согласно
  `cfg.INPUT.FORMAT`;
- используемый Cityscapes config задаёт `INPUT.FORMAT: RGB`, поэтому wrapper
  принимает BGR и внутри меняет его на RGB;
- текущий predictor использует `torch.no_grad()`, не `torch.inference_mode()`.

### CLIP

В `model_id.py` и `model_ood.py` device выбирается так:

```text
cuda, если torch.cuda.is_available(), иначе cpu
```

Выбрать отдельный device или CUDA index через config сейчас нельзя. Новый
`SemanticValidator` должен получать resolved device от единого pipeline
config, но не должен менять device существующего pipeline при выключенных
новых flags.

### SAM

Официальная реализация находится в
`objectomaly/masks/sam.py::SAMMaskGenerator`:

- `generate(image_bgr) -> MaskBundle`;
- lazy initialization загружает модель один раз;
- поддержаны `vit_h`, `vit_l`, `vit_b`, `sam2`, `fastsam`;
- стандартный reproducibility config использует SAM ViT-H;
- default automatic-mask параметры: `points_per_side=32`,
  `pred_iou_thresh=0.88`, `stability_score_thresh=0.95`.

`objectomaly/masks/base.py::MaskBundle` хранит `(N,H,W)` boolean masks,
quality, area, bbox, generator, timings и extra metadata. Перед OASC должен
вызываться `objectomaly/masks/postprocess.py::postprocess`: filter, mask-IoU
NMS и overlap partition. Базовые значения upstream: `q_min=0.7`,
`area_frac_min=0.0005`, `area_frac_max=0.5`, `nms_iou=0.7`, partition
`small_to_large`.

## 7. Точная карта этапов Objectomaly

| Этап | Официальный интерфейс | Контракт |
|---|---|---|
| End-to-end (legacy/core) | `objectomaly/core/pipeline.py::ObjectomalyPipeline.__call__` | BGR image → `PipelineOutput` |
| CAS | `objectomaly/core/cas.py::CoarseAnomalyScoring.__call__` | BGR image → coarse `(H,W)` |
| SAM | `objectomaly/masks/sam.py::SAMMaskGenerator.generate` | BGR image → `MaskBundle` |
| mask filtering/partition | `objectomaly/masks/postprocess.py::postprocess` | raw `MaskBundle` → disjoint filtered bundle |
| OASC (experiment API) | `objectomaly/refinement/oasc.py::apply_oasc` | coarse + `MaskBundle` + variant → calibrated map |
| MBP/BBRR (experiment API) | `objectomaly/refinement/mbp.py::apply_mbp` | calibrated map + `MaskBundle` + variant → refined map |
| Mask2Former outputs RAAS | `BaseSegmentationModel.get_probs_and_seg` | query masks/scores |

Upstream содержит два параллельных API. `core/ObjectomalyPipeline` использует
legacy `List[np.ndarray]`, из-за чего теряет quality metadata и не вызывает
полный `postprocess`. Reproducibility experiment configs используют более
новый API `refinement.apply_oasc/apply_mbp` с вариантами
`quality_aware_residual_blending` и `boundary_band_residual`. В RAAS нельзя
случайно смешать эти два протокола; для интеграции выбран современный
experiment API.

## 8. Предлагаемая конечная архитектура

Целевой data flow в RAAS-owned orchestrator:

```text
Mask2Former once -> RAAS Maskomaly adapter -> coarse map
SAMMaskGenerator.generate -> MaskBundle -> postprocess
apply_oasc(variant=quality_aware_residual_blending)
  -> suspicious SAM regions
  -> road proposal: road mask -> filled polygon -> gaps
  -> shared SemanticValidator (loaded once)
       -> RoadGapValidator
       -> GlobalNormalSuppressor
  -> fusion with road protection
  -> apply_mbp(variant=boundary_band_residual)
  -> final anomaly map
```

Требования к реализации:

- Mask2Former, SAM и CLIP загружаются по одному разу на pipeline instance;
- Mask2Former не запускается отдельно для Objectomaly и road-aware веток;
- text embeddings вычисляются один раз;
- все новые ветки выключены по умолчанию;
- при обоих flags `false` результат должен совпадать с зафиксированным
  Objectomaly experiment baseline до допустимой численной погрешности;
- MBP вызывается только после semantic/road fusion;
- старые aliases Maskomaly продолжают работать;
- официальный SMIYC evaluator остаётся неизменным.

## 9. Предлагаемые файлы интеграции

```text
Maskomaly/configs/objectomaly_fusion.yaml       # flags и thresholds
Maskomaly/configs/prompts.yaml                  # normal/ID/OOD prompts
Maskomaly/maskomaly/road_aware.py               # только geometry/proposals
Maskomaly/maskomaly/semantic_validator.py       # одна CLIP model + embeddings
Maskomaly/maskomaly/semantic_fusion.py          # road protection/suppression
Maskomaly/maskomaly/debug_artifacts.py           # PNG + decisions.json
Maskomaly/maskomaly/objectomaly_adapter.py       # RAAS backbone + orchestrator
Maskomaly/scripts/export_objectomaly_inputs.py   # coarse maps/cache manifest
Maskomaly/scripts/import_objectomaly_outputs.py  # refined maps для evaluator
Maskomaly/tests/test_road_aware.py               # pure NumPy/OpenCV tests
Maskomaly/tests/test_semantic_fusion.py          # pure math/mock CLIP tests
Maskomaly/tests/test_pipeline_disabled.py        # baseline equivalence
third_party/Objectomaly                          # pinned upstream submodule
```

Не следует создавать отдельные `RoadClipModel` и `GlobalClipModel`.
`SemanticValidator` должен владеть одной CLIP model и всеми precomputed text
embeddings. Road/global policy должны быть обычными consumers этого backend.
Upstream submodule не модифицируется; адаптация и новая policy принадлежат
RAAS.

## 10. Предлагаемый config contract

Этот config не добавляется на этапе аудита. Стартовая схема для следующего
инфраструктурного commit:

```yaml
objectomaly:
  source_commit: 66d2ad2a1b02d79389f4265d9d1d99ab6412324f
  integration_mode: cache
  cas:
    use_tta_hflip: false
    use_tta_vflip: false
  masks:
    generator: sam
    variant: vit_h
    points_per_side: 32
    pred_iou_thresh: 0.88
    stability_score_thresh: 0.95
    q_min: 0.7
    area_frac_min: 0.0005
    area_frac_max: 0.5
    nms_iou: 0.7
    partition_order: small_to_large
  oasc:
    variant: quality_aware_residual_blending
  mbp:
    variant: boundary_band_residual

road_aware:
  enabled: false
  road_query_index: 20
  road_mask_threshold: 0.5
  min_component_area: 0
  context_padding: 0.25
  id_threshold: 0.85
  anomaly_score: 1.0
  normal_score: 0.05

global_validator:
  enabled: false
  backend: openai_clip
  model: ViT-B/32
  device: auto
  batch_size: 1
  candidate_threshold: 0.6
  context_padding: 0.25
  crop_fusion: mean
  road_protection_threshold: 0.1
  suppression_strength: 0.9
  road_protected_suppression_strength: 0.15

debug:
  enabled: false
  save_images: true
  save_decisions: true
```

Текущие hardcoded значения вынесены как defaults только для сохранения
существующего road-aware поведения. Они не объявляются оптимальными для
Objectomaly.

## 11. Точки будущей интеграции

### Road-aware proposal

Входы:

```text
original RGB/BGR with explicit color contract
query mask probabilities
query class probabilities
original H/W
```

Выход должен быть структурой, а не только mask:

```text
road_mask
filled_road_polygon
road_gap
components: id, bbox, area, mask
```

Он создаётся из уже имеющихся query outputs до MBP и не должен менять CAS,
SAM или OASC.

### Global semantic validation

После OASC для каждой SAM mask `S_k` сначала вычисляется:

```text
alpha_k = mean(oasc_anomaly_map[S_k])
```

Если `alpha_k < candidate_threshold`, CLIP не вызывается. Иначе формируются
masked/tight и context crops, после чего общий backend возвращает normal
scores и top prompts.

Road protection:

```text
r_k = area(S_k AND validated_road_anomaly) / area(S_k)
```

При `r_k >= road_protection_threshold` применяется только слабое подавление.
В остальных случаях:

```text
A_new(x) = A_old(x) * (1 - suppression_strength * p_normal)
```

Для protected mask используется
`road_protected_suppression_strength`. Нельзя бинарно обнулять SAM region.

### Fusion и MBP

Новая policy не встраивается внутрь legacy
`objectomaly/core/pipeline.py::ObjectomalyPipeline`, поскольку тот теряет
`MaskBundle` metadata. RAAS-owned orchestrator явно вызывает:

```text
RAAS Maskomaly adapter
  -> SAMMaskGenerator.generate
  -> masks.postprocess
  -> refinement.oasc.apply_oasc
  -> road/global semantic fusion
  -> refinement.mbp.apply_mbp
```

Это сохраняет официальный код OASC/MBP неизменным и даёт одну явную точку
между ними для road-aware/CLIP fusion.

## 12. Debug contract

Одна debug-директория на кадр:

```text
<output>/<frame>/
  rgb.jpg
  01_cas.png
  02_oasc.png
  03_road_mask.png
  04_road_polygon.png
  05_road_gap.png
  06_road_validated.png
  07_global_candidates.png
  08_global_clip_classes.png
  09_after_suppression.png
  10_final_mbp.png
  decisions.json
```

Если этап выключен, writer должен либо пропустить его, либо явно записать
статус `disabled` в `decisions.json`; запрещено сохранять старый файл от
предыдущего запуска под видом нового результата.

Для каждой region/component сохранять:

```text
source (sam или road_gap)
region/component id
bbox
area
mean anomaly score до fusion
road overlap
top prompts и scores
decision
applied suppression/output score
```

## 13. Последовательность маленьких commit

### Commit 0 — pinned source и cache-first baseline

- после проверки лицензии добавить `third_party/Objectomaly` как submodule на
  commit `66d2ad2a1b02d79389f4265d9d1d99ab6412324f`;
- не копировать файлы Objectomaly в Maskomaly и не менять submodule;
- создать отдельное окружение Objectomaly, не обновляя PyTorch в RAAS;
- зафиксировать SAM ViT-H checkpoint checksum;
- RAAS сохраняет coarse maps как `float32 .npy` и manifest с frame id/shape;
- Objectomaly сохраняет postprocessed `MaskBundle` cache и refined maps;
- зафиксировать hashes/NPZ baseline на небольшом наборе кадров.

Acceptance: число coarse/refined maps равно числу кадров; shapes и диапазон
совпадают; повторный refinement по cache детерминирован; baseline считается
официальным SMIYC evaluator без изменения формул.

### Commit 1 — RAAS adapter и config, output unchanged

- добавить typed/default-validated config loader;
- добавить `MaskomalyBackboneAdapter.anomaly_score(image_bgr)`;
- добавить RAAS-owned orchestrator с явными calls `postprocess`, `apply_oasc`,
  fusion hook и `apply_mbp`;
- добавить flags `road_aware.enabled=false` и
  `global_validator.enabled=false`;
- передать config в pipeline без изменения вычислений;
- добавить regression test disabled pipeline.

Acceptance: старый config и оба выключенных flags дают baseline output.

### Commit 2 — pure road geometry и debug

- извлечь road mask/polygon/gap в `road_aware.py`;
- не подключать CLIP/global suppression;
- добавить mask unit tests;
- сохранить пять визуализаций геометрии.

Acceptance на сервере: визуально проверены highway, cars, small obstacles и
night frames; baseline не меняется при выключенном flag.

### Commit 3 — shared SemanticValidator

- одна OpenAI CLIP `ViT-B/32` model;
- precomputed prompt embeddings;
- configurable device/batch size;
- mock backend для CPU unit tests;
- пока не влиять на anomaly map.

Acceptance: одна загрузка model на process, deterministic mock tests,
работающий CPU fallback.

### Commit 4 — CLIP validation road gaps

- connected components, min area, context padding;
- structured decisions и visualizations;
- configurable low/high scores;
- включение только через `road_aware.enabled`.

Acceptance: проверены component decisions и road-only metrics/heatmaps.

### Commit 5 — global SAM-region normal suppression

- использовать настоящие SAM masks после OASC;
- candidate threshold по `alpha_k`;
- masked + context crops;
- normal prompts из YAML;
- road protection `r_k`;
- мягкое multiplicative suppression.

Acceptance: sky/noise barriers подавляются, защищённые road obstacles не
удаляются; имеются mock tests для всех ветвей policy.

### Commit 6 — fusion перед существующим MBP

- подключить обе ветки в Objectomaly orchestrator;
- выполнить MBP после fusion;
- унифицировать debug artifact layout;
- зарегистрировать новый alias в folder inference и official SMIYC runner.

Acceptance: full pipeline на 10–20 серверных кадрах, затем официальный SMIYC
run по `docs/SMIYC_METRICS_PROTOCOL_RU.md`.

## 14. Локальные и серверные проверки

Локально, без тяжёлых моделей:

- config parsing/defaults/backward compatibility;
- binary mask geometry;
- connected component filtering;
- crop/padding/color conversion;
- road overlap `r_k`;
- suppression/fusion math;
- mock CLIP batching/decisions;
- debug JSON schema;
- disabled-output regression fixture.

На сервере по checkpoints:

1. исходный Objectomaly baseline;
2. road polygon;
3. road gaps;
4. road gap + CLIP;
5. SAM regions + global CLIP;
6. full fusion + MBP;
7. официальный SMIYC evaluation.

## 15. Решения перед функциональной интеграцией

Уже установлено:

1. официальный source: `https://github.com/hon121215/Objectomaly`;
2. проверенный commit: `66d2ad2a1b02d79389f4265d9d1d99ab6412324f`;
3. целевые enhanced-варианты reproducibility matrix: SAM ViT-H, OASC
   `quality_aware_residual_blending`, MBP/BBRR `boundary_band_residual`;
   рядом с ними upstream обязательно прогоняет original-варианты как baseline;
4. upstream Maskomaly backbone работает только с precomputed maps;
5. полный upstream requirements нельзя устанавливать в окружение RAAS.

До Commit 0 остаются организационные решения:

1. получить/подтвердить разрешение на использование Objectomaly: в upstream
   отсутствует project-level LICENSE;
2. указать путь к SAM ViT-H checkpoint и зафиксировать его SHA-256;
3. выбрать 10–20 фиксированных кадров для regression/debug проверки;
4. подтвердить сохранение `maskomaly`, `maskomaly_id`, `maskomaly_ood` как
   публичных aliases (рекомендация: сохранить);
5. подтвердить cache-first integration как первый воспроизводимый milestone.

После этого можно добавлять submodule и cache bridge. Vendor-копия и установка
Objectomaly requirements поверх `raas` environment исключены.
