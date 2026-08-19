# Единый протокол расчёта метрик SMIYC

Статус документа: **обязательный командный протокол**.

Цель: все участники команды должны получать сопоставимые метрики для любых
моделей на публичных validation-разметках Segment Me If You Can (SMIYC).
Формулы нельзя переimplementировать в коде модели или заменять аналогами из
`sklearn`, `torchmetrics` и других библиотек. Единственный источник итоговых
значений — зафиксированная версия официального `road-anomaly-benchmark`.

## 1. Что именно сравниваем

Обязательные датасеты и имена split:

| Каталог | Официальное имя split | Кадров |
|---|---|---:|
| `dataset_AnomalyTrack` | `AnomalyTrack-validation` | 10 |
| `dataset_ObstacleTrack` | `ObstacleTrack-validation` | 30 |

Обязательные итоговые метрики:

| Поле отчёта | Официальное поле | Уровень | Направление |
|---|---|---|---|
| `AUPR` | `PixBinaryClass.area_PRC` | пиксельный | выше лучше |
| `FPR@95` | `PixBinaryClass.tpr95_fpr` | пиксельный | ниже лучше |
| `sIoU GT` | `SegEval-*.sIoU_gt` | компонентный | выше лучше |
| `PPV` | `SegEval-*.prec_pred` | компонентный | выше лучше |
| `mean F1` | `SegEval-*.f1_mean` | компонентный | выше лучше |

В отчёте все пять метрик выражаются в процентах: исходное значение из
диапазона `[0, 1]` умножается на 100. Разности результатов указываются в
**процентных пунктах**, а не в процентах относительного изменения.

Этот протокол относится только к публичным validation split. Результаты test
split без закрытой разметки нельзя называть локальными официальными test
метриками.

## 2. Зафиксированная реализация

Использовать:

```text
RAAS evaluator: Maskomaly/scripts/run_smiyc_eval.py
road-anomaly-benchmark commit: 1c7804e0749686452eb02bbcfb81234e38795f8a
```

Проверка версии:

```bash
git submodule update --init --recursive
git submodule status third_party/road-anomaly-benchmark
```

Ожидаемое начало второй команды:

```text
 1c7804e0749686452eb02bbcfb81234e38795f8a
```

Если hash отличается, результаты нельзя объединять в одну сравнительную
таблицу до пересчёта одной версией evaluator.

Не устанавливать полный frozen `requirements.txt` внешнего benchmark: он
может заменить PyTorch, torchvision и NumPy основного проекта. Для данного
репозитория устанавливаются только минимальные зависимости:

```bash
python -m pip install -r Maskomaly/requirements-smiyc.txt
```

Для строгой воспроизводимости вместе с результатами сохранить:

```bash
git rev-parse HEAD
git submodule status third_party/road-anomaly-benchmark
python --version
python -m pip freeze
nvidia-smi
```

Версии среды влияют на inference. Если уже сохранённые anomaly maps побайтово
одинаковы, CPU-расчёт метрик данным evaluator должен совпадать.

## 3. Контракт предсказания модели

На каждый официальный кадр модель обязана вернуть одну двумерную anomaly map:

```text
shape: H x W
смысл: большее значение = более аномальный пиксель
диапазон после нормализации: [0, 1]
NaN/Inf: запрещены
```

Правила wrapper:

1. Официальный loader отдаёт RGB, а RAAS/Detectron2 получает BGR.
2. Если размер карты не равен размеру исходного кадра, используется bilinear
   resize (`cv2.INTER_LINEAR`).
3. Карта преобразуется в `float32`, проверяется и ограничивается через
   `clip(..., 0, 1)`.
4. `Evaluation.save_output` сохраняет её как **float16 HDF5**.
5. Метрики считаются по сохранённой float16-карте, а не по исходному tensor.

Квантование в float16 является частью принятого протокола. Нельзя одному
методу считать метрики до сохранения по float32, а другому — после сохранения
по float16.

Обработка должна завершиться для всех кадров. Частичный результат запрещено
публиковать. Ожидается ровно 10 файлов для AnomalyTrack и 30 файлов для
ObstacleTrack на каждый `method_name`.

Каждому эксперименту дать уникальное имя метода или отдельный `--output`.
Повторное использование имени может незаметно смешать старые и новые HDF5.

## 4. Ground truth, ROI и агрегирование

После преобразования официальным loader label map имеет значения:

```text
0   = normal / in-distribution / road
1   = anomaly / obstacle
255 = void / ignore
```

Пиксели `255` исключаются. Нельзя самостоятельно заменять ROI прямоугольником,
маской дороги или полным изображением.

Pixel-level confusion matrices суммируются по всем валидным пикселям всего
split. Это **не** среднее метрик по изображениям.

Component-level значения и количества компонент также агрегируются по всему
split. `sIoU GT` усредняется по GT-компонентам, а `PPV` — по предсказанным
компонентам; изображения и компоненты не взвешиваются одинаковой площадью.

## 5. Pixel-level метрики

Запускается только конфигурация `PixBinaryClass`:

```text
bin_strategy = percentiles
num_bins = 768 на кадр
```

Для каждого threshold формируются:

```text
TP = anomaly-пиксель предсказан как anomaly
FP = normal-пиксель предсказан как anomaly
FN = anomaly-пиксель предсказан как normal
TN = normal-пиксель предсказан как normal
```

### AUPR

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
AUPR      = sum((recall[i] - recall[i-1]) * precision[i])
```

Это AP-подобное суммирование, согласованное со стратегией sklearn, а не
трапецеидальная интеграция PR-кривой. Поэтому `np.trapz`, другая сетка
threshold или прямой вызов сторонней AP-функции не являются заменой
официальному результату.

### FPR@95

```text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

Берётся FPR в первой точке кривой, где `TPR >= 0.95`. Если 95% TPR не
достигается, результат равен `1.0` (в отчёте `100%`).

### Best pixel-F1 threshold

Для каждого threshold:

```text
pixel_F1 = 2*TP / (2*TP + FP + FN)
```

Evaluator выбирает threshold с максимальным pixel F1 отдельно для каждой
комбинации `model x dataset` и сохраняет его в результате `PixBinaryClass`.
Само значение best pixel F1 не выводится в нашей основной таблице. Выбранный
threshold затем используется для всех component-level метрик.

Важно: это официальный validation-протокол с оптимальным threshold на том же
validation split. Нельзя подменять его фиксированным `0.5` или общим
threshold для всех моделей. Если нужен отдельный фиксированный operating
point, его следует публиковать как дополнительный, неофициальный эксперимент.

## 6. Преобразование в компоненты

Перед component evaluation карта бинаризуется строго так:

```text
prediction = anomaly_p > best_pixel_f1_threshold
```

Равенство threshold относится к normal (`<= threshold`). Затем из prediction
и GT строятся 8-связные компоненты с ядром `3 x 3` из единиц. Ignore-пиксели
обнуляются и не участвуют в компонентах.

Маленькие компоненты фильтруются официальными track-specific настройками:

| Track | Удалить predicted component, если размер меньше | Игнорировать GT component, если размер меньше |
|---|---:|---:|
| AnomalyTrack | 500 px | 100 px |
| ObstacleTrack | 50 px | 10 px |

Компонента ровно порогового размера сохраняется.

## 7. Component-level метрики

### sIoU GT

Для каждого GT-компонента `G` выбираются все predicted-компоненты, которые с
ним пересекаются. Пусть их объединение — `P(G)`, а `A` — пиксели этих
predicted-компонент, принадлежащие другим GT-компонентам. Тогда:

```text
intersection   = |G intersection P(G)|
adjusted_union = |G| + |P(G)| - intersection - |A|
sIoU(G)        = intersection / adjusted_union
sIoU GT        = mean(sIoU(G)) по всем GT-компонентам split
```

Adjustment нужен, чтобы корректнее учитывать одну predicted-компоненту,
пересекающую несколько GT-объектов.

### PPV

Для каждой predicted-компоненты `P`:

```text
PPV(P) = |пиксели P, попавшие в любой GT-компонент| / |P|
PPV    = mean(PPV(P)) по всем predicted-компонентам split
```

Это невзвешенное среднее по компонентам. Если predicted-компонент нет, PPV
не определён и записывается как `null` / `n/a`, а не как ноль.

### mean F1

`mean F1` — **не pixel F1**. Он считается по компонентам для 11 порогов:

```text
tau = 0.25, 0.30, 0.35, ..., 0.75
TP_tau = число GT-компонент с sIoU >= tau
FN_tau = число GT-компонент с sIoU < tau
FP_tau = число predicted-компонент с PPV < tau
F1_tau = 2*TP_tau / (2*TP_tau + FP_tau + FN_tau)
mean F1 = mean(F1_tau) по 11 значениям tau
```

Поэтому высокий AUPR может сочетаться с существенно меньшим mean F1: AUPR
оценивает ранжирование отдельных пикселей по всем threshold, а mean F1 строго
штрафует пропущенные, неточно очерченные, раздробленные и ложные объекты.

## 8. Эталонная последовательность запуска

Из корня RAAS:

```bash
git submodule update --init --recursive
python -m pip install -r Maskomaly/requirements-smiyc.txt

cd Maskomaly/scripts
python run_smiyc_eval.py \
  --datasets-root /absolute/path/to/datasets \
  --output ../../results/smiyc_official \
  --models maskomaly maskomaly_id maskomaly_ood \
  --phase all
```

Раздельный запуск:

```bash
# GPU: создать prediction HDF5
python run_smiyc_eval.py \
  --datasets-root /absolute/path/to/datasets \
  --output ../../results/smiyc_official \
  --models maskomaly maskomaly_id maskomaly_ood \
  --phase inference

# CPU: повторно посчитать метрики по тем же HDF5
python run_smiyc_eval.py \
  --datasets-root /absolute/path/to/datasets \
  --output ../../results/smiyc_official \
  --models maskomaly maskomaly_id maskomaly_ood \
  --phase metrics
```

Для других моделей допустим свой inference wrapper, но он обязан соблюдать
контракт раздела 3 и сохранять карты через официальный `Evaluation.save_output`.
Стадии метрик и их порядок менять нельзя:

```text
1. PixBinaryClass
2. SegEval-AnomalyTrack или SegEval-ObstacleTrack
```

## 9. Где брать итоговые числа

Единственные файлы для общей сравнительной таблицы:

```text
<output>/summary.csv
<output>/summary.json
```

Официальные промежуточные результаты:

```text
<output>/anomaly_p/<method>/<dataset>/<frame>.hdf5
<output>/PixBinaryClass/data/
<output>/SegEval-AnomalyTrack/data/
<output>/SegEval-ObstacleTrack/data/
```

Не переписывать числа вручную из сокращённого terminal output, если доступен
CSV/JSON. В публикационной таблице округлять только при отображении до двух
знаков; расчёты разностей выполнять из неокруглённых значений CSV/JSON.

## 10. Что приложить к каждому командному результату

Минимальный пакет результата:

1. `summary.csv` и `summary.json`.
2. Git commit модели/проекта.
3. Hash submodule evaluator.
4. Точное имя model/config/checkpoint и checksum checkpoint.
5. Команду запуска и абсолютный dataset root (без копирования приватных
   данных).
6. Число обработанных кадров: `10 + 30` на модель.
7. Лог отсутствия пропущенных HDF5.
8. Для расследования расхождений — `pip freeze`, версия CUDA/GPU и checksum
   prediction HDF5.

Checksum весов:

```bash
sha256sum /path/to/checkpoint
```

Checksum всех prediction-файлов в стабильном порядке:

```bash
find <output>/anomaly_p/<method> -type f -name '*.hdf5' -print0 \
  | sort -z \
  | xargs -0 sha256sum
```

Если prediction checksums совпадают, а метрики различаются, причина находится
в evaluator/environment. Если checksums различаются, сначала исследуется
inference, preprocessing, checkpoint и формат anomaly score.

## 11. Запрещённые способы сравнения

Результат не считается совместимым с этим протоколом, если сделано хотя бы
одно из следующего:

- использован legacy `run_eval.py` с `SMIYCANO`/`SMIYCOBS`;
- файлы взяты не из официального validation split;
- ignore/ROI трактовались вручную;
- RGB передан RAAS/Detectron2 без преобразования в BGR;
- anomaly map инвертирована или не означает «больше = аномальнее»;
- размер карты изменён nearest-neighbor или другим resize вместо bilinear;
- метрики посчитаны по float32 до официального float16 HDF5;
- AUPR/FPR/F1 рассчитаны сторонней библиотекой;
- component threshold установлен в `0.5`;
- использован один общий threshold вместо индивидуального best pixel-F1;
- изменены minimum component sizes или тип связности;
- усреднены per-image метрики вместо глобальной агрегации;
- опубликован результат по неполному числу кадров;
- старые prediction-файлы смешаны с новым экспериментом;
- изменён commit официального evaluator без полного пересчёта всех моделей.

## 12. Короткий текст для передачи другому агенту

Скопируйте блок ниже в задачу агенту:

```text
Считай SMIYC-метрики строго по docs/SMIYC_METRICS_PROTOCOL_RU.md.
Используй только Maskomaly/scripts/run_smiyc_eval.py и официальный
road-anomaly-benchmark commit 1c7804e0749686452eb02bbcfb81234e38795f8a.
Оцени AnomalyTrack-validation (10 кадров) и ObstacleTrack-validation
(30 кадров). Prediction — HxW anomaly score, больше = аномальнее; RGB loader
преобразовать в BGR для RAAS; resize только bilinear; проверить finite;
clip [0,1]; сохранить через Evaluation.save_output как float16 HDF5.
Сначала запусти PixBinaryClass, затем соответствующий SegEval. Не используй
порог 0.5: component metrics используют индивидуальный best pixel-F1 threshold
этой модели и датасета. Не меняй ROI/ignore, размеры компонент и формулы.
Верни summary.csv и summary.json в процентах, commits модели/evaluator,
checksum checkpoint, число кадров и точную команду запуска. Не публикуй
результат, если отсутствует хотя бы один prediction-файл.
```

## 13. Контрольный reference run RAAS

Текущий полный запуск RAAS дал следующие значения в процентах. Они служат
smoke-check, а не заменяют checksums и сведения о среде:

| model | dataset | AUPR | FPR@95 | sIoU GT | PPV | mean F1 |
|---|---|---:|---:|---:|---:|---:|
| `maskomaly` | `AnomalyTrack-validation` | 94.09 | 3.40 | 80.07 | 44.86 | 63.60 |
| `maskomaly` | `ObstacleTrack-validation` | 88.62 | 2.81 | 49.19 | 54.06 | 53.21 |
| `maskomaly_id` | `AnomalyTrack-validation` | 93.18 | 3.42 | 80.10 | 43.43 | 62.26 |
| `maskomaly_id` | `ObstacleTrack-validation` | 91.21 | 0.42 | 55.16 | 53.61 | 58.65 |
| `maskomaly_ood` | `AnomalyTrack-validation` | 93.18 | 3.42 | 80.10 | 43.43 | 62.26 |
| `maskomaly_ood` | `ObstacleTrack-validation` | 91.21 | 0.42 | 55.16 | 53.61 | 58.65 |

Совпадение только округлённых значений не доказывает идентичность inference.
Для строгой проверки сравнивать HDF5 checksums или непосредственно массивы.
