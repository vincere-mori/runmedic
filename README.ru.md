<div align="center">

# 🩺 runmedic

**Доктор для обучения моделей.** Даёшь ему логи, получаешь диагноз с доказательствами и конкретными правками.
Работает как CLI и как скилл для AI-агентов (Claude Code, Codex и другие).

[![CI](https://github.com/vincere-mori/runmedic/actions/workflows/ci.yml/badge.svg)](https://github.com/vincere-mori/runmedic/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

[English](README.md) · **Русский**

</div>

---

Лосс ушёл в NaN в три ночи. Val loss растёт уже второй день. GPU загружена на 40%, и никто не понимает почему. Все кривые есть в TensorBoard или W&B, но чтобы их прочитать, всё ещё нужен опытный глаз.

runmedic читает те же логи и говорит, что сломалось, показывает цифры, которые это доказывают, и что поменять:

```text
$ runmedic diagnose examples/logs/nan

Found: 1 critical, 1 warning

1. [CRITICAL] `train/loss` became NaN/inf at step 324
   · `train/loss` is inf at step 324 (last finite value 7.32e+33 at step 323)
   · grad norm jumped to 4.79e+11 before the NaN (4.85e+08x its median)
   · learning rate at that point: 2.49e-04
   · the learning rate was still rising when it broke: 124.5x its starting value
   Likely causes:
     - The LR schedule never stops increasing (warmup that never ends, wrong `total_steps`, or a custom lambda)
     - Exploding gradients: the learning rate is too high for this point of training
   What to try:
     1. Check the scheduler: print `scheduler.get_last_lr()` over steps and make sure it peaks and then decays
     2. Clip gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
     3. Lower the peak learning rate 2-10x
     4. Resume from a checkpoint before the blow-up instead of restarting from scratch
```

Это настоящее обучение на PyTorch из [examples/](examples/broken_training.py) со сломанным расписанием LR, и отчёт указывает именно на него.

Со скиллом агент идёт дальше: запускает диагностику, открывает твой код обучения, находит строку, из-за которой всё сломалось, предлагает минимальный дифф и проверяет следующий прогон.

## Что ловит

| Проверка | Что значит | Нужно в логах |
|---|---|---|
| `non_finite` | Лосс или норма градиента стали NaN/inf, и что к этому привело: рост лосса, взрыв градиентов, LR, который всё растёт, или баг с первого шага | loss |
| `divergence` | Лосс снижался, а потом пошёл вверх; сколько прогресса потеряно | train loss |
| `no_learning` | Лосс стоит на месте; замечает loss ≈ ln(k) (равномерное предсказание по k классам) и LR = 0 | train loss |
| `loss_spikes` | Резкие скачки относительно тренда и шаги, которые стоит посмотреть; ранние скачки связывает с отсутствием warmup | train loss |
| `overfitting` | Val loss растёт, а train падает; шаг лучшего чекпойнта | train + val loss |
| `val_below_train` | Val стабильно ниже train: обычно нормально, иногда утечка | train + val loss |
| `grad_spikes` | Норма градиента сильно выше медианы, подсказка порога клиппинга | grad norm |
| `constant_lr_plateau` | Лосс встал, а LR не затухает | LR + train loss |
| `input_bound` | GPU простаивает или большая часть шага уходит на ожидание данных | загрузка GPU или время данных/шага |
| `slow_steps` | Медленные шаги-выбросы, в том числе периодические (eval или чекпойнт внутри цикла) | время шага |
| `slowdown` | Скорость обучения падает со временем | throughput |
| `memory_growth` | Память GPU или хоста стабильно растёт (с типичными причинами утечек) | память |
| `log:*` | Известные ошибки в stdout: CUDA OOM, убитый воркер DataLoader / shm, переполнение AMP, таймаут NCCL, тензоры на разных устройствах, повторный backward, in-place операции, неиспользуемые параметры в DDP | текстовый лог |

Пороги специально консервативные: тихий отчёт полезнее шумного, который через неделю перестаёшь читать. Для здорового прогона будет «No problems found».

## Какие логи понимает

| Источник | Что передать |
|---|---|
| TensorBoard | папку с логами или файл `events.out.tfevents*` (свой парсер, TensorFlow не нужен) |
| MLflow | `mlflow.db` (по умолчанию с MLflow 3) или папку `mlruns/` |
| Weights & Biases | ссылку на ран или `wandb:entity/project/run_id` (нужен `pip install "runmedic[wandb]"` и `wandb login`) |
| PyTorch Lightning | `lightning_logs/version_N/` (TensorBoard) или `metrics.csv` |
| HF Transformers Trainer | папку с результатами (`trainer_state.json` последнего чекпойнта) |
| CSV / JSON Lines | любую таблицу со столбцом шага |
| Обычный stdout | `step 100 loss=0.52`, словари `{'loss': 0.52, ...}`, формат nanoGPT `iter 100: loss 2.34`, прогресс-бары Lightning |

В одной папке можно смешивать форматы: runmedic найдёт все раны внутри и проверит каждый. Имена метрик распознаются сами (`train/loss`, `val_loss`, `eval_loss`, `learning_rate`, `grad_norm`, `system/gpu_0_utilization_percentage` и так далее). `runmedic series PATH` покажет, как он их понял, а `--train-loss NAME` и похожие флаги позволяют поправить.

## Установка

```bash
pip install git+https://github.com/vincere-mori/runmedic
```

или без установки:

```bash
uvx --from git+https://github.com/vincere-mori/runmedic runmedic diagnose runs/
```

Ноль зависимостей, Python 3.10+.

## Использование

```bash
runmedic diagnose runs/exp1                 # отчёт для человека
runmedic diagnose mlflow.db --run my-run    # только раны, в имени которых есть "my-run"
runmedic diagnose runs/ --json              # для агентов, скриптов и дашбордов
runmedic diagnose train.log --fail-on critical   # код выхода 1 при критичных находках, для CI
runmedic series runs/exp1                   # какие ряды найдены и как прочитаны
```

## Скилл для агентов

Скилл учит агента всему циклу: найти логи, запустить runmedic, прочитать код обучения, объяснить причину с доказательствами, предложить минимальную правку и проверить её на следующем прогоне. Вместе с ним идёт [плейбук](plugins/runmedic/skills/runmedic/references/playbook.md): как подтвердить и как чинить каждую проблему, плюс сниппет логирования для ранов, где не хватает метрик.

**Установка одной фразой.** Отправь агенту:

```
Установи скилл runmedic по инструкции: https://raw.githubusercontent.com/vincere-mori/runmedic/main/INSTALL.md
```

<details>
<summary>Установить вручную</summary>

**Claude Code:**

```
/plugin marketplace add vincere-mori/runmedic
/plugin install runmedic@runmedic
```

**Codex и другие агенты с поддержкой `SKILL.md`:**

```bash
git clone --depth 1 https://github.com/vincere-mori/runmedic.git
mkdir -p ~/.agents/skills && cp -r runmedic/plugins/runmedic/skills/runmedic ~/.agents/skills/
```

</details>

Дальше просто спрашивай: *«почему у меня взорвалось обучение?»*, *«это переобучение?»*, *«почему GPU загружена на 40%?»*.

## Попробовать на демо

```bash
git clone https://github.com/vincere-mori/runmedic && cd runmedic
pip install -e .
runmedic diagnose examples/logs
```

В `examples/logs/` четыре настоящих прогона на PyTorch: `nan` (сломанное расписание LR), `overfit` (большая модель на маленьком шумном датасете), `slow-data` (медленный `__getitem__` и синхронный чекпойнт каждые 50 шагов) и `healthy`. Пересоздать их можно так: `python examples/broken_training.py <сценарий>` (нужны `torch` и `tensorboard`).

## Как устроено

1. **Загрузка.** Для каждого логгера свой маленький ридер. Файлы TensorBoard разбираются напрямую (формат TFRecord и минимальный декодер protobuf для скалярных метрик), MLflow читается из файлового хранилища или sqlite-базы, W&B через публичный API.
2. **Интерпретация.** Имена рядов сопоставляются с ролями: train loss, val loss, LR, норма градиента, загрузка GPU, память, время шага и данных, throughput.
3. **Проверки.** Каждая проверка это небольшая эвристика над сглаженными кривыми и робастной статистикой (EMA, z-оценки по медиане и MAD, ранговая корреляция). Она ищет одну конкретную проблему и собирает доказательства.
4. **Объяснение.** У каждой находки есть доказательства, вероятные причины в порядке того, насколько их поддерживают данные, и правки.

Внутри нет ML-модели, поэтому всё быстро, детерминированно и работает офлайн.

## Ограничения

- runmedic читает кривые, а не код. Он говорит, *что* произошло и *почему* скорее всего; подтвердить причину в коде это задача скилла или твоя.
- Качество диагноза зависит от того, что логируется. Без нормы градиента и LR отчёт про NaN менее конкретный, без времени шага и данных не видно узкого места в загрузке. В плейбуке есть сниппет логирования.
- Офлайн-раны W&B (`wandb/offline-run-*`) пока не читаются напрямую: синхронизируй их или выгрузи историю в CSV.

## Похожие работы

Исследовательские инструменты вроде AutoTrainer (ICSE 2021) и TrainCheck (OSDI 2025) ищут проблемы обучения, встраиваясь в сам процесс обучения. runmedic работает постфактум по уже существующим логам, с любым из популярных логгеров, и сделан так, чтобы передавать находки агенту, который потом чинит код.

## Как помочь

Самое ценное это баг-репорт с логом, который runmedic продиагностировал неправильно: открой issue с раном (или CSV-выгрузкой метрик) и тем, что на самом деле было не так. Новые проверки и форматы логов тоже приветствуются; для каждой проверки нужен тест с синтетической кривой, на которой она срабатывает, и с той, на которой молчит.

```bash
pip install -e . pytest tensorboard ruff
pytest && ruff check . && ruff format --check .
```

## Лицензия

MIT
